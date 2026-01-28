"""
Script to process mesh output directory and create merged point clouds
from handle and head masks across all camera views, visualized with viser.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image


def convert_depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """Convert depth to meters (handles both mm and m inputs)."""
    # If the max value is greater than 100, then it's likely in mm
    in_mm = depth.max() > 100
    if in_mm:
        return depth / 1000.0
    else:
        return depth


def depth_to_points(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
    max_depth_m: float = np.inf,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert depth image to 3D points in camera frame.
    
    Args:
        depth_m: Depth image in meters (H, W)
        K: Camera intrinsic matrix (3, 3)
        rgb: Optional RGB image (H, W, 3)
        mask: Optional binary mask (H, W) - only include points where mask is True
        stride: Subsampling stride
        max_depth_m: Maximum depth to include
    
    Returns:
        pts_c: Points in camera frame (N, 3)
        cols: Colors (N, 3) or None
    """
    h, w = depth_m.shape
    v_coords, u_coords = np.indices((h, w))
    
    if stride > 1:
        v_coords = v_coords[::stride, ::stride]
        u_coords = u_coords[::stride, ::stride]
        depth = depth_m[::stride, ::stride]
        if rgb is not None:
            colors = rgb[::stride, ::stride, :]
        else:
            colors = None
        if mask is not None:
            mask_strided = mask[::stride, ::stride]
        else:
            mask_strided = None
    else:
        depth = depth_m
        colors = rgb
        mask_strided = mask

    z = depth.reshape(-1)
    valid = (z > 0.0) & (z < max_depth_m)
    
    # Apply mask if provided
    if mask_strided is not None:
        mask_flat = mask_strided.reshape(-1)
        valid = valid & mask_flat

    z = z[valid]
    u = u_coords.reshape(-1)[valid]
    v = v_coords.reshape(-1)[valid]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pts_c = np.stack([x, y, z], axis=1)

    if colors is not None:
        cols = colors.reshape(-1, 3)[valid]
    else:
        cols = None
        
    return pts_c, cols


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform points from camera frame to world frame."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t[None, :]


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: (3, 3) rotation matrix
    
    Returns:
        Tuple of (w, x, y, z) quaternion components
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (w, x, y, z)


def compute_handle_frame(
    handle_origin: np.ndarray,
    head_origin: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Compute a coordinate frame with origin at handle_origin and x-axis pointing toward head_origin.
    
    The y and z axes are constructed to be orthogonal to x while staying as aligned
    as possible with the original world axes.
    
    Args:
        handle_origin: (3,) position of handle centroid
        head_origin: (3,) position of head centroid
    
    Returns:
        T_W_H: (4, 4) transformation matrix (world from handle frame)
        quaternion: (w, x, y, z) quaternion representing the orientation
    """
    # X-axis: direction from handle to head
    x_axis = head_origin - handle_origin
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Find which original axis is most perpendicular to x_axis
    # to use as a reference for constructing y and z
    world_axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    dots = [abs(np.dot(x_axis, ax)) for ax in world_axes]
    # Pick the axis most perpendicular to x (smallest dot product)
    perp_idx = np.argmin(dots)
    ref_axis = world_axes[perp_idx]
    
    # Z-axis: perpendicular to x and reference axis
    z_axis = np.cross(x_axis, ref_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Y-axis: perpendicular to x and z (right-hand rule)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build rotation matrix (columns are the new axes in world frame)
    R_W_H = np.column_stack([x_axis, y_axis, z_axis])  # World from Handle frame
    
    # Build 4x4 transform: T_W_H (world from handle)
    T_W_H = np.eye(4)
    T_W_H[:3, :3] = R_W_H
    T_W_H[:3, 3] = handle_origin
    
    # Convert rotation matrix to quaternion
    quaternion = rotation_matrix_to_quaternion(R_W_H)
    
    return T_W_H, quaternion


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    return np.array(Image.open(path))


def load_mask(path: Path) -> np.ndarray:
    """Load mask as boolean array."""
    mask = np.array(Image.open(path))
    # Handle different mask formats
    if len(mask.shape) == 3:
        mask = mask[..., 0]  # Take first channel
    return mask > 0  # Convert to boolean


def save_mesh(mesh: trimesh.Trimesh, name: str, parent_dir: Path):
    """
    Converts a vertex-colored mesh to a textured mesh (obj+mtl+png)
    and saves it in its own subdirectory to avoid file conflicts.
    """
    # 1. Create a subdirectory for this specific mesh
    # e.g. output/original_mesh/
    save_dir = parent_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Export as .glb
    file_path = save_dir / f"{name}.glb"
    mesh.export(file_path)

    # 3. Export as .obj
    # This will produce: '{name}.obj', 'material_0.mtl', 'material_0.png'
    new_mesh = trimesh.load(file_path)
    new_mesh.export(save_dir / f"{name}.obj", file_type="obj")
    return file_path



@dataclass
class ProcessMeshConfig:
    """Configuration for mesh processing."""
    
    output_dir: Path
    """Directory containing mesh/mesh.obj, rgb/, depth/, cam_K.txt, handle_masks/, head_masks/, cam_poses.npy"""
    
    point_size: float = 0.002
    """Point size for visualization"""
    
    max_depth_m: float = 10.0
    """Maximum depth in meters to include"""
    
    subsample_stride: int = 1
    """Stride for subsampling points"""
    
    def __post_init__(self) -> None:
        assert self.output_dir.exists(), f"Output directory not found: {self.output_dir}"

def visualize_bbox(
    server: viser.ViserServer,
    bbox_size: np.ndarray,
    position: np.ndarray,
    wxyz: np.ndarray,
    name: str,
    radius: float = 0.005,
    color: Tuple[int, int, int] = (255, 0, 0),
    add_corner_keypoints: bool = False,
    add_dotted_lines: bool = False,
    dot_gap: float = 0.005,
    add_lines: bool = True,
    line_thickness: float = 0.001,
) -> None:
    """
    Visualize a bounding box in viser.
    
    Args:
        server: Viser server instance
        bbox_size: Size of the bounding box (x, y, z)
        position: Position of the bounding box center
        wxyz: Quaternion orientation (w, x, y, z)
        name: Name for the scene elements
        radius: Radius of corner spheres
        color: RGB color tuple
        add_corner_keypoints: If True, add spheres at corners
        add_dotted_lines: If True, add small spheres along edges
        dot_gap: Gap between dots along edges (in world units)
        add_lines: If True, add solid cylinder lines along edges
        line_thickness: Radius of the cylinder lines
    """
    # HACK
    HACK_CHANGE_BBOX_SIZE = False
    if HACK_CHANGE_BBOX_SIZE:
        bbox_size = np.copy(bbox_size)
        bbox_size[0] *= 1.2

    # Visualize the bounding box in viser
    server.scene.add_frame(
        f"/{name}",
        wxyz=wxyz,
        position=position,
        # axes_length=0.3,
        # axes_radius=0.01,
        axes_length=0.0,
        axes_radius=0.0,
    )
    server.scene.add_box(
        f"/{name}/bbox",
        color=color,
        dimensions=bbox_size,
        visible=True,
    )
    
    # Define corner offsets in handle frame (bbox centered at origin)
    half_size = bbox_size / 2
    corner_offsets = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],  # 0
        [-half_size[0], -half_size[1], +half_size[2]],  # 1
        [-half_size[0], +half_size[1], -half_size[2]],  # 2
        [-half_size[0], +half_size[1], +half_size[2]],  # 3
        [+half_size[0], -half_size[1], -half_size[2]],  # 4
        [+half_size[0], -half_size[1], +half_size[2]],  # 5
        [+half_size[0], +half_size[1], -half_size[2]],  # 6
        [+half_size[0], +half_size[1], +half_size[2]],  # 7
    ])
    
    # Add spheres at the corners
    if add_corner_keypoints:
        for i, offset in enumerate(corner_offsets):
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            server.scene.add_mesh_simple(
                f"/{name}/bbox_corner_{i}",
                vertices=sphere.vertices,
                faces=sphere.faces,
                color=color,
                position=offset,
            )
    
    # Add dotted lines along edges
    if add_dotted_lines:
        # Define the 12 edges of the box as pairs of corner indices
        edges = [
            # Bottom face edges
            (0, 1), (0, 2), (1, 3), (2, 3),
            # Top face edges
            (4, 5), (4, 6), (5, 7), (6, 7),
            # Vertical edges connecting bottom and top
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        dot_radius = radius * 0.4  # Smaller radius for edge dots
        
        # Always add corner dots when dotted lines are enabled
        for i, offset in enumerate(corner_offsets):
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=dot_radius)
            server.scene.add_mesh_simple(
                f"/{name}/bbox_dotted_corner_{i}",
                vertices=sphere.vertices,
                faces=sphere.faces,
                color=color,
                position=offset,
            )
        
        for edge_idx, (start_idx, end_idx) in enumerate(edges):
            start_pos = corner_offsets[start_idx]
            end_pos = corner_offsets[end_idx]
            
            # Calculate edge length and number of dots based on gap
            edge_vec = end_pos - start_pos
            edge_length = np.linalg.norm(edge_vec)
            
            if edge_length > 0:
                # Number of interior dots (excluding corners)
                num_dots = max(0, int(edge_length / dot_gap) - 1)
                
                if num_dots > 0:
                    # Evenly space dots along the edge (between corners)
                    for dot_idx in range(1, num_dots + 1):
                        t = dot_idx / (num_dots + 1)  # Interpolation factor (0 < t < 1)
                        dot_pos = start_pos + t * edge_vec
                        
                        sphere = trimesh.creation.icosphere(subdivisions=1, radius=dot_radius)
                        server.scene.add_mesh_simple(
                            f"/{name}/bbox_edge_{edge_idx}_dot_{dot_idx}",
                            vertices=sphere.vertices,
                            faces=sphere.faces,
                            color=color,
                            position=dot_pos,
                        )
    
    # Add solid lines along edges
    if add_lines:
        # Define the 12 edges of the box as pairs of corner indices
        edges = [
            # Bottom face edges
            (0, 1), (0, 2), (1, 3), (2, 3),
            # Top face edges
            (4, 5), (4, 6), (5, 7), (6, 7),
            # Vertical edges connecting bottom and top
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        for edge_idx, (start_idx, end_idx) in enumerate(edges):
            start_pos = corner_offsets[start_idx]
            end_pos = corner_offsets[end_idx]
            
            # Calculate edge properties
            edge_vec = end_pos - start_pos
            edge_length = np.linalg.norm(edge_vec)
            
            if edge_length > 0:
                # Create a cylinder along the edge
                # trimesh.creation.cylinder creates a cylinder along Z-axis, centered at origin
                cylinder = trimesh.creation.cylinder(radius=line_thickness, height=edge_length, sections=8)
                
                # We need to rotate and translate the cylinder to align with the edge
                # The cylinder is along Z, we need to align it with edge_vec
                edge_dir = edge_vec / edge_length
                
                # Create rotation matrix to align Z-axis with edge direction
                z_axis = np.array([0, 0, 1])
                if np.allclose(edge_dir, z_axis):
                    rotation_matrix = np.eye(3)
                elif np.allclose(edge_dir, -z_axis):
                    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                else:
                    # Rodrigues' rotation formula
                    v = np.cross(z_axis, edge_dir)
                    s = np.linalg.norm(v)
                    c = np.dot(z_axis, edge_dir)
                    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
                
                # Apply rotation to cylinder vertices
                rotated_vertices = (rotation_matrix @ cylinder.vertices.T).T
                
                # Translate to edge midpoint
                midpoint = (start_pos + end_pos) / 2
                translated_vertices = rotated_vertices + midpoint
                
                server.scene.add_mesh_simple(
                    f"/{name}/bbox_line_{edge_idx}",
                    vertices=translated_vertices.astype(np.float32),
                    faces=cylinder.faces,
                    color=color,
                )

def process_mesh(config: ProcessMeshConfig) -> None:
    """
    Process mesh output directory and visualize merged point clouds.
    """
    output_dir = config.output_dir
    
    # Check required files/directories exist
    mesh_path = output_dir / "mesh" / "mesh.obj"
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    handle_masks_dir = output_dir / "handle_masks"
    head_masks_dir = output_dir / "head_masks"
    cam_K_path = output_dir / "cam_K.txt"
    cam_poses_path = output_dir / "cam_poses.npy"
    
    assert mesh_path.exists(), f"Mesh not found: {mesh_path}"
    assert rgb_dir.exists(), f"RGB directory not found: {rgb_dir}"
    assert depth_dir.exists(), f"Depth directory not found: {depth_dir}"
    assert handle_masks_dir.exists(), f"Handle masks directory not found: {handle_masks_dir}"
    assert head_masks_dir.exists(), f"Head masks directory not found: {head_masks_dir}"
    assert cam_K_path.exists(), f"Camera intrinsics not found: {cam_K_path}"
    assert cam_poses_path.exists(), f"Camera poses not found: {cam_poses_path}"
    
    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Load camera intrinsics
    K = np.loadtxt(cam_K_path)
    assert K.shape == (3, 3), f"K shape: {K.shape}, expected: (3, 3)"
    print(f"Loaded camera intrinsics K:\n{K}")
    
    # Load camera poses (N, 4, 4)
    cam_poses = np.load(cam_poses_path)
    assert cam_poses.ndim == 3 and cam_poses.shape[1:] == (4, 4), f"cam_poses shape: {cam_poses.shape}, expected: (N, 4, 4)"
    num_frames = cam_poses.shape[0]
    print(f"Loaded {num_frames} camera poses")
    
    # Get sorted list of frames
    rgb_files = sorted(rgb_dir.glob("*.png"))
    print(f"Found {len(rgb_files)} RGB frames")
    
    # Collect points from all views
    handle_points_list: List[np.ndarray] = []
    handle_colors_list: List[np.ndarray] = []
    head_points_list: List[np.ndarray] = []
    head_colors_list: List[np.ndarray] = []
    
    for i, rgb_path in enumerate(rgb_files):
        frame_name = rgb_path.stem
        depth_path = depth_dir / f"{frame_name}.png"
        handle_mask_path = handle_masks_dir / f"{frame_name}.png"
        head_mask_path = head_masks_dir / f"{frame_name}.png"
        
        if not depth_path.exists():
            print(f"  Skipping frame {i}: depth not found")
            continue
        if not handle_mask_path.exists():
            print(f"  Skipping frame {i}: handle mask not found")
            continue
        if not head_mask_path.exists():
            print(f"  Skipping frame {i}: head mask not found")
            continue
        if i >= num_frames:
            print(f"  Skipping frame {i}: no camera pose")
            continue
            
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing frame {i + 1}/{len(rgb_files)}")
        
        # Load images
        rgb = load_image(rgb_path)
        depth = load_image(depth_path).astype(np.float32)
        depth_m = convert_depth_to_meters(depth)
        handle_mask = load_mask(handle_mask_path)
        head_mask = load_mask(head_mask_path)
        
        # Get camera pose (world from camera)
        T_W_C = cam_poses[i]
        
        # Process handle points
        if handle_mask.any():
            pts_c, cols = depth_to_points(
                depth_m, K, rgb=rgb, mask=handle_mask,
                stride=config.subsample_stride, max_depth_m=config.max_depth_m
            )
            if len(pts_c) > 0:
                pts_w = transform_points(T_W_C, pts_c)
                handle_points_list.append(pts_w)
                handle_colors_list.append(cols)
        
        # Process head points
        if head_mask.any():
            pts_c, cols = depth_to_points(
                depth_m, K, rgb=rgb, mask=head_mask,
                stride=config.subsample_stride, max_depth_m=config.max_depth_m
            )
            if len(pts_c) > 0:
                pts_w = transform_points(T_W_C, pts_c)
                head_points_list.append(pts_w)
                head_colors_list.append(cols)
    
    # Merge point clouds
    if handle_points_list:
        handle_points = np.concatenate(handle_points_list, axis=0).astype(np.float32)
        handle_colors = np.concatenate(handle_colors_list, axis=0).astype(np.uint8)
        print(f"Handle point cloud: {len(handle_points)} points")

        # Remove outliers from handle_points (keep 10th to 90th percentile in x, y, z)
        handle_mask = np.ones(len(handle_points), dtype=bool)
        for dim in range(3):
            p10 = np.percentile(handle_points[:, dim], 10)
            p90 = np.percentile(handle_points[:, dim], 90)
            handle_mask &= (handle_points[:, dim] >= p10) & (handle_points[:, dim] <= p90)
        handle_points = handle_points[handle_mask]
        handle_colors = handle_colors[handle_mask]
        print(f"Handle points after outlier removal: {len(handle_points)} points")
    else:
        handle_points = np.zeros((0, 3), dtype=np.float32)
        handle_colors = np.zeros((0, 3), dtype=np.uint8)
        print("No handle points found")
    
    if head_points_list:
        head_points = np.concatenate(head_points_list, axis=0).astype(np.float32)
        head_colors = np.concatenate(head_colors_list, axis=0).astype(np.uint8)
        print(f"Head point cloud: {len(head_points)} points")

        # Remove outliers from head_points (keep 10th to 90th percentile in x, y, z)
        head_mask = np.ones(len(head_points), dtype=bool)
        for dim in range(3):
            p10 = np.percentile(head_points[:, dim], 10)
            p90 = np.percentile(head_points[:, dim], 90)
            head_mask &= (head_points[:, dim] >= p10) & (head_points[:, dim] <= p90)
        head_points = head_points[head_mask]
        head_colors = head_colors[head_mask]
        print(f"Head points after outlier removal: {len(head_points)} points")
    else:
        head_points = np.zeros((0, 3), dtype=np.float32)
        head_colors = np.zeros((0, 3), dtype=np.uint8)
        print("No head points found")
    
    # Visualize with viser
    print("\nStarting viser server...")
    server = viser.ViserServer()
    
    # Add mesh
    server.scene.add_mesh_trimesh(
        "/mesh",
        mesh=mesh,
    )
    print("Added mesh to viser")
    
    # Add handle point cloud
    if len(handle_points) > 0:
        server.scene.add_point_cloud(
            "/handle_point_cloud",
            points=handle_points,
            colors=handle_colors,
            point_size=config.point_size,
        )
        print("Added handle point cloud to viser")
    
    # Add head point cloud
    if len(head_points) > 0:
        server.scene.add_point_cloud(
            "/head_point_cloud",
            points=head_points,
            colors=head_colors,
            point_size=config.point_size,
        )
        print("Added head point cloud to viser")

    # Add adjusted object origin
    if len(handle_points) > 0 and len(head_points) > 0:
        handle_origin = np.mean(handle_points, axis=0)
        head_origin = np.mean(head_points, axis=0)

        # Compute handle frame (x-axis points from handle to head)
        T_W_H, (qw, qx, qy, qz) = compute_handle_frame(handle_origin, head_origin)
        
        # Inverse transform: T_H_W (handle from world) - to transform mesh into handle frame
        T_H_W = np.linalg.inv(T_W_H)
        
        # Compute handle bounding box size in handle frame coordinates (x, y, z)
        # Transform handle points to handle frame
        handle_points_in_handle_frame = transform_points(T_H_W, handle_points)
        # Get axis-aligned bounding box in handle frame
        handle_min = handle_points_in_handle_frame.min(axis=0)
        handle_max = handle_points_in_handle_frame.max(axis=0)
        handle_bbox_size = handle_max - handle_min  # (3,) array: (x_size, y_size, z_size)
        print(f"Handle bounding box size (x, y, z): {handle_bbox_size}")

        server.scene.add_frame(
            "/handle_frame",
            wxyz=(qw, qx, qy, qz),
            position=handle_origin,
            axes_length=0.3,
            axes_radius=0.01,
        )
        x_axis = T_W_H[:3, 0]  # First column is x-axis
        print(f"Handle frame origin: {handle_origin}")
        print(f"Head origin: {head_origin}")
        print(f"X-axis (handle->head): {x_axis}")
        
        # Transform mesh to handle frame and save
        mesh_transformed = mesh.copy()
        mesh_transformed.apply_transform(T_H_W)
        
        # Save transformed mesh
        save_mesh(
            mesh=mesh_transformed,
            name="mesh_handle_frame",
            parent_dir=output_dir,
        )
        
        # Also save the transform for reference
        transform_path = output_dir / "T_W_H.npy"
        np.save(transform_path, T_W_H)
        print(f"Saved handle frame transform (T_W_H) to: {transform_path}")

        # Also save the handle bounding box size in txt file with numpy
        handle_bbox_size_path = output_dir / "handle_bbox_size.txt"
        np.savetxt(handle_bbox_size_path, handle_bbox_size)
        print(f"Saved handle bounding box size to: {handle_bbox_size_path}")

        # Visualize transformed mesh in viser (at origin, for verification)
        server.scene.add_mesh_trimesh(
            "/mesh_handle_frame",
            mesh=mesh_transformed,
        )
        print("Added transformed mesh to viser (hidden by default)")

        # Visualize the handle bounding box in viser
        visualize_bbox(
            server=server,
            bbox_size=handle_bbox_size,
            position=handle_origin,
            wxyz=np.array([qw, qx, qy, qz]),
            name="handle_bbox_frame",
            color=(255, 0, 0),
        )
        
        # Crop mesh to handle bounding box to create handle_mesh
        # The bounding box is defined in handle frame, so we need to:
        # 1. Transform mesh to handle frame
        # 2. Check which faces are inside the bbox
        # 3. Extract submesh
        # 4. Save both in world frame and handle frame
        
        # Get face centroids in handle frame
        mesh_in_handle_frame = mesh.copy()
        mesh_in_handle_frame.apply_transform(T_H_W)
        face_centroids = mesh_in_handle_frame.triangles_center
        
        # Mask faces inside the handle bounding box (in handle frame coordinates)
        # The bbox is centered at origin in handle frame, so min/max are:
        bbox_min_handle = handle_min
        bbox_max_handle = handle_max
        
        mask = np.all((face_centroids >= bbox_min_handle) & (face_centroids <= bbox_max_handle), axis=1)
        
        if mask.sum() > 0:
            # Extract submesh (in handle frame)
            handle_mesh_handle_frame = mesh_in_handle_frame.submesh([mask], append=True)
            print(f"Handle mesh: {len(handle_mesh_handle_frame.vertices)} vertices, {len(handle_mesh_handle_frame.faces)} faces (from {mask.sum()} masked faces)")
            
            # Transform back to world frame
            handle_mesh_world = handle_mesh_handle_frame.copy()
            handle_mesh_world.apply_transform(T_W_H)
            
            # Save handle mesh in world frame
            save_mesh(
                mesh=handle_mesh_world,
                name="handle_mesh",
                parent_dir=output_dir,
            )
            print("Saved handle_mesh (world frame)")
            
            # Save handle mesh in handle frame (centered at handle origin)
            save_mesh(
                mesh=handle_mesh_handle_frame,
                name="handle_mesh_handle_frame",
                parent_dir=output_dir,
            )
            print("Saved handle_mesh_handle_frame (handle frame)")
            
            # Visualize handle mesh in viser (world frame)
            server.scene.add_mesh_trimesh(
                "/handle_mesh",
                mesh=handle_mesh_world,
                visible=False,
            )
            print("Added handle_mesh to viser")
            
            # Visualize handle mesh in handle frame (at origin)
            server.scene.add_mesh_trimesh(
                "/handle_mesh_at_origin",
                mesh=handle_mesh_handle_frame,
                visible=False,
            )
            print("Added handle_mesh_at_origin to viser")
            
            # Compute bounding box of the handle mesh (in handle frame)
            handle_mesh_vertices_hf = np.array(handle_mesh_handle_frame.vertices)
            handle_mesh_min = handle_mesh_vertices_hf.min(axis=0)
            handle_mesh_max = handle_mesh_vertices_hf.max(axis=0)
            handle_mesh_bbox_size = handle_mesh_max - handle_mesh_min
            handle_mesh_bbox_center_hf = (handle_mesh_min + handle_mesh_max) / 2
            
            # Transform center to world frame for visualization
            handle_mesh_bbox_center_world = T_W_H[:3, :3] @ handle_mesh_bbox_center_hf + T_W_H[:3, 3]
            
            print(f"Handle mesh bounding box size (x, y, z): {handle_mesh_bbox_size}")
            print(f"Handle mesh bounding box center (handle frame): {handle_mesh_bbox_center_hf}")
            print(f"Handle mesh bounding box center (world frame): {handle_mesh_bbox_center_world}")
            
            # Save handle mesh bounding box size
            handle_mesh_bbox_size_path = output_dir / "handle_mesh_bbox_size.txt"
            np.savetxt(handle_mesh_bbox_size_path, handle_mesh_bbox_size)
            print(f"Saved handle mesh bounding box size to: {handle_mesh_bbox_size_path}")
            
            # Visualize handle mesh bounding box in viser (blue)
            visualize_bbox(
                server=server,
                bbox_size=handle_mesh_bbox_size,
                position=handle_mesh_bbox_center_world,
                wxyz=np.array([qw, qx, qy, qz]),
                name="handle_mesh_bbox_frame",
                color=(0, 0, 255),
            )
            print("Added handle_mesh_bbox to viser (blue)")
        else:
            print("WARNING: No faces found inside handle bounding box!")
    
    # Add coordinate frame at origin
    server.scene.add_frame("/world_frame", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=0.1, axes_radius=0.01)
    
    # Add camera frustums for visualization
    for i in range(num_frames):
        T_W_C = cam_poses[i]
        pos = T_W_C[:3, 3]
        R = T_W_C[:3, :3]
        wxyz = rotation_matrix_to_quaternion(R)
        
        server.scene.add_frame(
            f"/cameras/camera_{i:04d}",
            wxyz=wxyz,
            position=pos,
            axes_length=0.05,
            axes_radius=0.01,
        )
    
    print("\nViser server running at: http://localhost:8080")
    print("Press Ctrl+C to exit")
    
    # Keep server running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")


def main() -> None:
    """Main entry point."""
    config = tyro.cli(ProcessMeshConfig)
    
    print("=" * 80)
    print(f"Process Mesh Configuration:\n{tyro.extras.to_yaml(config)}")
    print("=" * 80 + "\n")
    
    process_mesh(config)


if __name__ == "__main__":
    main()
