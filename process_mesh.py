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


@dataclass
class ProcessMeshConfig:
    """Configuration for mesh processing."""
    
    output_dir: Path
    """Directory containing mesh.obj, rgb/, depth/, cam_K.txt, handle_masks/, head_masks/, cam_poses.npy"""
    
    point_size: float = 0.002
    """Point size for visualization"""
    
    max_depth_m: float = 10.0
    """Maximum depth in meters to include"""
    
    subsample_stride: int = 1
    """Stride for subsampling points"""
    
    def __post_init__(self) -> None:
        assert self.output_dir.exists(), f"Output directory not found: {self.output_dir}"


def process_mesh(config: ProcessMeshConfig) -> None:
    """
    Process mesh output directory and visualize merged point clouds.
    """
    output_dir = config.output_dir
    
    # Check required files/directories exist
    mesh_path = output_dir / "mesh.obj"
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
    else:
        handle_points = np.zeros((0, 3), dtype=np.float32)
        handle_colors = np.zeros((0, 3), dtype=np.uint8)
        print("No handle points found")
    
    if head_points_list:
        head_points = np.concatenate(head_points_list, axis=0).astype(np.float32)
        head_colors = np.concatenate(head_colors_list, axis=0).astype(np.uint8)
        print(f"Head point cloud: {len(head_points)} points")
    else:
        head_points = np.zeros((0, 3), dtype=np.float32)
        head_colors = np.zeros((0, 3), dtype=np.uint8)
        print("No head points found")
    
    # Visualize with viser
    print("\nStarting viser server...")
    server = viser.ViserServer()
    
    # Add mesh
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    
    server.scene.add_mesh_simple(
        "/mesh",
        vertices=vertices,
        faces=faces,
        color=(0.7, 0.7, 0.7),  # Light gray base color
        wireframe=False,
    )
    print(f"Added mesh to viser ({len(vertices)} vertices, {len(faces)} faces)")
    
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
    
    # Add coordinate frame at origin
    server.scene.add_frame("/world_frame", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=0.1)
    
    # Add camera frustums for visualization
    for i in range(num_frames):
        T_W_C = cam_poses[i]
        pos = T_W_C[:3, 3]
        # Convert rotation matrix to quaternion (wxyz)
        R = T_W_C[:3, :3]
        # Simple rotation matrix to quaternion conversion
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
        
        server.scene.add_frame(
            f"/cameras/camera_{i:04d}",
            wxyz=(w, x, y, z),
            position=pos,
            axes_length=0.05,
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
