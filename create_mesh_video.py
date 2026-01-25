"""
Script to render a 3D mesh from multiple camera angles in a circular orbit
and stitch the images together into a video.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess

import numpy as np
import pyvista as pv
import trimesh
import tyro
from PIL import Image


@dataclass
class MeshVideoConfig:
    """Configuration for mesh video generation."""
    
    mesh_filepath: Path = (
        Path(__file__).parent / "example_data" / "woodblock_mesh" / "3DModel.obj"
    )
    """Path to the mesh file (.obj)"""
    
    output_video_path: Path | None = None
    """Path to output video. If None, saves to same folder as mesh with name 'mesh_orbit.mp4'"""
    
    image_width: int = 800
    """Width of rendered images"""
    
    image_height: int = 600
    """Height of rendered images"""
    
    num_frames: int = 120
    """Number of frames in the orbit (more = smoother video)"""
    
    fps: int = 30
    """Frames per second for output video"""
    
    orbit_radius_scale: float = 4.0
    """Scale factor for camera distance from mesh center (relative to mesh bounding sphere)"""
    
    elevation_angle: float = 30.0
    """Camera elevation angle in degrees (0 = horizontal, 90 = top-down)"""
    
    background_color: str = "white"
    """Background color for rendering"""
    
    save_frames: bool = True
    """Whether to save individual frames as images (rgb/ and depth/ directories)"""
    
    def __post_init__(self) -> None:
        assert self.mesh_filepath.exists(), f"Mesh file not found: {self.mesh_filepath}"
        if self.output_video_path is None:
            self.output_video_path = self.mesh_filepath.parent / "mesh_orbit.mp4"


def compute_camera_position(
    centroid: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    Compute camera position on a sphere around the centroid.
    
    Args:
        centroid: Center point to orbit around (3,)
        radius: Distance from centroid
        azimuth_deg: Horizontal angle in degrees (0-360)
        elevation_deg: Vertical angle in degrees (0 = horizontal, 90 = top)
    
    Returns:
        Camera position as (3,) array
    """
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    
    # Spherical to Cartesian conversion
    # x = r * cos(elevation) * cos(azimuth)
    # y = r * sin(elevation)  (y is up)
    # z = r * cos(elevation) * sin(azimuth)
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.sin(elevation_rad)
    z = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    
    return centroid + np.array([x, y, z])


def compute_camera_extrinsics(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    up_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute camera extrinsic matrix (world from camera transform, T_W_C).
    
    Args:
        camera_pos: Camera position in world frame (3,)
        target_pos: Look-at target position in world frame (3,)
        up_vector: Up vector in world frame (3,)
    
    Returns:
        T_W_C: (4, 4) transformation matrix (world from camera)
    """
    # Compute camera coordinate axes in world frame
    # z-axis points from camera to target (forward)
    z_axis = target_pos - camera_pos
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # x-axis is perpendicular to z and up (right)
    x_axis = np.cross(z_axis, up_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # y-axis is perpendicular to z and x (down in camera convention)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Rotation matrix: columns are camera axes in world frame
    R_W_C = np.column_stack([x_axis, y_axis, z_axis])
    
    # Build 4x4 transformation matrix
    T_W_C = np.eye(4)
    T_W_C[:3, :3] = R_W_C
    T_W_C[:3, 3] = camera_pos
    
    return T_W_C


def render_frame(
    plotter: pv.Plotter,
    centroid: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a single frame with the camera at the specified position.
    
    Returns:
        Tuple of (rgb_image, depth_image, T_W_C) as numpy arrays
        - rgb_image: (H, W, 3) uint8
        - depth_image: (H, W) float32 with depth values in world units (meters)
        - T_W_C: (4, 4) camera extrinsic matrix (world from camera)
    """
    camera_pos = compute_camera_position(centroid, radius, azimuth_deg, elevation_deg)
    target_pos = centroid
    up_vector = np.array([0, 1, 0])  # y-up
    
    plotter.camera_position = [
        camera_pos.tolist(),
        target_pos.tolist(),
        up_vector.tolist(),
    ]
    
    # Compute camera extrinsics
    T_W_C = compute_camera_extrinsics(camera_pos, target_pos, up_vector)
    
    # Must call render() to update the scene before taking screenshot
    plotter.render()
    
    # Get RGB image
    rgb = plotter.screenshot(transparent_background=False)
    
    # Get depth buffer - returns z-buffer values in NDC (normalized device coordinates)
    # These are NOT actual depth values, need to convert using clipping planes
    z_buffer = np.array(plotter.get_image_depth())
    
    # Get camera clipping range (near, far)
    near, far = plotter.camera.clipping_range
    
    # Convert z-buffer (NDC) to linear depth
    # z_buffer is in range [-1, 1] where -1 is near plane, 1 is far plane (OpenGL convention)
    # But PyVista may return values outside this range for background (NaN or very negative)
    # Linear depth formula: depth = 2 * near * far / (far + near - z_ndc * (far - near))
    # However, get_image_depth() returns values that need different handling
    
    # Actually, get_image_depth() returns the actual z-distance from camera in world units
    # but with negative values (camera looks down -z in its local frame)
    # The NaN values are for background pixels
    depth = -z_buffer  # Flip sign to get positive depth
    
    return rgb, depth, T_W_C


def get_camera_intrinsics(plotter: pv.Plotter, width: int, height: int) -> np.ndarray:
    """
    Extract camera intrinsic matrix K from PyVista plotter.
    
    Returns:
        K: (3, 3) camera intrinsic matrix
    """
    # Get the camera's vertical field of view in degrees
    camera = plotter.camera
    fov_y_deg = camera.view_angle  # Vertical FOV in degrees
    fov_y_rad = np.radians(fov_y_deg)
    
    # Compute focal length in pixels
    # f_y = height / (2 * tan(fov_y / 2))
    f_y = height / (2.0 * np.tan(fov_y_rad / 2.0))
    
    # Assume square pixels (f_x = f_y)
    f_x = f_y
    
    # Principal point at image center
    c_x = width / 2.0
    c_y = height / 2.0
    
    # Construct intrinsic matrix
    K = np.array([
        [f_x, 0.0, c_x],
        [0.0, f_y, c_y],
        [0.0, 0.0, 1.0],
    ])
    
    return K


def create_mesh_video(config: MeshVideoConfig) -> None:
    """
    Create a video of a mesh rotating (camera orbiting around it).
    """
    print(f"Loading mesh from: {config.mesh_filepath}")
    
    # Load mesh with trimesh to get geometry info
    mesh = trimesh.load(config.mesh_filepath)
    centroid = mesh.centroid
    
    # Compute bounding sphere radius for camera distance
    vertices = np.array(mesh.vertices)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    bounding_radius = distances.max()
    
    camera_radius = bounding_radius * config.orbit_radius_scale
    
    print(f"Mesh centroid: {centroid}")
    print(f"Bounding radius: {bounding_radius:.4f}")
    print(f"Camera radius: {camera_radius:.4f}")
    
    # Initialize PyVista plotter
    pl = pv.Plotter(off_screen=True, window_size=[config.image_width, config.image_height])
    pl.import_obj(str(config.mesh_filepath))
    pl.set_background(config.background_color)
    
    # Generate frames
    rgb_frames = []
    cam_poses = []
    azimuth_angles = np.linspace(0, 360, config.num_frames, endpoint=False)
    
    print(f"Rendering {config.num_frames} frames...")
    
    # Create directories for frames if saving
    if config.save_frames:
        rgb_dir = config.output_video_path.parent / "rgb"
        depth_dir = config.output_video_path.parent / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        
        # Need to render once to initialize camera before extracting intrinsics
        pl.render()
        
        # Save camera intrinsics
        K = get_camera_intrinsics(pl, config.image_width, config.image_height)
        cam_K_path = config.output_video_path.parent / "cam_K.txt"
        np.savetxt(cam_K_path, K)
        print(f"Saved camera intrinsics to: {cam_K_path}")
        print(f"Camera intrinsic matrix K:\n{K}")
    
    for i, azimuth in enumerate(azimuth_angles):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i + 1}/{config.num_frames} (azimuth: {azimuth:.1f}°)")
        
        rgb, depth_buffer, T_W_C = render_frame(
            pl,
            centroid,
            camera_radius,
            azimuth,
            config.elevation_angle,
        )
        rgb_frames.append(rgb)
        cam_poses.append(T_W_C)
        
        if config.save_frames:
            # Save RGB image
            rgb_img = Image.fromarray(rgb)
            rgb_img.save(rgb_dir / f"frame_{i:04d}.png")
            
            # Convert depth to mm (uint16)
            # depth_buffer contains actual depth values in world units (meters assumed)
            # Convert to mm
            depth_mm = depth_buffer * 1000.0
            # Handle NaN/inf values (background pixels) - set to 0
            depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=65535.0, neginf=0.0)
            # Clip to valid uint16 range (0-65535 mm = 0-65.535 m)
            depth_mm = np.clip(depth_mm, 0, 65535)
            depth_uint16 = depth_mm.astype(np.uint16)
            
            # Save as 16-bit PNG
            depth_img = Image.fromarray(depth_uint16, mode='I;16')
            depth_img.save(depth_dir / f"frame_{i:04d}.png")
    
    # Save camera poses as (N, 4, 4) numpy array
    if config.save_frames:
        cam_poses_array = np.stack(cam_poses, axis=0)  # (N, 4, 4)
        cam_poses_path = config.output_video_path.parent / "cam_poses.npy"
        np.save(cam_poses_path, cam_poses_array)
        print(f"Saved camera poses to: {cam_poses_path}")
    
    pl.close()
    
    print(f"Rendered {len(rgb_frames)} frames")
    
    if config.save_frames:
        print(f"Saved RGB frames to: {rgb_dir}")
        print(f"Saved depth frames to: {depth_dir}")
    
    # Create video using ffmpeg
    print(f"Creating video at: {config.output_video_path}")
    config.output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use saved frames if available, otherwise use temp directory
    if config.save_frames:
        # Use the already saved RGB frames
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(config.fps),
            "-i", str(rgb_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",  # High quality
            str(config.output_video_path),
        ]
        
        print(f"Running: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")
    else:
        # Write frames to temp directory and use ffmpeg
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Save frames as images
            for i, frame in enumerate(rgb_frames):
                img = Image.fromarray(frame)
                img.save(tmpdir / f"frame_{i:04d}.png")
            
            # Use ffmpeg to create video
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(config.fps),
                "-i", str(tmpdir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                str(config.output_video_path),
            ]
            
            print(f"Running: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"ffmpeg stderr: {result.stderr}")
                raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")
    
    print(f"Video saved to: {config.output_video_path}")
    print(f"Duration: {config.num_frames / config.fps:.2f} seconds")


def main() -> None:
    """Main entry point."""
    config = tyro.cli(MeshVideoConfig)
    
    print("=" * 80)
    print(f"Mesh Video Configuration:\n{tyro.extras.to_yaml(config)}")
    print("=" * 80 + "\n")
    
    create_mesh_video(config)


if __name__ == "__main__":
    main()
