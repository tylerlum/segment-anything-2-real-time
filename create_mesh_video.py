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
    
    save_frames: bool = False
    """Whether to save individual frames as images"""
    
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


def render_frame(
    plotter: pv.Plotter,
    centroid: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """
    Render a single frame with the camera at the specified position.
    
    Returns:
        Image as numpy array (H, W, 3)
    """
    camera_pos = compute_camera_position(centroid, radius, azimuth_deg, elevation_deg)
    target_pos = centroid
    up_vector = np.array([0, 1, 0])  # y-up
    
    plotter.camera_position = [
        camera_pos.tolist(),
        target_pos.tolist(),
        up_vector.tolist(),
    ]
    
    # Must call render() to update the scene before taking screenshot
    plotter.render()
    
    return plotter.screenshot(transparent_background=False)


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
    frames = []
    azimuth_angles = np.linspace(0, 360, config.num_frames, endpoint=False)
    
    print(f"Rendering {config.num_frames} frames...")
    
    # Create temp directory for frames if saving
    if config.save_frames:
        frames_dir = config.output_video_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    for i, azimuth in enumerate(azimuth_angles):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Frame {i + 1}/{config.num_frames} (azimuth: {azimuth:.1f}°)")
        
        frame = render_frame(
            pl,
            centroid,
            camera_radius,
            azimuth,
            config.elevation_angle,
        )
        frames.append(frame)
        
        if config.save_frames:
            img = Image.fromarray(frame)
            img.save(frames_dir / f"frame_{i:04d}.png")
    
    pl.close()
    
    print(f"Rendered {len(frames)} frames")
    
    # Create video using ffmpeg
    print(f"Creating video at: {config.output_video_path}")
    config.output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write frames to temp directory and use ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save frames as images
        for i, frame in enumerate(frames):
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
