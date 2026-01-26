from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Paths
DEMO_DIR = Path("/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/anvil_brush/sweep_forward")
assert DEMO_DIR.exists(), f"Demo directory not found: {DEMO_DIR}"
RGB_DIR = DEMO_DIR / "rgb"
assert RGB_DIR.exists(), f"RGB directory not found: {RGB_DIR}"
MASK_DIR = DEMO_DIR / "masks"
assert MASK_DIR.exists(), f"Mask directory not found: {MASK_DIR}"

# Get sorted list of RGB and mask files
RGB_FILES = sorted(list(RGB_DIR.glob("*.png")))
assert len(RGB_FILES) > 0, f"No RGB files found in {RGB_DIR}"
print(f"Found {len(RGB_FILES)} RGB files")
MASK_FILES = sorted(list(MASK_DIR.glob("*.png")))
assert len(MASK_FILES) > 0, f"No mask files found in {MASK_DIR}"
print(f"Found {len(MASK_FILES)} mask files")

# Frame indices to use
FRAME_INDICES = [0, 200, 400]

# Colors for each pose (vibrant, distinct colors)
POSE_COLORS = [
    (255, 80, 80),    # Red for pose 1
    (80, 200, 80),    # Green for pose 2  
    (80, 120, 255),   # Blue for pose 3
]

# Load the RGB and mask images
rgb_images = [Image.open(RGB_FILES[i]) for i in FRAME_INDICES]
mask_images = [Image.open(MASK_FILES[i]) for i in FRAME_INDICES]


def get_mask_bool(mask_image: Image.Image) -> np.ndarray:
    """Convert mask image to boolean array."""
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[..., 0]
    return mask_array > 0


def create_composite_with_object_poses(
    rgb_images: list,
    mask_images: list,
    pose_colors: list,
    background_alpha: float = 0.3,
    object_alpha: float = 0.9,
    outline_width: int = 3,
) -> np.ndarray:
    """
    Create a composite image showing object pose evolution.
    
    - Background from middle frame, greyed out
    - Objects from all frames overlaid with colored outlines
    - Objects are vibrant, background is faded
    """
    # Use middle frame as background reference
    bg_idx = len(rgb_images) // 2
    bg_rgb = np.array(rgb_images[bg_idx]).astype(np.float32)
    
    H, W, C = bg_rgb.shape
    
    # Create greyed-out background (desaturate and fade)
    bg_gray = np.mean(bg_rgb, axis=2, keepdims=True)
    bg_faded = (bg_gray * background_alpha + 255 * (1 - background_alpha)).astype(np.uint8)
    bg_faded = np.repeat(bg_faded, 3, axis=2)
    
    # Start with faded background
    composite = bg_faded.astype(np.float32)
    
    # Overlay each object pose
    for i, (rgb_img, mask_img, color) in enumerate(zip(rgb_images, mask_images, pose_colors)):
        rgb_array = np.array(rgb_img).astype(np.float32)
        mask_bool = get_mask_bool(mask_img)
        
        # Create outline by dilating mask and subtracting original
        from scipy import ndimage
        dilated = ndimage.binary_dilation(mask_bool, iterations=outline_width)
        outline = dilated & ~mask_bool
        
        # Blend object region
        for c in range(3):
            # Object interior: show actual RGB with slight color tint
            composite[:, :, c] = np.where(
                mask_bool,
                rgb_array[:, :, c] * object_alpha + color[c] * (1 - object_alpha) * 0.3,
                composite[:, :, c]
            )
            # Outline: solid color
            composite[:, :, c] = np.where(
                outline,
                color[c],
                composite[:, :, c]
            )
    
    return np.clip(composite, 0, 255).astype(np.uint8)


def create_pose_sequence_figure(
    rgb_images: list,
    mask_images: list,
    pose_colors: list,
    background_alpha: float = 0.25,
    use_colored_outlines: bool = True,
) -> np.ndarray:
    """
    Create a single image showing pose sequence with ghosted context.
    
    - Light grey background showing human context
    - Vibrant colored objects showing pose progression
    - Arrows or numbers indicating sequence
    
    Args:
        use_colored_outlines: If True, add colored outlines. If False, just show objects.
    """
    # Get image dimensions
    H, W, _ = np.array(rgb_images[0]).shape
    
    # Create composite background from all frames (averaged and faded)
    bg_composite = np.zeros((H, W, 3), dtype=np.float32)
    for rgb_img in rgb_images:
        bg_composite += np.array(rgb_img).astype(np.float32)
    bg_composite /= len(rgb_images)
    
    # Convert to grayscale and fade
    bg_gray = np.mean(bg_composite, axis=2, keepdims=True)
    # Make it light gray (closer to white)
    bg_faded = bg_gray * background_alpha + 255 * (1 - background_alpha)
    bg_faded = np.repeat(bg_faded, 3, axis=2)
    
    composite = bg_faded.copy()
    
    # Import scipy for morphological operations
    from scipy import ndimage
    
    # Overlay objects from each frame (later frames on top)
    for i, (rgb_img, mask_img, color) in enumerate(zip(rgb_images, mask_images, pose_colors)):
        rgb_array = np.array(rgb_img).astype(np.float32)
        mask_bool = get_mask_bool(mask_img)
        
        # Object interior: show actual RGB (vibrant)
        for c in range(3):
            composite[:, :, c] = np.where(mask_bool, rgb_array[:, :, c], composite[:, :, c])
        
        # Add colored outline only if requested
        if use_colored_outlines:
            outline_width = 4
            dilated = ndimage.binary_dilation(mask_bool, iterations=outline_width)
            outline = dilated & ~mask_bool
            for c in range(3):
                composite[:, :, c] = np.where(outline, color[c], composite[:, :, c])
    
    return np.clip(composite, 0, 255).astype(np.uint8)


def create_stacked_poses_figure(
    rgb_images: list,
    mask_images: list,
    pose_colors: list,
) -> np.ndarray:
    """
    Create figure with all object poses stacked/overlaid on neutral background.
    Objects are shown with colored outlines, background is clean white/light gray.
    """
    H, W, _ = np.array(rgb_images[0]).shape
    
    # Start with light gray background
    composite = np.full((H, W, 3), 245, dtype=np.float32)  # Near-white
    
    from scipy import ndimage
    
    # Overlay objects (first frame at bottom, last on top)
    for i, (rgb_img, mask_img, color) in enumerate(zip(rgb_images, mask_images, pose_colors)):
        rgb_array = np.array(rgb_img).astype(np.float32)
        mask_bool = get_mask_bool(mask_img)
        
        # Create outline
        outline_width = 5
        dilated = ndimage.binary_dilation(mask_bool, iterations=outline_width)
        outline = dilated & ~mask_bool
        
        # Add slight transparency for earlier poses
        alpha = 0.7 + 0.3 * (i / (len(rgb_images) - 1)) if len(rgb_images) > 1 else 1.0
        
        # Object interior
        for c in range(3):
            composite[:, :, c] = np.where(
                mask_bool,
                rgb_array[:, :, c] * alpha + composite[:, :, c] * (1 - alpha),
                composite[:, :, c]
            )
        
        # Colored outline (solid)
        for c in range(3):
            composite[:, :, c] = np.where(outline, color[c], composite[:, :, c])
    
    return np.clip(composite, 0, 255).astype(np.uint8)


# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the main figure - pose sequence with context
print("Creating pose sequence figure...")
pose_sequence_img = create_pose_sequence_figure(
    rgb_images, mask_images, POSE_COLORS,
    background_alpha=0.2,
)

# Create alternative - stacked poses on clean background  
print("Creating stacked poses figure...")
stacked_poses_img = create_stacked_poses_figure(
    rgb_images, mask_images, POSE_COLORS,
)

# Create a figure showing both options
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(pose_sequence_img)
axes[0].set_title("Object Pose Sequence (with context)", fontsize=14)
axes[0].axis("off")

axes[1].imshow(stacked_poses_img)
axes[1].set_title("Object Pose Sequence (clean background)", fontsize=14)
axes[1].axis("off")

# Add legend
legend_text = "Pose 1 (Red) → Pose 2 (Green) → Pose 3 (Blue)"
fig.text(0.5, 0.02, legend_text, ha='center', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig(OUTPUT_DIR / "pose_sequence_comparison.png", dpi=200, bbox_inches='tight')
print(f"Saved comparison to {OUTPUT_DIR / 'pose_sequence_comparison.png'}")

# Save individual high-res versions
Image.fromarray(pose_sequence_img).save(OUTPUT_DIR / "pose_sequence_with_context.png")
Image.fromarray(stacked_poses_img).save(OUTPUT_DIR / "pose_sequence_clean.png")
print(f"Saved individual images to {OUTPUT_DIR}")


def create_single_pose_with_context(
    rgb_images: list,
    mask_images: list,
    pose_idx: int,
    pose_color: tuple,
    background_alpha: float = 0.2,
) -> np.ndarray:
    """
    Create an image showing a single object pose with greyed-out context.
    Same style as pose_sequence_with_context but only one pose highlighted.
    """
    from scipy import ndimage
    
    # Get image dimensions
    H, W, _ = np.array(rgb_images[0]).shape
    
    # Create composite background from all frames (averaged and faded)
    bg_composite = np.zeros((H, W, 3), dtype=np.float32)
    for rgb_img in rgb_images:
        bg_composite += np.array(rgb_img).astype(np.float32)
    bg_composite /= len(rgb_images)
    
    # Convert to grayscale and fade
    bg_gray = np.mean(bg_composite, axis=2, keepdims=True)
    # Make it light gray (closer to white)
    bg_faded = bg_gray * background_alpha + 255 * (1 - background_alpha)
    bg_faded = np.repeat(bg_faded, 3, axis=2)
    
    composite = bg_faded.copy()
    
    # Overlay only the single pose
    rgb_array = np.array(rgb_images[pose_idx]).astype(np.float32)
    mask_bool = get_mask_bool(mask_images[pose_idx])
    
    # Create thick outline
    outline_width = 4
    dilated = ndimage.binary_dilation(mask_bool, iterations=outline_width)
    outline = dilated & ~mask_bool
    
    # Object interior: show actual RGB (vibrant)
    for c in range(3):
        composite[:, :, c] = np.where(mask_bool, rgb_array[:, :, c], composite[:, :, c])
    
    # Add colored outline
    for c in range(3):
        composite[:, :, c] = np.where(outline, pose_color[c], composite[:, :, c])
    
    return np.clip(composite, 0, 255).astype(np.uint8)


# Create individual pose images (one for each pose)
print("Creating individual pose images...")
POSE_NAMES = ["pose_1_red", "pose_2_green", "pose_3_blue"]

for i, (color, name) in enumerate(zip(POSE_COLORS, POSE_NAMES)):
    single_pose_img = create_single_pose_with_context(
        rgb_images, mask_images,
        pose_idx=i,
        pose_color=color,
        background_alpha=0.2,
    )
    output_path = OUTPUT_DIR / f"pose_sequence_{name}.png"
    Image.fromarray(single_pose_img).save(output_path)
    print(f"Saved {output_path}")

print("Done!")

plt.show()
