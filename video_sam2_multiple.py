from video_sam2 import run_video_sam2 as run_video_sam2_same_process, get_user_point
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Optional

def run_video_sam2(
    input_dir: Path,
    output_dir: Path,
    use_negative_prompt: bool,
    use_second_prompt: bool,
    prompt_x: Optional[int] = None,
    prompt_y: Optional[int] = None,
    negative_prompt_x: Optional[int] = None,
    negative_prompt_y: Optional[int] = None,
    second_prompt_x: Optional[int] = None,
    second_prompt_y: Optional[int] = None,
    visualize: bool = False,
) -> None:
    RUN_WITH_SAME_PROCESS = False  # Same process seems to OOM
    if RUN_WITH_SAME_PROCESS:
        run_video_sam2_same_process(
            input_dir=input_dir,
            output_dir=output_dir,
            use_negative_prompt=use_negative_prompt,
            use_second_prompt=use_second_prompt,
            prompt_x=prompt_x,
            prompt_y=prompt_y,
            negative_prompt_x=negative_prompt_x,
            negative_prompt_y=negative_prompt_y,
            second_prompt_x=second_prompt_x,
            second_prompt_y=second_prompt_y,
            visualize=True,
        )
    else:
        cmd = (
            "python video_sam2.py "
            + f"--input_dir {input_dir} "
            + f"--output_dir {output_dir} "
            + ("--use_negative_prompt " if use_negative_prompt else "")
            + ("--use_second_prompt " if use_second_prompt else "")
            + f"--prompt_x {prompt_x} "
            + f"--prompt_y {prompt_y} "
            + (f"--negative_prompt_x {negative_prompt_x} " if negative_prompt_x is not None else "")
            + (f"--negative_prompt_y {negative_prompt_y} " if negative_prompt_y is not None else "")
            + (f"--second_prompt_x {second_prompt_x} " if second_prompt_x is not None else "")
            + (f"--second_prompt_y {second_prompt_y} " if second_prompt_y is not None else "")
            + ("--visualize " if visualize else "")
        )
        print(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

"""
/juno/u/kedia/FoundationPose/human_videos/Jan_15
├── brush
│   ├── anvil_brush
│   │   └── 20260115_231442
│   ├── lab_brush
│   │   └── 20260115_231622
│   └── red_brush
│       └── 20260115_231110
├── eraser
│   ├── amazon_eraser
│   │   └── 20260115_232955
│   ├── anvil_eraser
│   │   └── 20260115_233226
│   └── expo_eraser
│       └── 20260115_233123
├── hammer
│   ├── hammer_2
│   │   ├── clockwise
│   │   └── counter_clockwise
│   └── mallet
│       ├── clockwise
│       └── counter_clockwise
├── marker
│   ├── 040_large_marker
│   │   └── 20260115_232812
│   ├── sharpie_closed
│   │   └── 20260115_232717
│   └── staples_open
│       └── 20260115_232506
├── screwdriver
│   ├── black_screwdriver
│   │   └── 20260115_235139
│   ├── real_flat_screwdriver
│   │   └── 20260115_235042
│   └── red_screwdriver
│       └── 20260115_235241
├── spatula
│   ├── black_spatula
│   │   └── 20260115_233647
│   ├── spoon_spatula
│   │   └── 20260115_233602
│   └── wooden_spatula
│       └── 20260115_233858
"""



def main():
    # List of demo directories
    DEMO_DIRS = [
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/anvil_brush/20260115_231442/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/lab_brush/20260115_231622/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/brush/red_brush/20260115_231110/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/amazon_eraser/20260115_232955/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/anvil_eraser/20260115_233226/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/eraser/expo_eraser/20260115_233123/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/hammer_2/clockwise/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/hammer_2/counter_clockwise/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/mallet/clockwise"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/hammer/mallet/counter_clockwise"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/040_large_marker/20260115_232812"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/sharpie_closed/20260115_232717/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/marker/staples_open/20260115_232506/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/black_screwdriver/20260115_235139/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/real_flat_screwdriver/20260115_235042/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/screwdriver/red_screwdriver/20260115_235241/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/black_spatula/20260115_233647/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/spoon_spatula/20260115_233602/"),
        Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15/spatula/wooden_spatula/20260115_233858/"),
    ]
    # Validate
    for demo_dir in tqdm(DEMO_DIRS, desc="Validating demo directories"):
        assert demo_dir.exists(), f"Demo directory {demo_dir} does not exist"
        assert (demo_dir / "rgb").exists(), f"RGB directory {demo_dir / 'rgb'} does not exist"

    # Get first image for each demo directory
    first_img_filepaths = []
    for demo_dir in tqdm(DEMO_DIRS, desc="Getting first image for each demo directory"):
        input_dir = demo_dir / "rgb"
        jpg_filepaths = sorted(list(input_dir.glob("*.jpg")))
        png_filepaths = sorted(list(input_dir.glob("*.png")))
        if len(jpg_filepaths) == 0 and len(png_filepaths) == 0:
            raise ValueError(f"No frames found in the input directory {input_dir}")
        elif len(jpg_filepaths) == 0:
            first_img_filepath = png_filepaths[0]
        else:
            first_img_filepath = jpg_filepaths[0]
        first_img_filepaths.append(first_img_filepath)

    # Get prompts for each image
    prompts = []
    negative_prompts = []
    for first_img_filepath in tqdm(first_img_filepaths, desc="Getting prompts for each image"):
        img = np.array(Image.open(first_img_filepath))
        prompt_x, prompt_y = get_user_point(img, title=f"Click on the image to select a point (Frame 0) for {first_img_filepath}")
        negative_prompt_x, negative_prompt_y = get_user_point(img, title=f"Click on the image to select the NEGATIVE point (Frame 0) for {first_img_filepath}")
        prompts.append((prompt_x, prompt_y))
        negative_prompts.append((negative_prompt_x, negative_prompt_y))

    # Run SAM2 for each demo directory
    for demo_dir, prompt, negative_prompt in tqdm(zip(DEMO_DIRS, prompts, negative_prompts), desc="Running SAM2 for each demo directory", total=len(DEMO_DIRS)):
        input_dir = demo_dir / "rgb"
        output_dir = demo_dir.parent / "hand_mask"
        prompt_x, prompt_y = prompt
        negative_prompt_x, negative_prompt_y = negative_prompt
        run_video_sam2(
            input_dir=input_dir,
            output_dir=output_dir,
            use_negative_prompt=True,
            use_second_prompt=False,
            prompt_x=prompt[0],
            prompt_y=prompt[1],
            negative_prompt_x=negative_prompt[0],
            negative_prompt_y=negative_prompt[1],
        )

if __name__ == "__main__":
    main()