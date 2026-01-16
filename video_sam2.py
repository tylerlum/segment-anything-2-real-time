import os
from typing import Optional, Tuple

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def get_user_point(rgb_image: np.ndarray, title: str) -> Tuple[int, int]:
    # Get prompt as click
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.imshow(rgb_image)
    plt.axis("off")
    points = plt.ginput(1)  # get one click
    plt.close()

    x, y = int(points[0][0]), int(points[0][1])
    return x, y


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
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    # Create the SAM2 predictor
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = Path(__file__).parent / "checkpoints/sam2_hiera_large.pt"
    assert sam2_checkpoint.exists(), f"SAM2 checkpoint not found: {sam2_checkpoint}"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(
        config_file=model_cfg, ckpt_path=sam2_checkpoint, device=device
    )

    def show_mask(
        mask: np.ndarray,
        ax: plt.Axes,
        obj_id: Optional[int] = None,
        random_color: bool = False,
    ) -> None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(
        coords: np.ndarray, labels: np.ndarray, ax: plt.Axes, marker_size: int = 200
    ) -> None:
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    # WARNING: SAM2 requires jpg images, so if they are png, we need to convert them to jpg
    # scan all the JPEG frame names in this directory
    jpg_filepaths = sorted(list(input_dir.glob("*.jpg")))
    png_filepaths = sorted(list(input_dir.glob("*.png")))
    if len(jpg_filepaths) == 0 and len(png_filepaths) == 0:
        raise ValueError("No frames found in the input directory")
    elif len(jpg_filepaths) == 0:
        print(
            f"Found {len(jpg_filepaths)} jpg frames and {len(png_filepaths)} png frames"
        )
        print("Converting png frames to jpg")
        for png_filepath in png_filepaths:
            with Image.open(png_filepath) as img:
                rgb_img = img.convert("RGB")
                jpg_file = png_filepath.with_suffix(".jpg")
                rgb_img.save(jpg_file, "JPEG")
            jpg_filepaths.append(jpg_file)
    else:
        assert len(jpg_filepaths) == len(png_filepaths), (
            "The number of jpg and png frames must be the same, something might be wrong..."
        )
        print(
            f"Found {len(jpg_filepaths)} jpg frames and {len(png_filepaths)} png frames"
        )
        print("Using the jpg frames")

    assert len(jpg_filepaths) > 0, "No frames found in the input directory"

    if visualize:
        # take a look the first video frame
        frame_idx = 0
        print("=" * 80)
        print(f"DEBUG: Visualizing the frame {frame_idx}")
        print("=" * 80 + "\n")
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {frame_idx}")
        plt.imshow(Image.open(jpg_filepaths[frame_idx]))
        plt.show()

    inference_state = predictor.init_state(video_path=str(input_dir))

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = (
        1  # give a unique id to each object we interact with (it can be any integers)
    )

    # Get prompt as click
    img = np.array(Image.open(jpg_filepaths[ann_frame_idx]))
    if prompt_x is not None and prompt_y is not None:
        x, y = prompt_x, prompt_y
        print(f"Using provided point: ({x}, {y})")
    else:
        x, y = get_user_point(
            rgb_image=img,
            title=f"Click on the image to select a point (Frame {ann_frame_idx})",
        )
        print(f"Clicked point: ({x}, {y})")

    if use_negative_prompt:
        if negative_prompt_x is not None and negative_prompt_y is not None:
            neg_x, neg_y = negative_prompt_x, negative_prompt_y
            print(f"Using provided negative point: ({neg_x}, {neg_y})")
        else:
            # Get negative prompt as click
            neg_x, neg_y = get_user_point(
                rgb_image=img,
                title=f"Click on the image to select the NEGATIVE point (Frame {ann_frame_idx})",
            )
            print(f"Clicked point: ({neg_x}, {neg_y})")

        points = np.array([[x, y], [neg_x, neg_y]], dtype=np.float32)

        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 0], dtype=np.int32)
    elif use_second_prompt:
        if second_prompt_x is not None and second_prompt_y is not None:
            second_x, second_y = second_prompt_x, second_prompt_y
            print(f"Using provided second point: ({second_x}, {second_y})")
        else:
            # Get second prompt as click
            second_x, second_y = get_user_point(
                rgb_image=img,
                title=f"Click on the image to select the SECOND point (Frame {ann_frame_idx})",
            )
        print(f"Clicked point: ({second_x}, {second_y})")

        points = np.array([[x, y], [second_x, second_y]], dtype=np.float32)

        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1, 1], dtype=np.int32)
    else:
        points = np.array([[x, y]], dtype=np.float32)

        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], dtype=np.int32)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    if visualize:
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(jpg_filepaths[ann_frame_idx]))
        show_points(points, labels, plt.gca())
        show_mask(
            (out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0]
        )
        plt.show()

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    output_dir.mkdir(parents=True, exist_ok=True)
    for out_frame_idx in range(0, len(jpg_filepaths), vis_frame_stride):
        mask = video_segments[out_frame_idx][1]

        if mask.ndim == 2:
            height, width = mask.shape
        elif mask.ndim == 3:
            assert mask.shape[0] == 1 or mask.shape[-1] == 1, (
                f"The mask should be a single channel, but got shape {mask.shape}"
            )
            if mask.shape[0] == 1:
                height, width = mask.shape[1:]
                mask = mask.squeeze(axis=0)
            else:
                height, width = mask.shape[:-1]
                mask = mask.squeeze(axis=-1)
        else:
            raise ValueError(f"Invalid mask shape: {mask.shape}")

        mask_img = np.zeros((height, width, 3))
        mask_img[mask > 0] = [255, 255, 255]  # object is in white
        cv2.imwrite(
            str(output_dir / f"{out_frame_idx:05d}.png"),
            mask_img,
        )

        if visualize:
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(jpg_filepaths[out_frame_idx]))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.axis("off")
            plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)

    # Flags to ask user to select more than one point
    parser.add_argument("--use_negative_prompt", action="store_true")
    parser.add_argument("--use_second_prompt", action="store_true")

    # Coordinates for the prompt to be passed in without user interaction
    parser.add_argument("--prompt_x", type=int, default=None, help="X coordinate of the prompt")
    parser.add_argument("--prompt_y", type=int, default=None, help="Y coordinate of the prompt")
    parser.add_argument("--negative_prompt_x", type=int, default=None, help="X coordinate of the negative prompt")
    parser.add_argument("--negative_prompt_y", type=int, default=None, help="Y coordinate of the negative prompt")
    parser.add_argument("--second_prompt_x", type=int, default=None, help="X coordinate of the second prompt")
    parser.add_argument("--second_prompt_y", type=int, default=None, help="Y coordinate of the second prompt")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    run_video_sam2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_negative_prompt=args.use_negative_prompt,
        use_second_prompt=args.use_second_prompt,
        prompt_x=args.prompt_x,
        prompt_y=args.prompt_y,
        negative_prompt_x=args.negative_prompt_x,
        negative_prompt_y=args.negative_prompt_y,
        second_prompt_x=args.second_prompt_x,
        second_prompt_y=args.second_prompt_y,
        visualize=args.visualize,
    )

if __name__ == "__main__":
    main()