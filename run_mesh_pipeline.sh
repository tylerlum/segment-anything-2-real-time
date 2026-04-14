#!/usr/bin/env bash
set -euo pipefail

# This script is a documented template for the full SAM2 -> SAM3 -> mesh postprocess pipeline.
#
# You should edit the configuration section below for your own machine.
# In particular, replace repo paths, dataset paths, and prompt arguments as needed.
#
# For Tyler's current setup:
#   SAM2 repo: /home/tylerlum/github_repos/segment-anything-2-real-time/
#   SAM3 repo: /home/tylerlum/github_repos/sam-3d-objects/


################################################################################
# User configuration: update these for your own machine
################################################################################

# Path to this SAM2 repo.
SAM2_REPO="/home/tylerlum/github_repos/segment-anything-2-real-time"

# Path to the SAM3 repo that contains run_inference.py.
SAM3_REPO="/home/tylerlum/github_repos/sam-3d-objects"

# Input demo directory. Expected to contain rgb/ and depth/.
DEMO_DIR="/juno/u/kedia/FoundationPose/human_videos/Jan_17/spatula/spoon_spatula/flip_pancake"

# Output directory for SAM3 mesh results and downstream processed assets.
OUTPUT_DIR="${SAM3_REPO}/outputs/spoon_spatula/flip_pancake"

# Where to write the initial SAM2 object masks on the original RGB video.
OBJECT_MASK_DIR="${DEMO_DIR}/masks"

# Prompt arguments for the initial object segmentation.
# Replace these with your own preferred prompting mode.
#
# Examples:
#   OBJECT_PROMPT_ARGS=(--prompt_x 664 --prompt_y 335)
#   OBJECT_PROMPT_ARGS=(--use_second_prompt)         # interactive two-positive-click mode
#   OBJECT_PROMPT_ARGS=(--use_negative_prompt)       # interactive positive+negative-click mode
OBJECT_PROMPT_ARGS=(--use_second_prompt)

# Prompt arguments for the handle and head masks on the rendered SAM3 RGB frames.
# These are usually separate interactive runs, so negative prompts are often useful.
HANDLE_PROMPT_ARGS=(--use_negative_prompt)
HEAD_PROMPT_ARGS=(--use_negative_prompt)


################################################################################
# Environment helpers: replace these if your setup differs
################################################################################

sam2() {
    # Activate the SAM2 environment and run commands from the SAM2 repo.
    #
    # Edit this function for your machine if:
    # - your SAM2 repo lives elsewhere
    # - your environment activation differs
    # - you use conda instead of uv
    #
    # Current Tyler setup:
    # - repo: /home/tylerlum/github_repos/segment-anything-2-real-time
    # - env:  source .venv/bin/activate
    cd "${SAM2_REPO}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PYTHONPATH="${SAM2_REPO}:${PYTHONPATH:-}"
    "$@"
}

sam3() {
    # Activate the SAM3 environment and run commands from the SAM3 repo.
    #
    # Edit this function for your machine if:
    # - your SAM3 repo lives elsewhere
    # - your SAM3 environment activation differs
    #
    # Current Tyler setup uses the SAM3 local .venv311.
    # If your SAM3 setup is conda-based, replace the source line with your own activation.
    #
    # Note: run_inference.py imports `viser` at top level, so `viser` must be installed
    # in the SAM3 environment, not the SAM2 environment.
    cd "${SAM3_REPO}"
    # shellcheck disable=SC1091
    source .venv311/bin/activate
    export PYTHONPATH="${SAM3_REPO}:${PYTHONPATH:-}"
    "$@"
}

require_sam2_imports() {
    sam2 python - <<'PY'
import cv2
import torch
print("SAM2 env OK")
PY
}

require_sam3_imports() {
    sam3 python - <<'PY'
import trimesh
import tyro
import viser
print("SAM3 env OK")
PY
}


################################################################################
# Sanity checks
################################################################################

if [[ ! -d "${DEMO_DIR}/rgb" ]]; then
    echo "Missing RGB directory: ${DEMO_DIR}/rgb" >&2
    exit 1
fi

if [[ ! -d "${DEMO_DIR}/depth" ]]; then
    echo "Missing depth directory: ${DEMO_DIR}/depth" >&2
    exit 1
fi

mkdir -p "${OBJECT_MASK_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Fail early with a clear message if either environment is missing required deps.
require_sam2_imports
require_sam3_imports


################################################################################
# Step 1: Run SAM2 on the original RGB frames to create object masks
################################################################################

# Replace OBJECT_PROMPT_ARGS above with your own prompt mode.
# If you use an interactive mode like --use_second_prompt, the script will open
# the first frame and wait for your clicks.
sam2 python video_sam2.py \
    --input_dir "${DEMO_DIR}/rgb" \
    --output_dir "${OBJECT_MASK_DIR}" \
    "${OBJECT_PROMPT_ARGS[@]}"


################################################################################
# Step 2: Run SAM3 to reconstruct the object mesh from rgb/depth/object masks
################################################################################

# This assumes SAM3's run_inference.py expects:
# - input_dir containing rgb/, depth/, and masks/
# - output_dir where mesh/, rgb/, depth/, cam_poses.npy, etc. will be written
sam3 python run_inference.py \
    --input_dir "${DEMO_DIR}" \
    --output_dir "${OUTPUT_DIR}"


################################################################################
# Step 3: Render the mesh into an orbit video and save RGB/depth/camera data
################################################################################

# This writes rendered RGB/depth frames into OUTPUT_DIR for downstream part masking.
sam2 python create_mesh_video.py \
    --mesh-filepath "${OUTPUT_DIR}/mesh/mesh.obj" \
    --output_dir "${OUTPUT_DIR}"


################################################################################
# Step 4: Run SAM2 on rendered frames to get handle masks
################################################################################

# Replace HANDLE_PROMPT_ARGS above with your own prompt mode if needed.
sam2 python video_sam2.py \
    --input_dir "${OUTPUT_DIR}/rgb" \
    --output_dir "${OUTPUT_DIR}/handle_masks" \
    "${HANDLE_PROMPT_ARGS[@]}"


################################################################################
# Step 5: Run SAM2 on rendered frames to get head masks
################################################################################

# Replace HEAD_PROMPT_ARGS above with your own prompt mode if needed.
sam2 python video_sam2.py \
    --input_dir "${OUTPUT_DIR}/rgb" \
    --output_dir "${OUTPUT_DIR}/head_masks" \
    "${HEAD_PROMPT_ARGS[@]}"


################################################################################
# Step 6: Merge masks across views, compute handle frame, crop handle mesh
################################################################################

sam2 python process_mesh.py \
    --output_dir "${OUTPUT_DIR}"


################################################################################
# Final artifact
################################################################################

echo "Expected transformed mesh:"
echo "  ${OUTPUT_DIR}/mesh_handle_frame/mesh_handle_frame.obj"
ls "${OUTPUT_DIR}/mesh_handle_frame/mesh_handle_frame.obj"
