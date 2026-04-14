#!/usr/bin/env bash
set -euo pipefail

print_step() {
    echo
    echo "########################################################################"
    echo "STEP $1: $2"
    echo "########################################################################"
}

# This script is a documented template for the hand-processing pipeline used by
# the SimToolReal kinematic retargeting baseline.
#
# At a high level it:
# 1. runs SAM2 on the original RGB frames to create `hand_mask/`
# 2. runs HaMeR Depth on rgb/depth/hand_mask to create `hand_pose_trajectory/`
#
# You should edit the configuration section below for your own machine.


################################################################################
# User configuration: update these for your own machine
################################################################################

# Path to this SAM2 repo.
SAM2_REPO="/home/tylerlum/github_repos/segment-anything-2-real-time"

# Path to the HaMeR Depth repo that contains run.py.
HAMER_DEPTH_REPO="/home/tylerlum/github_repos/hamer_depth"

# Input demo directory. Expected to contain rgb/, depth/, and cam_K.txt.
DEMO_DIR="/juno/u/kedia/FoundationPose/human_videos/Jan_17/brush/red_brush/sweep_forward"

# Where to write the SAM2 hand masks.
HAND_MASK_DIR="${DEMO_DIR}/hand_mask"

# Where to write the HaMeR Depth outputs (.json, .obj, .png per frame).
HAND_POSE_TRAJECTORY_DIR="${DEMO_DIR}/hand_pose_trajectory"

# Prompt arguments for the initial hand segmentation.
# Replace these with your own preferred prompting mode.
#
# Examples:
#   HAND_PROMPT_ARGS=(--prompt_x 643 --prompt_y 357)
#   HAND_PROMPT_ARGS=(--use_negative_prompt)
#   HAND_PROMPT_ARGS=(--use_second_prompt)
#
# Current default uses a negative point so you can click the hand first and then
# click the arm to exclude it when the colors are similar.
HAND_PROMPT_ARGS=(--use_negative_prompt)

# HaMeR Depth assumes right hands by default.
# Change this to LEFT if your demo contains a left hand.
HAND_TYPE="LEFT"

# Optional extra arguments for hamer_depth/run.py.
# Example:
#   HAMER_EXTRA_ARGS=(--ignore-exceptions)
HAMER_EXTRA_ARGS=()


################################################################################
# Environment helpers: replace these if your setup differs
################################################################################

sam2() {
    cd "${SAM2_REPO}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PYTHONPATH="${SAM2_REPO}:${PYTHONPATH:-}"
    "$@"
}

hamer_depth() {
    cd "${HAMER_DEPTH_REPO}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PYTHONPATH="${HAMER_DEPTH_REPO}:${PYTHONPATH:-}"
    "$@"
}

require_sam2_imports() {
    sam2 python - <<'PY'
import cv2
import torch
print("SAM2 env OK")
PY
}

require_hamer_depth_imports() {
    hamer_depth python - <<'PY'
import cv2
import tyro
from hamer_depth.detectors.detector_hamer import DetectorHamer
from hamer_depth.utils.hand_type import HandType
print("HaMeR Depth env OK")
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

if [[ ! -f "${DEMO_DIR}/cam_K.txt" ]]; then
    echo "Missing camera intrinsics file: ${DEMO_DIR}/cam_K.txt" >&2
    exit 1
fi

mkdir -p "${HAND_MASK_DIR}"
mkdir -p "${HAND_POSE_TRAJECTORY_DIR}"

print_step "0" "Checking SAM2 and HaMeR Depth environments"
require_sam2_imports
require_hamer_depth_imports


################################################################################
# Step 1: Run SAM2 on the original RGB frames to create hand masks
################################################################################

# If you use an interactive mode like --use_negative_prompt, the script will
# open the first frame and wait for your clicks.
print_step "1" "Running SAM2 on the original RGB frames to create hand masks"
sam2 python video_sam2.py \
    --input_dir "${DEMO_DIR}/rgb" \
    --output_dir "${HAND_MASK_DIR}" \
    "${HAND_PROMPT_ARGS[@]}"


################################################################################
# Step 2: Run HaMeR Depth to extract 3D hand poses
################################################################################

print_step "2" "Running HaMeR Depth to create hand_pose_trajectory"
hamer_depth python run.py \
    --rgb-path "${DEMO_DIR}/rgb" \
    --depth-path "${DEMO_DIR}/depth" \
    --mask-path "${HAND_MASK_DIR}" \
    --cam-intrinsics-path "${DEMO_DIR}/cam_K.txt" \
    --out-path "${HAND_POSE_TRAJECTORY_DIR}" \
    --hand-type "${HAND_TYPE}" \
    "${HAMER_EXTRA_ARGS[@]}"


################################################################################
# Final artifact
################################################################################

print_step "3" "Final artifacts"
echo "Expected hand mask directory:"
echo "  ${HAND_MASK_DIR}"
echo "Expected hand pose trajectory directory:"
echo "  ${HAND_POSE_TRAJECTORY_DIR}"
ls "${HAND_POSE_TRAJECTORY_DIR}" | head
