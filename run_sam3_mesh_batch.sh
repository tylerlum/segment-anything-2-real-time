#!/usr/bin/env bash
set -euo pipefail

SAM3_REPO="${SAM3_REPO:-/home/tylerlum/github_repos/sam-3d-objects}"
DATA_ROOT="${DATA_ROOT:-/juno/u/yufeid/project_storage/shared_folder/input_data/gt_depth/gt_depth_0323}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SAM3_REPO}/2026-04-30_mesh_outputs}"
TASK_GLOB="${TASK_GLOB:-*}"
MASK_GLOB="${MASK_GLOB:-*masks*}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MESH_MODE="${MESH_MODE:-texture}"
TEXTURE_SIZE="${TEXTURE_SIZE:-1024}"
TEXTURE_RENDER_RESOLUTION="${TEXTURE_RENDER_RESOLUTION:-1024}"
TEXTURE_NVIEWS="${TEXTURE_NVIEWS:-100}"

usage() {
    cat <<EOF
Usage: $0 [options]

Generate SAM3D meshes for every task/object under a data root.
Inputs are read-only. All staged inputs and mesh outputs are written under OUTPUT_ROOT.

Options:
  --data-root PATH      Root containing task folders with rgb/, depth/, cam_K.txt
  --output-root PATH    Local output root for meshes
  --sam3-repo PATH      sam-3d-objects repo path
  --task-glob GLOB      Task folder glob under DATA_ROOT, default: *
  --mask-glob GLOB      Mask folder glob inside each task, default: *masks*
  --mesh-mode MODE      SAM3D mesh mode: texture or vertex_color, default: texture
  --texture-size N      Texture atlas size for texture mode, default: 1024
  --texture-render-resolution N
                         Render resolution for texture baking, default: 1024
  --texture-nviews N    Number of texture-baking views, default: 100
  --no-skip-existing    Re-run tasks even if mesh/mesh.obj already exists
  --dry-run             Print planned jobs without running SAM3D
  -h, --help            Show this help

Example:
  $0 \\
    --data-root /juno/u/yufeid/project_storage/shared_folder/input_data/gt_depth/gt_depth_0323 \\
    --output-root /home/tylerlum/github_repos/sam-3d-objects/2026-04-30_mesh_outputs
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --sam3-repo)
            SAM3_REPO="$2"
            shift 2
            ;;
        --task-glob)
            TASK_GLOB="$2"
            shift 2
            ;;
        --mask-glob)
            MASK_GLOB="$2"
            shift 2
            ;;
        --mesh-mode)
            MESH_MODE="$2"
            shift 2
            ;;
        --texture-size)
            TEXTURE_SIZE="$2"
            shift 2
            ;;
        --texture-render-resolution)
            TEXTURE_RENDER_RESOLUTION="$2"
            shift 2
            ;;
        --texture-nviews)
            TEXTURE_NVIEWS="$2"
            shift 2
            ;;
        --no-skip-existing)
            SKIP_EXISTING=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

print_step() {
    echo
    echo "########################################################################"
    echo "$1"
    echo "########################################################################"
}

sam3() {
    cd "${SAM3_REPO}"
    # shellcheck disable=SC1091
    source .venv311/bin/activate
    export PYTHONPATH="${SAM3_REPO}:${PYTHONPATH:-}"
    "$@"
}

sanitize_name() {
    local raw="$1"
    raw="${raw##*masks}"
    raw="${raw#_}"
    raw="${raw#-}"
    raw="${raw#.}"
    raw="${raw//\*/}"
    raw="${raw#_}"
    raw="${raw#-}"
    raw="${raw#.}"
    raw="${raw%.}"
    if [[ -z "${raw}" ]]; then
        raw="object"
    fi
    printf '%s' "${raw}" | tr -cs '[:alnum:]_-' '_'
}

stage_inputs() {
    local rgb_dir="$1"
    local depth_dir="$2"
    local cam_k_path="$3"
    local masks_dir="$4"
    local stage_dir="$5"

    export RGB_DIR="${rgb_dir}"
    export DEPTH_DIR="${depth_dir}"
    export CAM_K_PATH="${cam_k_path}"
    export MASKS_DIR="${masks_dir}"
    export STAGE_DIR="${stage_dir}"

    sam3 python - <<'PY'
import os
import shutil
from pathlib import Path

from PIL import Image

rgb_dir = Path(os.environ["RGB_DIR"])
depth_dir = Path(os.environ["DEPTH_DIR"])
cam_k_path = Path(os.environ["CAM_K_PATH"])
masks_dir = Path(os.environ["MASKS_DIR"])
stage_dir = Path(os.environ["STAGE_DIR"])

rgb_files = sorted(rgb_dir.glob("*.png"))
depth_files = sorted(depth_dir.glob("*.png"))
mask_files = sorted(masks_dir.glob("*.png"))

if not rgb_files:
    raise SystemExit(f"No RGB frames found in {rgb_dir}")
if not depth_files:
    raise SystemExit(f"No depth frames found in {depth_dir}")
if not mask_files:
    raise SystemExit(f"No mask frames found in {masks_dir}")

stage_dir.mkdir(parents=True, exist_ok=True)
for child in ["rgb", "masks", "depth", "cam_K.txt"]:
    path = stage_dir / child
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)

with Image.open(rgb_files[0]) as rgb_img, Image.open(depth_files[0]) as depth_img, Image.open(mask_files[0]) as mask_img:
    rgb_size = rgb_img.size
    depth_size = depth_img.size
    mask_size = mask_img.size

if rgb_size == depth_size:
    (stage_dir / "rgb").symlink_to(rgb_dir)
    print(f"RGB size matches depth {depth_size}; symlinked rgb/")
else:
    (stage_dir / "rgb").mkdir()
    print(f"Warning: resizing RGB frames from {rgb_size[::-1]} to depth size {depth_size[::-1]}")
    for rgb_path in rgb_files:
        with Image.open(rgb_path).convert("RGB") as rgb_img:
            rgb_img.resize(depth_size, Image.Resampling.BILINEAR).save(stage_dir / "rgb" / rgb_path.name)

if mask_size == depth_size:
    (stage_dir / "masks").symlink_to(masks_dir)
    print(f"Mask size matches depth {depth_size}; symlinked masks/")
else:
    (stage_dir / "masks").mkdir()
    print(f"Warning: resizing mask frames from {mask_size[::-1]} to depth size {depth_size[::-1]}")
    for mask_path in mask_files:
        with Image.open(mask_path) as mask_img:
            mask_img.resize(depth_size, Image.Resampling.NEAREST).save(stage_dir / "masks" / mask_path.name)

(stage_dir / "depth").symlink_to(depth_dir)
(stage_dir / "cam_K.txt").symlink_to(cam_k_path)
print(f"Staged SAM3D inputs in {stage_dir}")
PY
}

run_one() {
    local task_dir="$1"
    local masks_dir="$2"

    local task_name object_name output_dir stage_dir
    task_name="$(basename "${task_dir}")"
    object_name="$(sanitize_name "$(basename "${masks_dir}")")"
    output_dir="${OUTPUT_ROOT}/${task_name}/${object_name}"
    stage_dir="${output_dir}/sam3_input"

    if [[ "${object_name}" == *robot_arm* ]]; then
        echo "Skipping robot arm masks: ${masks_dir}"
        return
    fi

    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/mesh/mesh.obj" ]]; then
        echo "Skipping existing mesh: ${output_dir}/mesh/mesh.obj"
        return
    fi

    echo "Task: ${task_name}"
    echo "Object: ${object_name}"
    echo "RGB: ${task_dir}/rgb"
    echo "Depth: ${task_dir}/depth"
    echo "cam_K: ${task_dir}/cam_K.txt"
    echo "Masks: ${masks_dir}"
    echo "Output: ${output_dir}"
    echo "Mesh mode: ${MESH_MODE}"
    echo "Texture size: ${TEXTURE_SIZE}"
    echo "Texture render resolution: ${TEXTURE_RENDER_RESOLUTION}"
    echo "Texture nviews: ${TEXTURE_NVIEWS}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi

    stage_inputs "${task_dir}/rgb" "${task_dir}/depth" "${task_dir}/cam_K.txt" "${masks_dir}" "${stage_dir}"
    sam3 python run_inference.py \
        --input_dir "${stage_dir}" \
        --output_dir "${output_dir}" \
        --mesh_mode "${MESH_MODE}" \
        --texture_size "${TEXTURE_SIZE}" \
        --texture_render_resolution "${TEXTURE_RENDER_RESOLUTION}" \
        --texture_nviews "${TEXTURE_NVIEWS}" \
        --non-interactive
}

print_step "Checking SAM3D environment"
sam3 python - <<'PY'
import PIL
import trimesh
import tyro
print("SAM3D env OK")
PY

print_step "Discovering mesh generation jobs"
shopt -s nullglob
task_dirs=("${DATA_ROOT}"/${TASK_GLOB})
if [[ ${#task_dirs[@]} -eq 0 ]]; then
    echo "No task folders matched: ${DATA_ROOT}/${TASK_GLOB}" >&2
    exit 1
fi

job_count=0
for task_dir in "${task_dirs[@]}"; do
    [[ -d "${task_dir}" ]] || continue
    [[ -d "${task_dir}/rgb" && -d "${task_dir}/depth" && -f "${task_dir}/cam_K.txt" ]] || {
        echo "Skipping incomplete task folder: ${task_dir}"
        continue
    }
    mask_dirs=("${task_dir}"/${MASK_GLOB})
    for masks_dir in "${mask_dirs[@]}"; do
        [[ -d "${masks_dir}" ]] || continue
        [[ "$(basename "${masks_dir}")" == *tracking_results* ]] && continue
        [[ "$(basename "${masks_dir}")" == *robot_arm* ]] && continue
        job_count=$((job_count + 1))
        print_step "Job ${job_count}"
        run_one "${task_dir}" "${masks_dir}"
    done
done

print_step "Done"
echo "Processed ${job_count} mesh generation job(s)."
echo "Mesh output root: ${OUTPUT_ROOT}"
