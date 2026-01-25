#!/usr/bin/env zsh
source ~/.zshrc

# Get directory with rgb and depth directories
export DEMO_DIR=/juno/u/kedia/FoundationPose/human_videos/Jan_17/hammer/toy_hammer/down_swing/
export OUTPUT_DIR=/home/tylerlum/github_repos/sam-3d-objects/outputs/toy_hammer/down_swing

# Run sam2 to get the object mask (only need on first frame, but can do on all)
sam2_ros_env
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
python video_sam2.py \
--input_dir $DEMO_DIR/rgb/ \
--output_dir $DEMO_DIR/masks/ \
--use_second_prompt  

# Run sam3 to get object mesh
sam3
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
python run_inference.py \
--input_dir $DEMO_DIR \
--output_dir $OUTPUT_DIR

# Create video of mesh
sam2_ros_env
python create_mesh_video.py \
--mesh-filepath $OUTPUT_DIR/mesh/mesh.obj 

# Get masks of handle and head
python video_sam2.py \
--input_dir $DEMO_DIR/rgb/ \
--output_dir $DEMO_DIR/handle_masks/ \
--use_negative_prompt  

python video_sam2.py \
--input_dir $DEMO_DIR/rgb/ \
--output_dir $DEMO_DIR/head_masks/ \
--use_negative_prompt  

# Create modified mesh
python process_mesh.py \
--output_dir $OUTPUT_DIR

# Get resulting mesh
ls $OUTPUT_DIR/mesh_handle_frame/mesh_handle_frame.obj