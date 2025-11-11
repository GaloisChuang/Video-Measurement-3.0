#!/bin/sh

# python measure.py \
#   --orig data/input/n_u_c_s/source_video.mp4 \
#   --edit data/p2o/VidToMe/n_u_c_s/output.mp4 \
#   --spec data/config/n_u_c_s.json \
#   --device cuda \
#   --sample_stride 1 \
#   --max_frames 15 \
#   --save_json ./metrics.json
# echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/videograin/n_u_c_s/step_0.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/TRACE/n_u_c_s/step_0.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/TokenFlow/n_u_c_s/n_u_c_s.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/RAVE/n_u_c_s/n_u_c_s.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/ControlVideo/n_u_c_s/step_0.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"