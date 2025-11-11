#!/bin/sh

python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/VidToMe/u_name/output.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/TRACE/u_name/step_0.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/TRACE/n_u_c_s/step_0.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/TokenFlow/u_name/Uname.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/RAVE/u_name/Uname.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/ControlVideo/u_name/step_0.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"