#!/bin/sh
python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/VidToMe/3ball/output.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."

python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/videograin/3ball/step_0.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."

python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/TRACE/3ball/step_0.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."

python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/TokenFlow/3ball/3bowl.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow..."

python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/RAVE/3ball/3ball.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"

python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/ControlVideo/3ball/3ball.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"