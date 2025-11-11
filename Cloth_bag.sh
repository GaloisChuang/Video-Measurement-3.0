#!/bin/sh
python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/VidToMe/cloth_bag/output.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."

python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/videograin/cloth_bag/step_0.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."

python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/TRACE/cloth_bag/step_0.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."

python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/TokenFlow/cloth_bag/cloth-bag.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow..."

python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/RAVE/cloth_bag/cloth-bag.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"

python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/ControlVideo/cloth_bag/cloth_bag.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"