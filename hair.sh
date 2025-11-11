#!/bin/sh

python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/VidToMe/hair/output.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/videograin/hair/step_0.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/TRACE/hair/step_0.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/TokenFlow/hair/Hair.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/RAVE/hair/Hair.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/ControlVideo/hair/hair.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"