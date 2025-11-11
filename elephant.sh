#!/bin/sh

python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/VidToMe/elephant/output.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/videograin/elephant/step_0.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/TRACE/elephant/step_0.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/TokenFlow/elephant/Elephant.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/RAVE/elephant/Elephant.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/ControlVideo/elephant/elephant.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"