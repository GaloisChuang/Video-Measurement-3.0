#!/bin/sh

python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/VidToMe/dog_cat/output.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/videograin/dog_cat/step_0.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/TRACE/dog_cat/step_0.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/TokenFlow/dog_cat/Dog_cat.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/RAVE/dog_cat/Dog_cat.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/ControlVideo/dog_cat/dog_cat.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"