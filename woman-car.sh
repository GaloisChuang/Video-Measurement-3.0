#!/bin/sh

python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/VidToMe/woman-car/output.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/videograin/woman_car/step_0.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/TRACE/woman-car/step_0.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/Tokenflow/woman-car/woman-car.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/RAVE/woman-car/woman-car.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/ControlVideo/woman-car/woman_car.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"