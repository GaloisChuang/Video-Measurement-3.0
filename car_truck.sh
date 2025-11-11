#!/bin/sh
python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/VidToMe/car_truck/output.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."

python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/videograin/car_truck/step_0.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."

python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/TRACE/car_truck/step_0.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."

python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/TokenFlow/car_truck/Car_truck.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow..."

python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/RAVE/car_truck/Car_truck.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"

python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/ControlVideo/car_truck/Car_truck.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"