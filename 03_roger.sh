#!/bin/sh
python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/VidToMe/03_Roger/output.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."

python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/videograin/03_Roger/step_0.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."

python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/TRACE/03_Roger/step_0.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."

python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/TokenFlow/03_Roger/03_with_walk.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow..."

python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/RAVE/03_Roger/03_with_walk.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"

python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/ControlVideo/03_Roger/step_0.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"