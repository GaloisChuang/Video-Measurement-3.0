#!/bin/sh

python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/VidToMe/lalaland/output.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/videograin/lalaland/step_0.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/TRACE/lalaland/step_0.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/TokenFlow/lalaland/Lalaland.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/RAVE/lalaland/Lalaland.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/ControlVideo/lalaland/step_0.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"