#!/bin/sh

python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/VidToMe/02_Roger/output.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/videograin/02_Roger/02_roger.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/TRACE/02_Roger/step_0.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/TokenFlow/02_Roger/02_walk_walk.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/RAVE/02_Roger/02_walk_walk.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/ControlVideo/02_Roger/02_roger.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"