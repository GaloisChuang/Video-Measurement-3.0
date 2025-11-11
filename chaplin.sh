#!/bin/sh

python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/ControlVideo/Charlie-pillar/Chaplin.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/videograin/Chaplin/step_0.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/TRACE/Chaplin-pillar/step_0.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/TRACE/Chaplin-pillar/step_0.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/videograin/Chaplin/step_0.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/ControlVideo/Charlie-pillar/Chaplin.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"