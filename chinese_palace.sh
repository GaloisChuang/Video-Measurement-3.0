#!/bin/sh

python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/VidToMe/chinese_palace/output.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/videograin/chinese_palace/step_0.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/TRACE/chinese_palace/step_0.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/TokenFlow/chinese_palace/Chinese_palace.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/RAVE/chinese_palace/Chinese_palace.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/ControlVideo/chinese_palace/step_0.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"