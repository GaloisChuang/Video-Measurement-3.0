#!/bin/sh

python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/VidToMe/man-obj/output.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring VidToMe..."


python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/videograin/man-obj/step_0.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring Video Grain..."


python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/TRACE/man-obj/step_0.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TRACE..."


python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/Tokenflow/man-obj/man-obj.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring TokenFlow"


python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/RAVE/man-obj/man-obj.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring RAVE"


python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/ControlVideo/man-obj/amn_obj.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Measuring ControlVideo"