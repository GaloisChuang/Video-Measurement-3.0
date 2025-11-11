#!/bin/sh

python measure.py \
  --orig data/input/02_roger/source_video.mp4 \
  --edit data/p2p/GAV/02_Roger/02_roger.mp4 \
  --spec data/config/02_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "02_roger done"

  python measure.py \
  --orig data/input/03_roger/source_video.mp4 \
  --edit data/p2p/GAV/03_Roger/03_roger.mp4 \
  --spec data/config/03_roger.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "03_roger done"

  python measure.py \
  --orig data/input/3bowl/source_video.mp4 \
  --edit data/o2o/GAV/3ball/3ball.mp4 \
  --spec data/config/3bowl.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "3ball done"

  python measure.py \
  --orig data/input/bag_cloth/source_video.mp4 \
  --edit data/o2o/GAV/cloth_bag/cloth-bag.mp4 \
  --spec data/config/bag_cloth.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "cloth_bag done"

  python measure.py \
  --orig data/input/car_truck/source_video.mp4 \
  --edit data/o2o/GAV/car_truck/car_truck.mp4 \
  --spec data/config/car_truck.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "car_truck done"

python measure.py \
  --orig data/input/Chaplin/source_video.mp4 \
  --edit data/p2o/GAV/Charlie-pillar/Charlie-pillar.mp4 \
  --spec data/config/Chaplin_pillar.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "Chaplin_pillar done"

  python measure.py \
  --orig data/input/chinese_palace/source_video.mp4 \
  --edit data/p2p/GAV/chinese_palace/chinese_palace.mp4 \
  --spec data/config/chinese_palace.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "chinese_palace done"

  python measure.py \
  --orig data/input/dog_cat/source_video.mp4 \
  --edit data/o2o/GAV/dog_cat/dog-cat.mp4 \
  --spec data/config/dog_cat.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "dog_cat done"

  python measure.py \
  --orig data/input/elephant/source_video.mp4 \
  --edit data/o2o/GAV/elephant/elephant.mp4 \
  --spec data/config/elephant.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "elephant done"

  python measure.py \
  --orig data/input/hair/source_video.mp4 \
  --edit data/o2o/GAV/hair/hair.mp4 \
  --spec data/config/hair.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "hair done"

python measure.py \
  --orig data/input/lalaland/source_video.mp4 \
  --edit data/p2p/GAV/lalaland/lalaland.mp4 \
  --spec data/config/lalaland.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "lalaland done"

  python measure.py \
  --orig data/input/man-obj/source_video.mp4 \
  --edit data/p2o/GAV/man-obj/man_obj.mp4 \
  --spec data/config/man_obj.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "man_obj done"

  python measure.py \
  --orig data/input/n_u_c_s/source_video.mp4 \
  --edit data/p2p/GAV/n_u_c_s/n_u_c_s.mp4 \
  --spec data/config/n_u_c_s.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "n_u_c_s done"

  python measure.py \
  --orig data/input/u_name/source_video.mp4 \
  --edit data/p2p/GAV/u_name/u_name.mp4 \
  --spec data/config/u_name.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "u_name done"

  python measure.py \
  --orig data/input/woman_car/source_video.mp4 \
  --edit data/p2o/GAV/woman-car/woman_car.mp4 \
  --spec data/config/woman_car.json \
  --device cuda \
  --sample_stride 1 \
  --max_frames 15 \
  --save_json ./metrics.json
echo "woman_car done"