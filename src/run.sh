#!/usr/bin/bash
model=decision_tree

python train.py --fold 0 --model $model
python train.py --fold 1 --model $model
python train.py --fold 2 --model $model
python train.py --fold 3 --model $model
python train.py --fold 4 --model $model