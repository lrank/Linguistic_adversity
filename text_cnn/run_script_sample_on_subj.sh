#!/bin/bash

python  train.py --dataset="subj" --is_noise_train=False --is_noise_test=False --noise_type="cf=0.5" --dropout_keep_prob=1.0

echo "dropout 0.5"
python  train.py --dataset="subj" --is_noise_train=False --is_noise_test=False --noise_type="cf=0.5" --dropout_keep_prob=0.5

echo "cf 0.5"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="cf=0.5" --dropout_keep_prob=1.0

echo "cf 1"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="cf=1.0" --dropout_keep_prob=1.0

echo "wn 0.5"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="wn=0.5" --dropout_keep_prob=1.0

echo "wn 1.0"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="wn=1.0" --dropout_keep_prob=1.0

echo "erg"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="erg" --dropout_keep_prob=1.0

echo "sc"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="sc" --dropout_keep_prob=1.0

echo "subj"
python  train.py --dataset="subj" --is_noise_train=True --is_noise_test=False --noise_type="all" --dropout_keep_prob=1.0