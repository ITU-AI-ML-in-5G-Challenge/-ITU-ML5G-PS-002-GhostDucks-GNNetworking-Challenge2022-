#!/usr/bin/env bash

# this script trains an oracle model. In our experiments, this training resulted in validation score of 6.15 at epoch 43

# uncomment if using conda environment and replace "gnnch" with your env name
#eval "$(conda shell.bash hook)"
#conda activate gnnch

# code root directory (containing datagen/datagen.py)
src=/home/yakovl/dev/GNNetworkingChallenge

# directory where to create the datasets (must exist)
datagen_root=${src}/generated_datasets_pkl

if [ ! -d "${datagen_root}" ]; then
  echo "${datagen_root} does not exist."
  exit 1
fi

train_dir=${src}/oracle_models/6.15
train_data=${train_dir}/data_pkl

mkdir -p ${train_dir}
mkdir ${train_data}

# copy subset of the data to the training data dir
cp -r ${datagen_root}/8 ${datagen_root}/10 ${datagen_root}/hard1 ${datagen_root}/hard4 ${datagen_root}/hard5 ${train_data}
mkdir ${train_data}/1 ${train_data}/15
cp -r ${datagen_root}/1/*_? ${train_data}/1
cp -r ${datagen_root}/15/*0 ${datagen_root}/15/*1 ${datagen_root}/15/*2 ${datagen_root}/15/*3 ${datagen_root}/15/*4 ${train_data}/15

cd ${train_dir} || exit

CUDA_VISIBLE_DEVICES=-1 python ${src}/custom_train.py -d ${train_data} -t ${src}/validation_dataset_pkl -nval 130 -lr 0.001 --decay 0.75 --decay-epochs 15 -steps 10000 -sample_mode all -epochs 200 -ckpt ${src}/RouteNet_Fermi/initial_weights/initial_weights --use_clearml False -task_name from-8-10-hard1-hard4-hard5-hard6-1-5k-15-5k_lr_1e-3_decay_0.75x15

