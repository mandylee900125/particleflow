#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec -B /scratch --nv $IMG python3 mlpf/tensorflow/tf_model.py \
  --datapath /scratch/joosep/TTbar_14TeV_TuneCUETP8M1_cfi \
  --target cand --ntrain 70000 --ntest 20000 --convlayer ghconv \
  --lr 1e-3 --nepochs 1000 --nbins 64 --attention-layer-cutoff 0.01 \
  --bin_size 100 --hidden-dim 512 --batch-size 5 --distance-dim 512 \
  --dropout 0.2 --train-cls --weights experiments/run_28/weights.17-69.778923.hdf5

#Pytorch  training
#singularity exec -B /home --nv $IMG \
#  python3 test/train_end2end.py \
#  --dataset /home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi \
#  --n_train 400 --n_val 100 \
#  --model PFNet7 --convlayer gravnet-radius --lr 0.005 \
#  --hidden_dim 32 --n_epochs 100 \
#  --l1 1000.0 --l2 100.0 --l3 1000.0 --space_dim 2 --nearest 5 --convlayer2 sgconv \
#  --target cand --batch_size 1 --activation leaky_relu \
#  --dropout 0.0 --encoding_dim 256 --optimizer adamw --radius 0.01 --input-encoding 0
