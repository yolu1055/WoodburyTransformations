#!/bin/bash
dataname="cifar10"
trainportion="train"
validportion="test"
datasetroot="./datasets"
outdir="./results"
hidden=512
depth=8
level=3
epochs=100
batchsize=16
gradclip=0.1
gradnorm=0.1
lr=0.001
decay=0.0000002
nll_gap=100
inference_gap=2000
save_gap=20000
nbits=8
numsamples=20
numeachrow=5
export CUDA_VISIBLE_DEVICES=1
python train_main.py --dataset_name=${dataname} --dataset_root=${datasetroot} --train_portion=${trainportion} --valid_portion=${validportion} --out_root=${outdir} --hidden_channels=${hidden} --flow_depth=${depth} --num_levels=${level} --num_epochs=${epochs} --batch_size=${batchsize} --lr=${lr} --decay=${decay} --max_grad_clip=${gradclip} --max_grad_norm=${gradnorm} --nll_gap=${nll_gap} --inference_gap=${inference_gap} --save_gap=${save_gap} --n_bits=${nbits} --n_samples=${numsamples} --sample_each_row=${numeachrow}
