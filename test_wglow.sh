#/bin/bash
dataname="cifar10"
portion="test"
datasetroot="./datasets"
outdir="./prediction"
modelpath=""
nsamples=32
neachrow=8
nbits=8
batchsize=16
hidden=512
depth=8
level=3
export CUDA_VISIBLE_DEVICES=1
python inference_main.py --dataset_name=${dataname} --dataset_root=${datasetroot} --portion=${portion} --out_root=${outdir} --model_path=${modelpath} --n_samples=${nsamples} --sample_each_row=${neachrow} --n_bits=${nbits} --hidden_channels=${hidden} --flow_depth=${depth} --num_levels=${level} --batch_size=${batchsize}
