import torch
from models import WoodburyGlow
import argparse
from GlowLearner import Inferencer
from utils import load_state, count_parameters
from datasets import load_cifar10
import os

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='infer WGlow')

    # input path
    parser.add_argument("-d", "--dataset_name", type=str, default="cifar10")
    parser.add_argument("-p", "--portion", type=str, default="test")
    parser.add_argument("-r", "--dataset_root", type=str, default="./datasets")
    parser.add_argument("--model_path", type=str, default="")

    # output path
    parser.add_argument("--out_root", type=str, default="./prediction")

    # data parameters
    parser.add_argument("--n_bits", type=int, default=5)

    # Glow parameters
    parser.add_argument("--image_shape", type=tuple, default=(3,32,32))
    parser.add_argument("--hidden_channels", type=int, default=5)
    parser.add_argument("-K", "--flow_depth", type=int, default=3)
    parser.add_argument("-L", "--num_levels", type=int, default=3)
    parser.add_argument("--is_me", type=bool, default=False)
    parser.add_argument("--d_c", default=[16,16,16], help="the rank of U_c, V_c")
    parser.add_argument("--d_s1", default=[16,16,16],
                        help="if ME-Woodbury, the rank of U_h, V_h, else, the rank of U_s, V_s ")
    parser.add_argument("--d_s2", default=[16,16,16], help="the rank of U_w, V_w")
    parser.add_argument("--learn_top", type=bool, default=True)


    # Infer parameters
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--sample_each_row", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()


    if not os.path.exists(args.dataset_root):
        os.makedirs(args.dataset_root)

    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)

    # dataset
    dataset = load_cifar10(args.dataset_root, 'test')
    #dataset = None


    # model
    ranks = []
    if args.is_me:
        ranks = [args.d_c, args.d_s1, args.d_s2]
    else:
        ranks = [args.d_c, args.d_s1]

    args.ranks = ranks

    model = WoodburyGlow(args)
    if args.cuda:
        model = model.cuda()
    assert args.model_path != "", (print("need to load a model"))
    state = load_state(args.model_path, args.cuda)
    model.load_state_dict(state["model"])
    del state

    print("number of parameters: {}".format(count_parameters(model)))



    # begin to test
    inferencer = Inferencer(model, dataset, args)
    #inferencer.Inference()
    inferencer.Sample(args.n_samples, args.sample_each_row)
