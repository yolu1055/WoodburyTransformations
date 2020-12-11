from utils import load_state, count_parameters
from datasets import load_cifar10
import torch
from GlowLearner import Trainer
from models import WoodburyGlow
import argparse
import os


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='train Woodbury Glow')

    # input path
    parser.add_argument("-d", "--dataset_name", type=str, default="cifar10")
    parser.add_argument("--train_portion", type=str, default="train")
    parser.add_argument("--valid_portion", type=str, default="test")
    parser.add_argument("-r", "--dataset_root", type=str, default="./datasets")

    # output path
    parser.add_argument("--out_root", type=str, default="./results")

    # data parameters
    parser.add_argument("--n_bits", type=int, default=8)

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

    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--betas", type=tuple, default=(0.9,0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--decay", type=float, default=0.0000002)
    parser.add_argument("--scheduler_decay", type=float, default=0.95)


    # Device parameters
    parser.add_argument("--data_parallel", type=bool, default=True)

    # Trainer parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_grad_clip", type=float, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=100)
    parser.add_argument("--max_checkpoints", type=int, default=100)
    parser.add_argument("--nll_gap", type=int, default=100)
    parser.add_argument("--inference_gap", type=int, default=100)
    parser.add_argument("--valid_gap", type=int, default=100)
    parser.add_argument("--save_gap", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--sample_each_row", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)

    # model path
    parser.add_argument("--model_path", type=str, default="")

    # Infer parameters

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.cuda = torch.cuda.is_available()


    if not os.path.exists(args.dataset_root):
        os.makedirs(args.dataset_root)

    if not os.path.exists(args.out_root):
        os.makedirs(args.out_root)


    print("device: {}".format(args.device))
    print("num of gpus: {}".format(args.num_gpus))

    ranks = []
    if args.is_me:
        ranks = [args.d_c, args.d_s1, args.d_s2]
    else:
        ranks = [args.d_c, args.d_s1]

    args.ranks = ranks


    # model
    model = WoodburyGlow(args)

    if args.cuda:
        model = model.cuda()

    print ("number of parameters: {}".format(count_parameters(model)))



    # dataset
    training_set = load_cifar10(args.dataset_root, args.train_portion)
    valid_set = load_cifar10(args.dataset_root, args.valid_portion)

    # optimizer
    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr,betas=args.betas, weight_decay=args.decay)



    # scheduler

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, args.scheduler_decay)


    if args.model_path != "":
        state = load_state(args.model_path, args.cuda)
        optim.load_state_dict(state["optim"])
        model.load_state_dict(state["model"])
        args.steps = state["iteration"] + 1
        if scheduler is not None and state.get("scheduler", None) is not None:
            scheduler.load_state_dict(state["scheduler"])
        del state


    # begin to train
    trainer = Trainer(model, optim, scheduler, training_set, valid_set, args)
    trainer.train()
