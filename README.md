# Woodbury Transformations

This code is a Python implementation of the Woodbury Transformations introduced in the paper 

"[Woodbury Transformations for Deep Generative Flows](https://arxiv.org/abs/2002.12229)". You Lu and Bert Huang. NeurIPS 2020.

Note: This code is used for the experiments on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). Some parts of the code are adapted from [chaiyujin](https://github.com/chaiyujin/glow-pytorch), and [openai](https://github.com/openai/glow). 

## Requirements:

This code was tested using the the following libraries.

- Python 3.6.7
- Pytorch 1.2.0

## Running

### Training
- Download the repo to your computer.
- In the terminal, run *./train_wglow.sh*.

### Inference
- Specify the model path in *./test_wglow.sh*.
- In the terminal, run *./test_wglow.sh*.

## Contact
Feel free to send me an email, if you have any questions or comments.
