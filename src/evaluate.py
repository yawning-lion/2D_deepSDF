import numpy as onp
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
from functools import partial
from jax.scipy.special import logsumexp
from jax.experimental import optimizers, stax
from jax.nn import selu,relu
from jax.experimental.stax import Dense, Relu, Sigmoid, Softplus, Selu, Tanh, Identity, FanOut, FanInConcat
from jax.numpy import tanh
from torch.utils.data import Dataset, DataLoader
from jax import random
import time
import pickle
import argparse
import math
from .utils import SDF_dataloader, plot_learning_curve
from .argument import args
from .nn_train import batch_forward
from numpy import sin, cos, pi
from .seeds_generator import run_refine_seeds_loop

def read_data(mode):
    boudary = np.load("/gpfs/share/home/1900011026/2D_deepSDF/data/data_set/{}_boundary_point.npy".format(mode))
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    return boundary, params

def evaluate_and_report():
    mode = 'train'
    boudary, params = read_data(mode)
    evaluation = onp.arange(args.num_shape_train + args.num_shape_infer)
    for i in range(args.num_shape_train):
        temp = run_refine_seeds_loop(boundary[i], i, params)
        evaluation[i] = np.linalg.norm(temp - boudary[i])    
    mode = 'infer'
    boudary, params = read_data(mode)
    for i in range(args.num_shape_infer):
        temp = run_refine_seeds_loop(boundary[i], i, params)
        evaluation[i + args.num_shape_train] = np.linalg.norm(temp - boudary[i])
    
    onp.save("/gpfs/share/home/1900011026/2D_deepSDF/data/evaluation.npy", evalution)
    print("done")

opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

if __name__ == '__main__':
    evaluate_and_report()