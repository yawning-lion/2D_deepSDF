import numpy as onp
import jax.numpy as np
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from jax import grad, jit, vmap, value_and_grad
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
from .nn_train import loss, batch_forward, single_forward, forward

def process_node_sign(node_sign):
    line = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]])
    line_node = node_sign[line]
    flag = np.multiply(line_node[:, 0], line_node[:, 1])
    seg_line = line[np.where(flag > 0)]
    return seg_line

def get_node_sign(square, params, shape):
    shape_array = np.repeat(shape, 4).reshape(4, 1)
    node = np.concatenate([square, shape_array], 1)
    node_sign = np.sign(forward(params, node)).reshape(-1)
    return node_sign
    
def get_seg_point(P0, step_size, params, shape):
    P1 = P0 + np.array([1., 0.]) * step_size
    P2 = P0 + np.array([0., 1.]) * step_size
    P3 = P0 + np.array([1., 1.]) * step_size
    square = np.stack((P0, P1, P2, P3), axis = 0)
    node_sign = get_node_sign(square, params, shape)
    logic_seg_line = process_node_sign(node_sign)
    seg_line = square[logic_seg_line]
    seg_point = (seg_line[:, 0] + seg_line[:, 1])/2
    return seg_point[:-1]#prevent repeating

def get_squares(floor, ceil, step_size):
    x = np.arange(floor, ceil, step_size)
    y = x
    X, Y = np.meshgrid(x, y)
    point = np.stack([X, Y], axis=2)
    shape = point.shape
    squares = point.reshape(shape[0]*shape[1], shape[2])
    return squares
	
batch_get_seg_point = vmap(get_seg_point, in_axes = (0, None, None, None), out_axes = 0)

def marching_squares(floor, ceil, step_size, params, shape):
    squares = get_squares(floor, ceil, step_size)
    seg_points = batch_get_seg_point(squares, step_size, params, shape)
    plt.figure(figsize=(5,5))
    plt.scatter(seg_points[:, 0], seg_points[:, 1], s = 1, c = 'r', marker = 'o')
    plt.savefig('/gpfs/share/home/1900011026/2D_deepSDF/data/img/marching_points_shape{}{}'.format(shape // 10, shape%10))
    plt.close()
    return seg_points
    
if __name__ == '__main__':
    mode = 'train'
    file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
    params = pickle.load(file_read)
    marching_squares(-3, 3, 0.05, params, 1)