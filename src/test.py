from .seeds_generator import batch_seeds_forward, single_seeds_forward
import pickle
from jax import numpy as np

'''
point = np.array([[0.,0.]])
latent = np.array([1,2,3])
shape = point.shape
print(shape)
row = shape[0]
latent = latent = np.expand_dims(latent, 0).repeat(row, axis=0)
print(latent.shape)
in_array = np.concatenate([point, latent], 1)
'''
mode = 'infer'
file_read = open("/gpfs/share/home/1900011026/2D_deepSDF/data/model/{}ed_params.txt".format(mode), "rb")
params = pickle.load(file_read)
nn = params[1]
latent_code = params[0]
latent = latent_code[0]
point = np.array([0.,0.])
print(single_seeds_forward(point, latent, nn))
