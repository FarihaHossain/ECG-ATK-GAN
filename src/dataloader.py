import numpy as np
from numpy import load
import tensorflow as tf

# generate points in latent space as input for the generator
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space   
    x_input = gaussian_filter1d(np.random.randint(high=1.0, low=0.0,size=(latent_dim*n_samples)),4)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input
    
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, real_sample, labels_input, labels_atck, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([real_sample,z_input, labels_input])
    y = zeros((n_samples, 1))
    #pyplot.scatter(images[:, 0], images[:, 1])
    return [images, labels_input, labels_atck], y    
    
def generate_real_random(dataset, n_samples):
    # split into images and labels
    images, labels, labels_atck = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels, labels_atck = images[ix], labels[ix], labels_atck[ix]
    y = ones((n_samples, 1))
    return [X, labels, labels_atck], y    
    
def generate_real_samples(dataset, batch_id, n_samples):
    images, labels, labels_atck = dataset
    start = batch_id*n_samples
    end = start+n_samples
    X, labels, labels_atck = images[start:end], labels[start:end], labels_atck[start:end]
    y = ones((n_samples, 1))
    return [X, labels, labels_atck], y    