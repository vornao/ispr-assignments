"""
Assignment 3

Implement from scratch an RBM and apply it to DSET3. 
The RBM should be implemented fully by you (both CD-1 training and inference steps) 
but you are free to use library functions for the rest (e.g. image loading and management, etc.).

1. Train an RBM with a number of hidden neurons selected by you (single layer) on the MNIST data (use the training set split provided by the website).
2. Use the trained RBM to encode a selection of test images (e.g. using one per digit type) using the corresponding activation of the hidden neurons.
3. Reconstruct the original test images from their hidden encoding and confront the reconstructions with the original 
   image (use a suitable quantitative metric to assess the reconstraction quality and also choose few examples to confront visually).
"""

import numpy as np


class RestrictedBoltzmannMachine:
    """
    Restricted Boltzmann Machine
    """
    def __init__(self, visible_nodes, hidden_nodes):
        """
        Initialize the parameters of the RBM
        :param visible_nodes: number of visible nodes
        :param hidden_nodes: number of hidden nodes
        """


        self. visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes

        # random initializing weights
        self.weights = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.visible_nodes))

        # initializing bias
        self.bias_hidden = np.zeros(self.hidden_nodes)
        self.bias_visible = np.zeros(self.visible_nodes)

    def train(self, data, epochs, learning_rate):
        
        for epoch in range(epochs):
            # computing the wake part of the gradiente
            ph_given_v = self._activation(np.dot(data, self.weights.T) + self.bias_hidden)
            # actual expected value of the hidden units given my training example
            wake = np.dot(ph_given_v.T, data)

            # computing the dream part of the gradiente
            pv_given_h = self._activation(np.dot(ph_given_v, self.weights) + self.bias_visible)
            # actual expected value of the visible units given my training example
            dream = np.dot(pv_given_h.T, ph_given_v)
            

            # now computing the dream part by trying to reconstruct the data

            # sample hidden units


           



    def gibbs_sampling(self, k, x):
        
        # sample hidden units
        hidden = self.sample_h(x)
        # sample visible units
        visible = self.sample_v(hidden)
        # sample hidden units
        hidden = self.sample_h(visible)

        return visible, hidden

    
    def _activation(self, x):
        # sigmoid activation function
        return 1.0 / (1.0 + np.exp(-x))





