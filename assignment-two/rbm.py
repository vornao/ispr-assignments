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
from sklearn.datasets import load_digits


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
        self.weights = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.visible_nodes))

        # initializing bias
        self.bias_hidden = np.zeros(self.hidden_nodes)
        self.bias_visible = np.zeros(self.visible_nodes)
        self.epoch_errors = np.array([])
        self.errors = np.array([])

    def train(self, dataset, epochs, learning_rate):
        """
        Train the RBM
        TODO: implement the mini-batch version of the algorithm.
        :param dataset: training dataset
        :param epochs: number of epochs
        :param learning_rate: learning rate
        :return: None
        """

        for epoch in range(epochs):
            self.epoch_errors = []
            for data in dataset:
                # computing the wake part of the gradiente (See equation 7 in Hinton's paper)
                hidden_prob_given_v = self.sigmoid(data.dot(self.weights.T) + self.bias_hidden)
                # actual expected value of the hidden units given my training example
                # nones are used to make the matrix multiplication work
                wake = np.dot(hidden_prob_given_v[:, None], data[:, None].T)

                # now we are giving a state to our hidden units based on the training sample.
                hidden_stocastic_state = hidden_prob_given_v > np.random.rand(self.hidden_nodes)

                # given these new hidden units states, try to reconstruct data probabilities, and then sample visible states.
                reconstruction_probs = self.sigmoid(hidden_stocastic_state.dot(self.weights) + self.bias_visible)
                # using states because [Hinton's paper, section 3.4]
                reconstruction_stocastic_state = reconstruction_probs > np.random.rand(self.visible_nodes)

                # getting probabilities of the hidden units given the reconstruction
                hidden_prob_given_rec = self.sigmoid(reconstruction_stocastic_state.dot(self.weights.T) + self.bias_hidden)
                dream = np.dot(hidden_prob_given_rec[:, None], reconstruction_stocastic_state[:, None].T)

                delta_w = (wake - dream)
                delta_bias_hidden = hidden_prob_given_v - hidden_prob_given_rec
                delta_bias_visible = data - reconstruction_stocastic_state

                error = np.sum((data - reconstruction_stocastic_state) ** 2)
                
                self.weights += learning_rate * delta_w
                self.bias_hidden += learning_rate * delta_bias_hidden
                self.bias_visible += learning_rate * delta_bias_visible
                self.epoch_errors.append(error)

            self.errors = np.append(self.errors, np.mean(self.epoch_errors))
        
        

    def reconstruct(self, data) -> np.ndarray:
        """
        Reconstruct the input data
        :param data: input data
        :return: reconstructed data
        """
        hidden_prob_given_v = self.sigmoid(data.dot(self.weights.T) + self.bias_hidden)
        reconstruction_probs = self.sigmoid(hidden_prob_given_v.dot(self.weights) + self.bias_visible)
        return reconstruction_probs
    

    def sigmoid(self, x):
        # sigmoid activation function
        return 1.0 / (1.0 + np.exp(-x))
        