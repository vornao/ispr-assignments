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

        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes
        self.weights = np.random.uniform(
            0, 0.1, (self.hidden_nodes, self.visible_nodes)
        )

        # initializing bias
        self.bias_hidden = np.zeros(self.hidden_nodes)
        self.bias_visible = np.zeros(self.visible_nodes)
        self.epoch_errors = np.array([])
        self.errors = np.array([])

    def train(self, dataset, epochs, learning_rate):
        """
        Train the RBM, using CD-1 algorithm.
        :param dataset: training dataset
        :param epochs: number of epochs
        :param learning_rate: learning rate
        :return: None
        """

        for epoch in range(epochs):
            self.epoch_errors = []
            for data in dataset:
                # computing the wake part of the gradiente (See equation 7 in Hinton's paper)
                hidden_prob_given_v = self.sigmoid(
                    data.dot(self.weights.T) + self.bias_hidden
                )
                # actual expected value of the hidden units given my training example
                # nones are used to make the matrix multiplication work
                wake = np.dot(hidden_prob_given_v[:, None], data[:, None].T)

                # now we are giving a state to our hidden units based on the training sample.
                # sampling from distributions, see Hinton's paper, sec 3.4.
                # When the hidden units are being driven by data, always use stochastic binary states.
                # When they are being drivenby reconstructions, always use probabilities without sampling.
                hidden_states = hidden_prob_given_v > np.random.rand(self.hidden_nodes)

                # given these new hidden units states, try to reconstruct data probabilities,
                # then sample visible states.
                recon_probs = self.sigmoid(
                    hidden_states.dot(self.weights) + self.bias_visible
                )

                # not using states because
                # "For the last update of the hidden units, it is silly to use stochastic binary states
                # because nothing depends on which state is chosen." [Hinton's paper, section 3.1, 3.4]
                recon_states = recon_probs

                # getting probabilities of the hidden units given the reconstruction
                hidden_prob_given_rec = self.sigmoid(
                    recon_states.dot(self.weights.T) + self.bias_hidden
                )

                dream = np.dot(
                    hidden_prob_given_rec[:, None],
                    recon_states[:, None].T,
                )

                delta_w = wake - dream
                delta_bias_hidden = hidden_prob_given_v - hidden_prob_given_rec
                delta_bias_visible = data - recon_states

                error = np.abs(data - recon_states)

                self.weights += learning_rate * delta_w
                self.bias_hidden += learning_rate * delta_bias_hidden
                self.bias_visible += learning_rate * delta_bias_visible
                self.epoch_errors.append(error)

            self.errors = np.append(self.errors, np.mean(self.epoch_errors))

    def reconstruct(self, data, sample=True) -> np.ndarray:
        """
        Reconstruct the input data
        :param data: input data
        :param sample: setting flag to true will return sampled states instead of probabilities.
        :return: reconstructed data
        """
        hidden_prob_given_v = self.sigmoid(data.dot(self.weights.T) + self.bias_hidden)
        recon_probs = self.sigmoid(
            hidden_prob_given_v.dot(self.weights) + self.bias_visible
        )
        if sample:
            return recon_probs > np.random.rand(self.visible_nodes)

        return recon_probs

    def sigmoid(self, x):
        # sigmoid activation function
        return 1.0 / (1.0 + np.exp(-x))
