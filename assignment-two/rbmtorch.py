import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k, device):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible, device=device) * 0.1)
        self.a = nn.Parameter(torch.zeros(1, n_hidden, device=device))
        self.b = nn.Parameter(torch.zeros(1, n_visible, device=device))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def v_to_h(self, v):
        p_h_given_v = F.sigmoid(F.linear(v, self.W, self.a))
        sample_h_given_v = self.sample_from_p(p_h_given_v)
        return p_h_given_v, sample_h_given_v
    
    def h_to_v(self, h):
        p_v_given_h = F.sigmoid(F.linear(h, self.W.t(), self.b))
        sample_v_given_h = self.sample_from_p(p_v_given_h)
        return p_v_given_h, sample_v_given_h
    
    def forward(self, v):
        pre_sigmoid_ph, h = self.v_to_h(v)
        for step in range(self.k):
            pre_sigmoid_pv, v = self.h_to_v(h)
            pre_sigmoid_ph, h = self.v_to_h(v)
        return v
    
    def free_energy(self, v):
        vbias_term = v.mv(self.b.t())
        wx_b = F.linear(v, self.W, self.a)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    
    def loss(self, v):
        free_energy_v = self.free_energy(v)
        v_sample = self.forward(v)
        free_energy_v_sample = self.free_energy(v_sample)
        return free_energy_v - free_energy_v_sample
    
    def train(self, dataset, epochs, learning_rate):
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for data in dataset:
                data = Variable(torch.FloatTensor(data))
                loss = self.loss(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def reconstruct(self, data, sample=True):
        data = Variable(torch.FloatTensor(data))
        v = self.forward(data)
        if sample:
            return v > torch.rand(v.size())
        return v
    
    def get_weights(self):
        return self.W.data.numpy()
    
    def get_hidden_bias(self):
        return self.a.data.numpy()
    
    def get_visible_bias(self):
        return self.b.data.numpy()
    
    def get_hidden_nodes(self):
        return self.W.size()[0]
    
    def get_visible_nodes(self):
        return self.W.size()[1]
    
    def get_k(self):
        return self.k
    