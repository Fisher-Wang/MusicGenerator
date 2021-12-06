from pickle import NONE
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import matplotlib.pyplot as plt
import copy
import scipy

music_length=32

class CYQNN(nn.Module):
    def __init__(self,n_extra_layers=0):
        super().__init__()
        self.fc1 = nn.Linear(music_length, 2 * music_length, bias=False)
        self.sig1 = torch.nn.Sigmoid()
        self.fc2 = nn.Linear(2 * music_length, 2 * music_length, bias=False)
        self.sig2 = torch.nn.Sigmoid()
        self.fc3 = nn.Linear(2 * music_length, 2, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.fc1.weight.data.normal_(0, 1)
        self.fc2.weight.data.normal_(0, 1)
        self.fc3.weight.data.normal_(0, 1)
        # self.fc4.weight.data.normal_(0, 0.02)
        # self.fc5.weight.data.normal_(0, 0.02)
        # print(self.fc1.weight.data)
        # print(self.fc2.weight.data)
        # print(self.fc3.weight.data)

    def forward(self, x):
        out = self.sig1(self.fc1(x))
        out = self.sig2(self.fc2(out))
        # out = self.sig3(self.fc3(out))
        # out = self.sig4(self.fc4(out))
        out = self.fc3(out)
        out = self.softmax(out)
        # out = self.relu2(self.fc2(out))
        # out = self.relu3(self.fc3(out))
        # out = self.fc4(out)
        # out = self.fc2(out)
        # out = self.sig(out)
        return out

net = CYQNN()
net.load_state_dict(torch.load("net_final.pkl"))
music = np.zeros(music_length)
for i in range(music_length):
    music[i] = random.randint(1, 100)

music = torch.Tensor(music)
music = music / 100 - 0.6
music = music.reshape((1, 32))
print(music.size())

output = net(music)
print(output)