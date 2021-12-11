# from net import CYQNN
import torch
from torch import nn

music_length = 32

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

    def forward(self, x):
        out = self.sig1(self.fc1(x))
        out = self.sig2(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out

def fitness_ANN(music):
    net = CYQNN()
    net.load_state_dict(torch.load("net_final.pkl"))
    music = torch.Tensor(music)
    music = (music + 52) / 100 - 0.6
    music = music.reshape((1, 32))
    output = net(music)
    return output[0][1]