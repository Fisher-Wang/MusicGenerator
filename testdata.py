from utils import setup_lib
setup_lib()

import os
from mingus.containers import Note, NoteContainer, Bar
from mingus.midi import fluidsynth
from mido import MidiFile
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

soundfont_path = os.path.join("FluidR3_GM", "FluidR3_GM.sf2")
fluidsynth.init(soundfont_path)

import numpy as np
notes = np.load("./dataset/net/notes.npy")
targets = np.load("./dataset/net/targets.npy")

i = 0
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

while 1:
    i = (int)(input())
    b = Bar(key='C', meter=(music_length,8))
    cnt = 0
    print("is it a music? :", targets[i])
    print("music sequnence: ", notes[i])
    music = copy.deepcopy(notes[i])
    music = torch.Tensor(music)
    music = music / 100 - 0.6
    music = music.reshape((1, 32))
    net_output = net(music)
    print("net output: ", net_output)

    for note in range(music_length):
        b.place_notes(Note((int)(notes[i][note])), 8)
        cnt += 1
    fluidsynth.play_Bar(b)


fluidsynth.play_Bar(b)