from utils import setup_lib
setup_lib()

import os
import numpy as np
from mingus.containers import Note, NoteContainer, Bar
from mingus.midi import fluidsynth
from mido import MidiFile
soundfont_path = os.path.join("FluidR3_GM", "FluidR3_GM.sf2")
fluidsynth.init(soundfont_path)

import copy


music_length = 32
filename_list = os.listdir("./dataset/all_song")

flag = 1

notes = []
targets = []

cnt = 0

for name in filename_list:
    notes_sub=np.zeros((music_length))
    midi_path = 'dataset/all_song/'+name
    c = 0
    if cnt < 1000:
        for msg in MidiFile(midi_path):
            if not msg.is_meta:
                if msg.type != 'note_on':
                    continue
                if msg.velocity == 0:
                    continue
                notes_sub[c%music_length] = (int)(msg.note)
                c += 1
                if c % music_length == 0 and cnt < 1000:
                    notes_temp = copy.deepcopy(notes_sub)
                    notes.append(notes_temp)
                    if c <= 128 and flag == 1:
                        print(notes)
                    targets.append(1)
                    cnt += 1
    
    flag = 0

print(len(targets))

import random
for i in range(1000):
    notes_sub=np.zeros((music_length))
    c = 0
    for j in range(music_length):
        notes_sub[c%music_length] = (int)(random.randint(35, 95))  #53,79 is our range
        c += 1
    notes_temp = copy.deepcopy(notes_sub)
    notes.append(notes_temp)
    targets.append(0)


np.save( "./dataset/net/notes.npy", notes)
np.save( "./dataset/net/targets.npy", targets)

