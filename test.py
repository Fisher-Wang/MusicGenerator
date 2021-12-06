from utils import setup_lib
setup_lib()

import os
from mingus.containers import Note, NoteContainer, Bar
from mingus.midi import fluidsynth

soundfont_path = os.path.join("FluidR3_GM", "FluidR3_GM.sf2")
fluidsynth.init(soundfont_path)

b = Bar(key='C', meter=(10,8))
b + "E-5"
b + "D#-5"
b + "E-5"
b + "D#-5"
b + "E-5"
b + "B-4"
b + "D-5"
b + "C-5"
b.place_notes('A-4', 4)
fluidsynth.play_Bar(b)