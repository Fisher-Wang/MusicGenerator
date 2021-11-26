import os

workingdir = os.path.abspath('.')
dllspath = os.path.join(workingdir, 'fluidsynth-2.2.4-win10-x64', 'bin')
os.environ['PATH'] = dllspath + os.pathsep + os.environ['PATH']

from ctypes.util import find_library
lib = find_library("libfluidsynth")
if lib is None:
    print("libfluidsynth not found!")

from mingus.containers import Note, NoteContainer, Bar
from mingus.midi import fluidsynth

soundfont_path = os.path.join(workingdir, "FluidR3_GM", "FluidR3_GM.sf2")
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