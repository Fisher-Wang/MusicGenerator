import os

def code2sound(code):
    note_disk =                      ["E-3", "F-3", "F#-3", "G-3", "G#-3", "A-3", "A#-3", "B-3",
    "C-4", "C#-4", "D-4", "D#-4", "E-4", "F-4", "F#-4", "G-4", "G#-4", "A-4", "A#-4", "B-4",
    "C-5", "C#-5", "D-5", "D#-5", "E-5", "F-5", "F#-5", "G-5"]

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

    b = Bar(key='C', meter=(32,8))
    for i in code:
        b + note_disk[i]
    b.place_notes('A-4', 4)
    fluidsynth.play_Bar(b)