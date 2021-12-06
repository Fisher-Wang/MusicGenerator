from utils import setup_lib
setup_lib()

import os
from mingus.containers import Note, NoteContainer, Bar
from mingus.midi import fluidsynth
from mido import MidiFile

soundfont_path = os.path.join("FluidR3_GM", "FluidR3_GM.sf2")
fluidsynth.init(soundfont_path)

notes = []
midi_path = 'dataset/piano-midi/bach/bach_846.mid'
c = 0
for msg in MidiFile(midi_path):
    if not msg.is_meta:
        if msg.type != 'note_on':
            continue
        if msg.velocity == 0:
            continue
        c += 1
        if c > 100:
            break
        notes.append(msg.note)

print(notes)
        
b = Bar(key='C', meter=(len(notes),8))
cnt = 0
for note in notes:
    b.place_notes(Note(note), 8)
    cnt += 1


fluidsynth.play_Bar(b)