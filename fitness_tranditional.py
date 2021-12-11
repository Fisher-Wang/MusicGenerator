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
        if c > 10:
            break
        notes.append(msg.note)
        
def fitness_tranditional(notes):
    from mingus.core import intervals
    score_dict = dict([
        ('major unison', 1),  # octave is the same
        ('perfect fourth', 1),
        ('perfect fifth', 1),
        ('minor third', 2),
        ('major third', 2),
        ('minor sixth', 2),
        ('major sixth', 2),
        ('major second', 3),
        ('augmented fourth', 3),
        ('diminished fifth', 3),
        ('diminished seventh', 3),
        ('major seventh', 3)
    ])
    
    tot_score = 0
    for i in range(len(notes) - 1):
        note1 = Note(notes[i])
        note2 = Note(notes[i+1])
        note1 = str(note1).replace("'", "").split('-')[0]
        note2 = str(note2).replace("'", "").split('-')[0]
        itv = intervals.determine(note1, note2)
        if itv not in score_dict:
            # print(itv)
            # raise('itv not found!')
            tot_score += 3
        else:
            tot_score += score_dict[itv]
        # print("%3s %3s %15s %1d" % (Note(notes[i]), Note(notes[i+1]), itv, score_dict[itv]))
    return tot_score

if __name__ == '__main__':
    score = fitness_tranditional(notes)
    print("score = %d" % (score))
    b = Bar(key='C', meter=(len(notes),8))
    for note in notes:
        b + Note(note)
    fluidsynth.play_Bar(b)