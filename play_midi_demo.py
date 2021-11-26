import mido
import time
port = mido.open_output('Microsoft GS Wavetable Synth 0')

notes = []
for msg in mido.MidiFile('dataset/piano-midi/mozart/mz_311_1.mid'):
    time.sleep(msg.time)
    if not msg.is_meta:
        if msg.type != 'note_on':
            continue
        if msg.velocity == 0 and msg.time == 0:
            continue
        
        port.send(msg)
        print(msg)
        notes.append(msg.note)