# Music Generator

Final project of course Music and Math in PKU, using generic algorithm. 

### Setup
To check your configuration:
```shell
python test.py 
```

### Demo
`play_midi_demo.py` shows how to play the music in a midi file using module mido. 
`convert_midi_demo.py` shows how to convert a midi file into the notes, and needs further improvement. 

### Dataset
Among all the musicians, Bach, Clementi and Mozart are selected, and Bach's music is commanded for our task. 

## Net
### TO change dataset

see `prepare_data.py`

### TO train the net

`python net.py`

### TO generate music

`python ea.py`

## Use your own fitness function(new!)
To use your own fitness function, modify the `evaluate` method in class `MusicIndivisual`