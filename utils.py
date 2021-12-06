import os

def setup_lib():
    workingdir = os.path.dirname(os.path.abspath(__file__))
    dllspath = os.path.join(workingdir, 'fluidsynth-2.2.4-win10-x64', 'bin')
    os.environ['PATH'] = dllspath + os.pathsep + os.environ['PATH']

    from ctypes.util import find_library
    lib = find_library("libfluidsynth")
    if lib is None:
        print("libfluidsynth not found!")