import os
from music21 import environment
import music21 as m21
import json
import numpy as np
from tensorflow import keras

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
DATASET_FILE = "file_dataset"
MAPPING_PATH = "mapping.json"

ACCEPTABLE_DURATION =[
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    3,
    4
]

env = environment.Environment(forcePlatform="windows")
env["musicxmlPath"] = "C:\\Program Files\MuseScore 4\\bin\\MuseScore4.exe"

def preprocess(dataset_path):
    #Load dataset
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)

   
    for i, song in enumerate(songs):

        #Filter durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATION):
            continue

        #Transpose in C/Am
        song = transpose(song)

        #Encoding melodies
        song = encode_song(song, 0.25)

        #Save dataset in a text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(song)
        


def load_songs_in_kern(dataset_path):
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path,file))
                songs.append(song)

    return songs

def has_acceptable_durations(song, ACCEPTABLE_DURATION):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in ACCEPTABLE_DURATION:
            return False
        
    return True

def transpose(song):
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    interval = None

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))

    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    transposed_song = song.transpose(interval)

    return transposed_song

def encode_song(song, time_step):
    encoded_song = []

    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi

        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        steps = int(event.duration.quarterLength /time_step)

        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song

def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()

    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_lenght):
    song_delimiter = "/ " * sequence_lenght

    songs = ""

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)

            song = load(file_path)
            
            songs = songs + song + song_delimiter

        songs = songs[:-1]
    
        with open(file_dataset_path, "w") as fp:
            fp.write(songs)
    
    return songs


def create_mapping(songs, mapping_path):
    mapping = {}
    songs = songs.split()
    vocabulary = list(set(songs))

    for i, symbol in enumerate(vocabulary):
        mapping[symbol] = i

    with open(mapping_path, "w") as fp:
        json.dump(mapping, fp, indent=4)

def convert_song_to_int(songs):
    int_songs = []

    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)

    songs = songs.split()

    for symbol in songs:
        int_songs.append(mapping[symbol])

    return int_songs

def generate_training_sentences(sequence_length):

    songs = load(DATASET_FILE)
    int_songs = convert_song_to_int(songs)


    input = []
    target = []
    num_sequences = len(int_songs) - sequence_length

    for i in range(num_sequences):
        input.append(int_songs[i:i+sequence_length])
        target.append(i+sequence_length)


if __name__=="__main__":
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, DATASET_FILE, 64)
    create_mapping(songs, MAPPING_PATH)
    convert_song_to_int(songs)