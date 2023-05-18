from tensorflow import keras
import json
import numpy as np
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

import music21 as m21

class melodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def _sample_with_temperature(self, probabilities, temperature):
    
        predictions = np.log(probabilities)/temperature

        probabilities = np.exp(predictions)/np.sum(np.exp(predictions))

        choices = range(len(probabilities))

        index = np.random.choice(choices, p=probabilities)

        return index


    def generate_melody(self, seed, num_steps, max_sequence_length, temprature):

        seed = seed.split()

        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temprature)

            seed.append(output_int)

            output_symbol = [k for k,v in self._mappings.items() if v == output_int][0]
            
            if output_symbol == "/":
                break
                
            melody.append(output_symbol)
            
        return melody
    

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file

        :param melody (list of str):
        :param min_duration (float): Duration of each time step in quarter length
        :param file_name (str): Name of midi file
        :return:
        """

        # create a music21 stream
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)




if __name__=="__main__":

    mg = melodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"

    melody = mg.generate_melody(seed=seed, num_steps=500, max_sequence_length=SEQUENCE_LENGTH, temprature=0.3)
    print(melody)

    mg.save_melody(melody)
