#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from preprocess import SEQUENCE_LENGTH, generate_training_sentences
from tensorflow import keras

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE = 16
MODEL_PATH = "model.h5"

def train(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE, epochs = EPOCHS, batch_size = BATCH_SIZE, model_path = MODEL_PATH):
    input, target = generate_training_sentences(SEQUENCE_LENGTH)

    model = build_model(output_units, num_units, loss, learning_rate)

    model.fit(input, target, epochs=epochs, batch_size=batch_size)
    model.save(model_path)


def build_model(output_units, num_units, loss, learning_rate):

    input_layer = keras.layers.Input(shape=(None, output_units))
    hidden_layer = keras.layers.LSTM(num_units[0])(input_layer)
    hidden_layer = keras.layers.Dropout(0.2)(hidden_layer)
    output_layer = keras.layers.Dense(output_units, activation="softmax")(hidden_layer)

    model = keras.Model(input_layer, output_layer)

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])
    model.summary()

    return model

if __name__=="__main__":
    train()