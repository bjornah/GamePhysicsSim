import tqdm
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

import GamePhysicsSim.Utils as Utils

def TrainModel():

    class_names = ['Thrust','Torque']
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6,)),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
