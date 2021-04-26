#!/usr/bin/env python
# basic imports

import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
from importlib import reload
import pickle
import pandas as pd
import seaborn as sns
import random

import pygame

import GamePhysicsSim as GPS
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf,ROOT_DIR
import GamePhysicsSim.NN as NN
import GamePhysicsSim.Visualise as Visualise
import GamePhysicsSim.Utils as Utils

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler
# import sklearn
# from sklearn import preprocessing
from tqdm.keras import TqdmCallback






def get_data_scaler(x_train, data_scaler_path, new_scaler=False, save_scaler=False, scaler_name='scaler_x.pkl'):
    if new_scaler:
        data_scaler_x = preprocessing.StandardScaler().fit(x_train.values)
        x_train_scaled = data_scaler_x.transform(x_train)
        if save_scaler:
            if not os.path.exists(data_scaler_path):
                os.makedirs(data_scaler_path)

            scaler_file = os.path.join(data_scaler_path,scaler_name)
            pickle.dump(data_scaler_x, open(scaler_file, 'wb'))
    else:
        scaler_file = os.path.join(data_scaler_path,scaler_name)
        data_scaler_x = pickle.load(open(scaler_file, 'rb'))
        x_train_scaled = data_scaler_x.transform(x_train)

    return data_scaler_x


if __name__ == "__main__":
    scaler_name='scaler_x.pkl'
    new_scaler = False
    save_scaler = True
    mod_number = 4

    confdir = conf['conf_version']

    training_data_dir = os.path.join(ROOT_DIR,f'Models/{confdir}/TrainingData')
    data_scaler_path = os.path.join(ROOT_DIR,f'Models/{confdir}/DataScalers')
    model_save_path = os.path.join(ROOT_DIR,f'Models/{confdir}')

    x_train = pd.read_csv(f'{training_data_dir}/train_data_errors_v1.csv',names=['vx','vy','w','delta_x','delta_y','delta_theta'])
    y_train = pd.read_csv(f'{training_data_dir}/train_data_response_v1.csv',names=['Thrust','Torque'])

    data_scaler_x = get_data_scaler(x_train, data_scaler_path, new_scaler=new_scaler, save_scaler=save_scaler, scaler_name=scaler_name)
    x_train_scaled = data_scaler_x.transform(x_train)

    # Define NN

    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(6,)),
    keras.layers.Dense(units=32,kernel_regularizer=keras.regularizers.l2(0.1), activation='relu'),
    keras.layers.Dense(units=32,kernel_regularizer=keras.regularizers.l2(0.1), activation='relu'),
    keras.layers.Dense(units=16,kernel_regularizer=keras.regularizers.l2(0.1), activation='relu'),
    keras.layers.Dense(units=2)
    ])

    BATCH_SIZE = 500
    STEPS_PER_EPOCH = len(x_train)/BATCH_SIZE

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
              loss='mean_absolute_error',
              metrics=['accuracy'])

    # perform initial fit

    history = model.fit(x_train_scaled, y_train.values, validation_split=0.2, epochs=50,
                      verbose=0, callbacks=[TqdmCallback(verbose=0)])
    fig = NN.plot_loss(history)
    fig.savefig(os.path.join(model_save_path,f'epochs_0_50_mod{mod_number}.pdf'),bbox_inches='tight')

    # run further
    history = model.fit(x_train_scaled, y_train.values, validation_split=0.2, epochs=500,
                      verbose=0, callbacks=[TqdmCallback(verbose=0)])
    fig = NN.plot_loss(history)
    fig.savefig(os.path.join(model_save_path,f'epochs_50_550_mod{mod_number}.pdf'),bbox_inches='tight')

    print(model.summary())

    mod_file = os.path.join(model_save_path,f'Model{mod_number}_Gen0')
    model.save(mod_file)
