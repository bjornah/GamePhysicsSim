import tqdm
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import GamePhysicsSim.Utils as Utils

# def BuildAndCompileModel(normedInputLayer = None):
#     # label_names = ['Thrust','Torque']
#     if normedInputLayer:
#         model = keras.Sequential([
#         normedInputLayer,
#         keras.layers.Dense(units=32, activation='relu'),
#         keras.layers.Dense(units=32, activation='relu'),
#         keras.layers.Dense(units=2)
#         ])
#     else:
#         model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(6,)),
#         keras.layers.Dense(units=32, activation='relu'),
#         keras.layers.Dense(units=32, activation='relu'),
#         keras.layers.Dense(units=2)
#         ])
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#                   loss='mean_absolute_error',
#                   metrics=['accuracy'])
#     return model

def CreateEmptyModelShell(input_shape):
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(units=64,kernel_regularizer=keras.regularizers.l2(0.1), activation='relu'),
    keras.layers.Dense(units=64,kernel_regularizer=keras.regularizers.l2(0.1), activation='relu'),
    keras.layers.Dense(units=2)
    ])
    return model

def LoadUncompiledModel(modelpath):
    return keras.models.load_model(modelpath,compile=False)

def MateModels(parentModels,input_shape):
    '''
    returns a model where the weights are the average of the weights from parentModels. It is assumed that all models share the same structure.
    '''
    parentWeights = [mod.get_weights() for mod in parentModels]
    childWeights = []
    for w_i_list in zip(*parentWeights):
        w_i_child = (np.sum(w_i_list,axis=0))/float(len(w_i_list))
        childWeights.append(w_i_child)
    middleChild = CreateEmptyModelShell(input_shape)
    middleChild.set_weights(childWeights)
    return middleChild

def PerturbModelWeights(model, epsilon = 0.1):
    '''
    returns model with added stochastic perturbations of scale epsilon to all weights
    '''
    weights = model.get_weights()
    perturbedWeights = []
    for wi in weights:
        sigma = np.sqrt(np.var(wi)*epsilon)
        noise = np.random.normal(0, sigma, wi.shape)
        perturbedWeights.append(wi + noise)
    model.set_weights(perturbedWeights)
    return model

def GetNextGeneration(parentModels, NrChildren, input_shape, Mating=False, epsilon = 0.1):
    children = []
    if Mating:
        middleChild = MateModels(parentModels)
        for i in range(NrChildren):
            child = keras.models.clone_model(middleChild)
            child.set_weights(middleChild.get_weights())
            child = PerturbModelWeights(child, epsilon = epsilon)
            children.append(child)
    else:
        for parent in parentModels:
            child = CreateEmptyModelShell(input_shape)
            child.set_weights(parent.get_weights())
            children.append(child)
        while len(children)<NrChildren:
            for parent in parentModels:
                child = CreateEmptyModelShell(input_shape)
                child.set_weights(parent.get_weights())
                child = PerturbModelWeights(child, epsilon = epsilon)
                children.append(child)
    return children

def plot_loss(history):
    fig,ax = plt.subplots()
    ax.plot(history.epoch,history.history['loss'], label='loss')
    ax.plot(history.epoch,history.history['val_loss'], label='val_loss')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error [MPG]')
    ax.legend()
    return fig
