#!/usr/bin/env python
# basic imports
import numpy as np
import sys
import pickle
import math
import os

import cProfile, pstats, io
from pstats import SortKey

# visualisation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

# debugging
from IPython.core.debugger import set_trace

# from GamePhysicsSim
import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf,ROOT_DIR
#import GamePhysicsSim.Visualise as Visualise
import GamePhysicsSim.NN as NN
# from GamePhysicsSim.Utils import profile

# keras
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

# third party keras related software
from KerasGA import GeneticAlgorithm


# sklearn
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import sklearn
# from sklearn import preprocessing

########################################################################################
# the code using KerasGA GeneticAlgorithm uses example code from https://github.com/yahiakr/KerasGA
########################################################################################

def CheckPointCheck(pod,CheckPointSize):
    return np.linalg.norm(pod.dest - pod.pos) < CheckPointSize

# def calculateFitness(fitness,tMax):
#     fitnessScore_model_list = []
#     for ID,scoreData in fitness.items():
#         fitnessScore = len(scoreData)-1 - Utils.sigmoid(scoreData[-1]/tMax)
#         for pod in podList:
#             if pod.ID == ID:
#                 fitnessScore_model_list.append((pod.model.get_weights(),fitnessScore))
#     return
    # fitnessScore = [tuple(d) for d in {'a':1,'b':2}.items()]
    # return fitnessScore

def reset_lists():
    IDList[:] = []
    podList[:] = []

def initiate_pods(modelList):
    # Initiate Pod
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)]) # random starting position
    theta0 = 3/2 * np.pi
    # podList = []
    for model in modelList:
        pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                            m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'], model=model,
                            ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'], wMax = conf['wMax'])

        podList.append(pod)
        IDList.append(pod.ID)
        pod.model = model
    # initiate dict for fitnessScores
    # for pod in podList:
        # fitness[pod.ID] = [0]

    # return podList

def generateCheckpoints():
    CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
    checkpointList[:] = [np.array(c) for c in CheckGen]

def setCheckpoints():
    # initiate the first checkpoint for each pod:
    for pod in podList:
        pod.setDest(checkpointList[0])

# @profile(output_file='update',sort_by='cumulative', lines_to_print=20, strip_dirs=True)
def update(t):
    for pod in podList:
        # set_trace()
        if (t%delta_t) == 0:
            x = scaler.transform(np.array([pod.getPodSteeringStats(delta_t=delta_t)]))
            pod.NN_getSteering(x)

        pod.AddThrust(Thrust = pod.Thrust)
        pod.AddTorque(Torque = pod.Torque)

        pod.Rotate(AngularDrag = AngularDrag, AngularFriction = AngularFriction, dt = dt)
        pod.Move(Drag = Drag, Friction = Friction, dt = dt)

        if CheckPointCheck(pod,CheckPointSize):
            # doneCheckPoints = 1+(math.ceil(pod.fitnessScore) % M )
            pod.doneCheckpoints += 1
            pod.setDest(checkpointList[(pod.doneCheckpoints % M)])
            # pod.setDest(checkpointList[doneCheckPoints])
            # pod.doneCheckpoints += 1 #= doneCheckPoints - Utils.sigmoid(t/tMax)
        pod.podpatch.center = tuple(pod.pos)
        FT = np.array([pod.Thrust*np.cos(pod.theta),pod.Thrust*np.sin(pod.theta)])*dt**2
        pod.accLine.set_data([pod.pos[0],pod.pos[0]+FT[0]],[pod.pos[1],pod.pos[1]+FT[1]])
        pod.velLine.set_data([pod.pos[0],pod.pos[0]+pod.v[0]],[pod.pos[1],pod.pos[1]+pod.v[1]])
        pod.destLine.set_data([pod.pos[0],pod.dest[0]],[pod.pos[1],pod.dest[1]])
        # ax.plot([pod.pos[0],pod.dest[0]],[pod.pos[1],pod.dest[1]]) # goal vec
        # FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
        # ax.plot([pod.pos[0],pod.pos[0]+FT[0]],[pod.pos[1],pod.pos[0]+FT[0]]) # acc vec
        # ax.plot([pod.pos[0],pod.pos[0]+pod.v[0]],[pod.pos[1],pod.pos[0]+pod.v[0]]) # v vec
    t+=dt

# @profile(output_file='plot_bkg',sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def plot_bkg(cpFigs):
    fig,ax = plt.subplots()
    ax.set_xlim([-int(X/5),int(X*1.5)])
    ax.set_ylim([-int(Y/5),int(Y*1.5)])
    for cpFig in cpFigs:
        ax.add_artist(cpFig)
    return fig,ax

# setup plot
def setupPlot():
    cpFigs = [plt.Circle(cp, CheckPointSize, color='r') for cp in checkpointList]
    fig,ax = plot_bkg(cpFigs)
    return fig,ax

# @profile(output_file='AddPodPatches',sort_by='cumulative', lines_to_print=10, strip_dirs=True)
def addPodPatches(ax):
    for j,pod in enumerate(podList):
        podPatch = plt.Circle((pod.pos), PODSIZE[0], color='b',alpha=0.5)
        pod.podpatch = podPatch
        ax.add_patch(podPatch)
        destLine, = ax.plot([pod.pos[0],pod.pos[0]],[pod.pos[1],pod.pos[0]],color='red')
        pod.destLine = destLine
        accLine, = ax.plot([pod.pos[0],pod.pos[0]],[pod.pos[1],pod.pos[0]],color='blue')
        pod.accLine = accLine
        velLine, = ax.plot([pod.pos[0],pod.pos[0]],[pod.pos[1],pod.pos[0]],color='purple')
        pod.velLine = velLine
    return ax
    # ax.add_patch(destLine)
    # destpatch = plt.Arrow(pod.pos[0],pod.pos[1], (pod.dest - pod.pos)[0], (pod.dest - pod.pos)[1], width=3.0,color='orange')
    # destpatch = plt.Arrow((pod.pos), (pod.dest-pod.pos), width=1.0)
    # pod.destpatch = destpatch
    # ax.add_patch(destpatch)

        # ax.plot([pod.pos[0],pod.dest[0]],[pod.pos[1],pod.dest[1]]) # goal vec
        # FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
        # ax.plot([pod.pos[0],pod.pos[0]+FT[0]],[pod.pos[1],pod.pos[0]+FT[0]]) # acc vec
        # ax.plot([pod.pos[0],pod.pos[0]+pod.v[0]],[pod.pos[1],pod.pos[0]+pod.v[0]]) # v vec

# @profile(output_file='runSim',sort_by='tottime', lines_to_print=100, strip_dirs=True)
def runSim(gen,modNr,savepath):
    fig,ax = setupPlot()
    ax = addPodPatches(ax)
    t = np.arange(0,tMax,dt)
    # anim = FuncAnimation(fig,animate,frames = t,fargs=(podList,))
    # writergif = animation.PillowWriter(fps=30)
    writervideo = animation.FFMpegWriter(fps=30)
    anim = FuncAnimation(fig,update,frames = t)
    # set_trace()
    anim.save(savepath+'/Model'+str(modNr)+'_Gen'+str(gen)+'.mp4', writer=writervideo)


def saveChildren(children,savepath,gen,modNr):
    for j,child in enumerate(children):
        child.save(savepath+'/Model'+str(modNr)+'_Gen'+str(gen)+'_'+str(j))

if __name__ == "__main__":

    ############### set some constants and import settings from the config file

    dt = conf['dt']
    delta_t = dt*3
    tMax = 15

    AngularDrag = conf['AngularDrag']
    AngularFriction = conf['AngularFriction']
    Drag = conf['AirDrag']
    Friction = conf['Friction']
    N = 10 # number of pods
    M = 10 # number of checkpoints
    # Sizes
    X = 800
    Y = 800
    PODSIZE = (30,30)
    CheckPointSize = 20

    IDList = []
    checkpointList = []
    podList = []

    ############################

    profile=False

    savepath = os.path.join(ROOT_DIR,f'Models/{conf["conf_version"]}/sim_results')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    data_scaler_path = os.path.join(ROOT_DIR,f'Models/{conf["conf_version"]}/DataScalers')
    scaler_file = os.path.join(data_scaler_path,'scaler_x.pkl')
    scaler = pickle.load(open(scaler_file, 'rb'))

    modNr = 1
    gen = 0
    model_save_path = os.path.join(ROOT_DIR,f'Models/{conf["conf_version"]}')
    mod_file = os.path.join(model_save_path,f'Model{modNr}_Gen{gen}')

    # initialise models
    model_0 = keras.models.load_model(mod_file,compile=False)
    GA = GeneticAlgorithm(model_0, population_size = N, selection_rate = 0.2, mutation_rate = 0.1)
    # population = GA.initial_population()
    # set_trace()

    modelList = []
    for i in range(N):
        # model = keras.models.load_model(f'{savepath}/Model{modNr}_Gen{gen_0}',compile=False)
        # model = keras.models.clone_model(model0)
        if i==0:
            model = model_0
        else:
            model = NN.PerturbModelWeights(model_0, epsilon = 0.05)

        modelList.append(model)

    # saveChildren(modelList,savepath,gen_0,modNr)


    ################## profiling
    if profile==True:
        pr = cProfile.Profile()
        pr.enable()

    generations = 10
    generateCheckpoints()
    for gen in range(1,generations+1):
        reset_lists()
        initiate_pods(modelList)
        if gen%5==0:
            checkpointList[:] = []
            generateCheckpoints()
        setCheckpoints()
        print('simulation starts')
        runSim(gen,modNr,savepath)

        population = []
        scores = []
        for pod in podList:
            pod.fitnessScore = pod.doneCheckpoints + (1 - min(1, np.linalg.norm(pod.dest - pod.pos) / (np.sqrt(X**2 + Y**2)/2) ))
            population.append(pod.model.get_weights())
            scores.append(pod.fitnessScore)

        # set_trace()

        top_performers = GA.strongest_parents(population,scores)
        pairs = []
        while len(pairs) != GA.population_size:
        	pairs.append( GA.pair(top_performers) )

        # Crossover:
        base_offsprings =  []
        for pair in pairs:
        	offsprings = GA.crossover(pair[0][0], pair[1][0])
        	# 'offsprings' contains two chromosomes
        	base_offsprings.append(offsprings[-1])

        # Mutation:
        new_population = GA.mutation(base_offsprings)

        next_generation_models = NN.GetNextGeneration(new_population = new_population, input_shape = (6,))
        if (gen%10) == 0:
            saveChildren(next_generation_models,savepath,gen,modNr)
        modelList = next_generation_models

    if profile==True:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        # print(s.getvalue())
        with open('profile.txt','w') as fo:
            fo.write(s.getvalue())
    ##################

    print('done with all generations')
