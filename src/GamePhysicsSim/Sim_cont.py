import numpy as np
import sys
# import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
# from ..Models.conf1.Config import conf
import GamePhysicsSim.Visualise as Visualise
import GamePhysicsSim.NN as NN
import os

import tensorflow as tf
from tensorflow import keras

from IPython.core.debugger import set_trace

# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import sklearn
# from sklearn import preprocessing
import pickle

# dt = conf['dt']

AngularDrag = conf['AngularDrag']
AngularFriction = conf['AngularFriction']
Drag = conf['AirDrag']
Friction = conf['Friction']
N = 15 # number of pods
M = 8 # number of checkpoints
# Sizes
X = 800
Y = 800
PODSIZE = (30,30)
CheckPointSize = 20

def CheckPointCheck(pod,CheckPointSize):
    return np.linalg.norm(pod.dest - pod.pos) < CheckPointSize

def calculateFitness(fitness,tMax):
    fitnessScore = {}
    for ID,scoreData in fitness.items():
        fitnessScore[ID] = len(scoreData)-1 - Utils.sigmoid(scoreData[-1]/tMax)
    return fitnessScore

def reset_lists(IDList,fitness,podList):
    fitness.clear()
    IDList[:] = []
    podList[:] = []

def initiate_pods(modelList):
    # Initiate Pod
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)]) # random starting position
    theta0 = 3/2 * np.pi
    # podList = []
    for model in modelList:
        pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                            m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                            ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'])
        podList.append(pod)
        IDList.append(pod.ID)
        pod.model = model
    # initiate dict for fitnessScores
    for pod in podList:
        fitness[pod.ID] = [0]

    # return podList

def generateCheckpoints():
    CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
    checkpointList[:] = [np.array(c) for c in CheckGen]

def setCheckpoints():
    # initiate the first checkpoint for each pod:
    for pod in podList:
        pod.setDest(checkpointList[0])

def setup_sim(modelList):
    '''
    sets up simulation, devoid of pygame animations
    '''
    initiate_pods(modelList)
    setCheckpoints()
    # return podList

def update(t):
    t+=dt
    for pod in podList:
        x = scaler.transform(np.array([pod.getPodSteeringStats(delta_t=dt)]))
        Thrust,Torque = pod.NN_getSteering(x)
        # Thrust,Torque = pod.NN_getSteering()
        pod.Rotate(Torque = Torque, AngularDrag = AngularDrag, AngularFriction = AngularFriction, dt = dt)
        pod.Move(Thrust = Thrust, Drag = Drag, Friction = Friction, dt = dt)
        if CheckPointCheck(pod,CheckPointSize):
            doneCheckPoints = ( len(fitness[pod.ID]) % M )
            pod.setDest(checkpointList[doneCheckPoints])
            fitness[pod.ID].append(t)
        pod.podpatch.center = tuple(pod.pos)
        FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])*dt**2
        pod.accLine.set_data([pod.pos[0],pod.pos[0]+FT[0]],[pod.pos[1],pod.pos[1]+FT[1]])
        pod.velLine.set_data([pod.pos[0],pod.pos[0]+pod.v[0]],[pod.pos[1],pod.pos[1]+pod.v[1]])
        pod.destLine.set_data([pod.pos[0],pod.dest[0]],[pod.pos[1],pod.dest[1]])
        # ax.plot([pod.pos[0],pod.dest[0]],[pod.pos[1],pod.dest[1]]) # goal vec
        # FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
        # ax.plot([pod.pos[0],pod.pos[0]+FT[0]],[pod.pos[1],pod.pos[0]+FT[0]]) # acc vec
        # ax.plot([pod.pos[0],pod.pos[0]+pod.v[0]],[pod.pos[1],pod.pos[0]+pod.v[0]]) # v vec


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


def runSim(gen,modNr,savepath):
    fig,ax = setupPlot()
    ax = addPodPatches(ax)
    t = np.arange(0,tMax,dt)
    # anim = FuncAnimation(fig,animate,frames = t,fargs=(podList,))
    anim = FuncAnimation(fig,update,frames = t)
    anim.save(f'{savepath}Model{modNr}_Gen{gen}.mp4',fps = 30)

def saveChildren(children,savepath,gen,modNr):
    for j,child in enumerate(children):
        child.save(f'{savepath}/Model{modNr}_Gen{gen}_{j}')

dt = 0.05
tMax = 100

IDList = []
fitness = {}
checkpointList = []
podList = []



savepath = '/Users/bjornahlgren/Documents/Privat/Projects/codingames/csb/Models/conf1/'
modNr = 4
gen0 = 20

scalerPath = '/Users/bjornahlgren/Documents/Privat/Projects/codingames/csb/DataScalers/scaler_Model1.pkl'
scaler = pickle.load(open(scalerPath, 'rb'))

# set_trace()

modelList = []
for i in range(N):
    modname = f'{savepath}/Model{modNr}_Gen{gen0}_{i}'
    if os.path.exists(modname):
        model = keras.models.load_model(modname,compile=False)
    else:
        modname = f'{savepath}/Model{modNr}_Gen{gen0}_{0}'
        model = keras.models.load_model(modname,compile=False)
    # model = keras.models.clone_model(model0)
    # model.set_weights(model0.get_weights())
    # model = NN.PerturbModelWeights(model, epsilon = 0.1)
    modelList.append(model)

generations = 30
generateCheckpoints()
for gen in range(gen0+1,gen0+1+generations):
    reset_lists(IDList,fitness,podList)
    initiate_pods(modelList)
    if gen%5==0:
        checkpointList[:] = []
        generateCheckpoints()
    setCheckpoints()

    runSim(gen,modNr,savepath)

    fitnessScore = calculateFitness(fitness,tMax)
    scoreX = []
    scoreY = []
    for ID,score in [[k,v] for k, v in sorted(fitnessScore.items(), key=lambda item: item[1])]:
        print(f'{ID} has a score of {score}')
        scoreY.append(score)
        for pod in podList:
            if pod.ID == ID:
                scoreX.append(pod)
    podList = [x for _,x in sorted(zip(scoreY,scoreX), key=lambda pair: pair[0])] # sort podlist based on fitnessScore
    parents = podList[-2:] # get best two pods
    # set_trace()
    parents.append(np.random.choice(podList[:-2])) # also take a random pod from the remaining pods
    parentModels = [pod.model for pod in parents]

    children = NN.GetNextGeneration(parentModels = parentModels, NrChildren = N, epsilon = 0.02)
    saveChildren(children,savepath,gen,modNr)
    modelList = children
