import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import random
import time

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

# from sklearn.preprocessing import StandardScaler
# import sklearn
from sklearn import preprocessing


confdir = conf['conf_version']

training_data_dir = os.path.join(ROOT_DIR,f'Models/{confdir}/TrainingData')
data_scaler_path = os.path.join(ROOT_DIR,f'Models/{confdir}/DataScalers')
model_save_path = os.path.join(ROOT_DIR,f'Models/{confdir}')

mod_file = os.path.join(model_save_path,'Model1_Gen0')

model1 = keras.models.load_model(mod_file,compile=True)
model1_nc = keras.models.load_model(mod_file,compile=False)

model = model1_nc

scaler_file = os.path.join(data_scaler_path,'scaler_x.pkl')
scaler = pickle.load(open(scaler_file, 'rb'))

FPS = 30 #make simul fast
dt = conf['dt']
AngularDrag = conf['AngularDrag']
AngularFriction = conf['AngularFriction']
Drag = conf['AirDrag']
Friction = conf['Friction']

N = 5 # number of pods
M = 10 # number of checkpoints

# Sizes
X = 800
Y = 800
PODSIZE = (30,30)
CheckPointSize = 15

# create checkpoints
CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
checkpointList = [np.array(c) for c in CheckGen]

podList = []
IDList = []
start = time.time()
# Initiate Pods
for i in range(N):
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)])
    theta0 = np.random.uniform(0,2*np.pi)
    pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                        m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'], model=model,
                        ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'], wMax = conf['wMax'])
    podList.append(pod)
    IDList.append(pod.ID)

# initiate the first checkpoint for each pod:
for pod in podList:
    # pod.setDest(checkpointList[0])
    pod.setDest(random.choice(checkpointList))

# initiate dict for fitnessScores
fitness = {}
for pod in podList:
    fitness[pod.ID] = [0]

# Set initial steering parameters
ThrustVec = np.zeros(N)
TorqueVec = np.zeros(N)

# in case you want to watch the result:
imgFile = os.path.join(ROOT_DIR,'images/SpaceShip2.png')

# Clock and FPS
fpsClock = pygame.time.Clock()
# Display
DISPLAYSURF = pygame.display.set_mode((X,Y), 0, 32)
pygame.display.set_caption('Pod training')
# Pods
podImgList = []
for pod in podList:
    podImg = pygame.image.load(imgFile).convert_alpha()
    podImg = pygame.transform.scale(podImg, PODSIZE)
    podImgList.append(podImg)

updateParamsNr = 0
t = 0
tMax = 10
while t<tMax: # the main game loop
    for event in pygame.event.get():
        if event.type in [pygame.QUIT,pygame.WINDOWEVENT_CLOSE,]:
            pygame.display.quit()
            pygame.quit()

    t+=dt
    updateParamsNr+=1
    DISPLAYSURF.fill(conf['WHITE'])

    for checkpoint in checkpointList:
        pygame.draw.circle(DISPLAYSURF, (0,0,150), checkpoint, CheckPointSize)

    for i,pod in enumerate(podList):
        x = scaler.transform(np.array([pod.getPodSteeringStats(delta_t=dt)]))
        pod.NN_getSteering(x)

        pod.AddThrust(Thrust = pod.Thrust)
        pod.AddTorque(Torque = pod.Torque)

        pod.Rotate(AngularDrag = AngularDrag, AngularFriction = AngularFriction, dt = dt)
        pod.Move(Drag = Drag, Friction = Friction, dt = dt)

        if Utils.CheckPointCheck(pod,CheckPointSize):
            if pod.ID not in IDList:
                print(f'{pod.ID} not in IDList. wtf.')
            try:
                doneCheckPoints = ( len(fitness[pod.ID]) % M )
            except:
                print(fitness[pod.ID], sys.exc_info())
            pod.setDest(random.choice(checkpointList))
            fitness[pod.ID].append(t)

        pod_pos = np.rint(pod.pos).astype(int)
        goalVec = pod.dest - pod_pos

        pygame.draw.line(DISPLAYSURF,(255,0,0),pod_pos,pod_pos + goalVec)

        FT = np.array([pod.Thrust*np.cos(pod.theta),pod.Thrust*np.sin(pod.theta)])
        pygame.draw.line(DISPLAYSURF,(255,255,0),pod_pos,pod_pos + FT)
        pygame.draw.line(DISPLAYSURF,(255,0,255),pod_pos,pod_pos + pod.v)

        if (pod_pos[0] > 0) & (pod_pos[1] > 0):
            angle = (pod.theta + np.pi/2.)/(2*np.pi)*360. # note that +np.pi/2. comes from the fact that we want an initial rotation of 90 deg of the original image
            image_pivot = (PODSIZE[0]/2.,PODSIZE[1]/2.) # around the point which we rotate, in the coordinate system of the image
            surface_pivot = pod_pos # the point around which we rotate, in the coordinate system of the screen
            Visualise.blitRotate(DISPLAYSURF, podImgList[i], surface_pivot, image_pivot, -angle)
    pygame.display.update()
    if updateParamsNr%20==0:
        print(t)
        print(pod.Thrust)
    fpsClock.tick(FPS)

end = time.time()

print(f'elapsed time  = {end-start}s\nfor \ntMax = {tMax}\ndt = {dt}\nFPS = {FPS})')
pygame.display.quit()
pygame.quit()


fitnessScore = Utils.calculateFitness(fitness,tMax)
for ID,score in fitnessScore.items():
    print(f'{ID} has a score of {score}')

print('done')
