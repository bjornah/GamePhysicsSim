import numpy as np
import sys
import pygame

import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
import GamePhysicsSim.Visualise as Visualise

N = 10 # number of pods
M = 10 # number of checkpoints

X = 1000
Y = 1000
PODSIZE = (30,30)

# create checkpoints
CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
checkpointList = [c for c in CheckGen]

DISPLAYSURF,podImg,fpsClock = initiate_animation(imgFile,_DISPLAYSIZE = (X,Y), PODSIZE = PODSIZE)



# Initiate Pod
pod_pos_0 = conf['pod_pos_0']
pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = 0, a0 = 0, alpha0 = 0, theta0 = np.pi*1/2.,
                    m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                    ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'])
