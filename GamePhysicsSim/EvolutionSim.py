import numpy as np
import sys
import pygame

import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
import GamePhysicsSim.Visualise as Visualise

def CheckPointCheck(pod,CheckPointSize):
    return np.linalg.norm(pod.dest - pod.pos) < CheckPointSize

def calculateFitness(fitness,tMax):
    fitnessScore = {}
    for ID,scoreData in fitness.items():
        fitnessScore[ID] = len(scoreData)-1 - Utils.sigmoid(scoreData[-1]/tMax)
    return fitnessScore

conf['FPS'] = 100 #make simul fast
dt = conf['dt']
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
CheckPointSize = 15

# create checkpoints
CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
checkpointList = [np.array(c) for c in CheckGen]

# initiate weights
W = np.zeros((6,32,32,2))

# Initiate Pod
# either use these options to generate pods with identical initial conditions, or set them further down to get unique settings for each pod (only for testing purposes)
# pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)]) # random starting position
# theta0 = 3/2 * np.pi
podList = []
IDList = []
for i in range(N):
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)])
    theta0 = np.random.uniform(0,2*np.pi)
    pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                        m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                        ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'])
    podList.append(pod)
    IDList.append(pod.ID)


# initiate each pod to the first checkpoint:
for pod in podList:
    pod.setDest(checkpointList[0])

# initiate dict for fitnessScores
fitness = {}
for pod in podList:
    fitness[pod.ID] = [0]

# Set initial steering parameters
ThrustVec = np.zeros(N)
TorqueVec = np.zeros(N)

# in case you want to watch the result:
# imgFile = None
imgFile = '/Users/bjornahlgren/Documents/Icke_jobbrelaterat/Projects/codingames/csb/SpaceShip2.png'
# DISPLAYSURF,podImg,fpsClock = Visualise.initiate_animation(imgFile,_DISPLAYSIZE = (X,Y), PODSIZE = PODSIZE)

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
tMax = 25
while t<tMax: # the main game loop
    for event in pygame.event.get():
        if event.type in [pygame.QUIT,pygame.WINDOWEVENT_CLOSE,]:
            pygame.display.quit()
            pygame.quit()
            sys.exit()

    t+=dt
    DISPLAYSURF.fill(conf['WHITE'])

    for checkpoint in checkpointList:
        pygame.draw.circle(DISPLAYSURF, (0,0,150), checkpoint, CheckPointSize)

    for i,pod in enumerate(podList):

        # Thrust,Torque = pod.NN_getSteering()
        (Torque,Thrust),(v_desired,a_required,w_required,alpha_required) = pod.GetSteering(pod.dest,dt)
        # print(Thrust)
        pod.Rotate(Torque = Torque, AngularDrag = AngularDrag, AngularFriction = AngularFriction, dt = dt)
        pod.Move(Thrust = Thrust, Drag = Drag, Friction = Friction, dt = dt)

        if CheckPointCheck(pod,CheckPointSize):
            if pod.ID not in IDList:
                print(f'{pod.ID} not in IDList. wtf.')
            try:
                doneCheckPoints = ( len(fitness[pod.ID]) % M )
            except:
                print(fitness[pod.ID], sys.exc_info())
            pod.setDest(checkpointList[doneCheckPoints])
            fitness[pod.ID].append(t)

            print(f'pod {pod.ID} reached checkpoint #{doneCheckPoints}')


        pod_pos = np.rint(pod.pos).astype(int)
        goalVec = pod.dest - pod_pos

        pygame.draw.line(DISPLAYSURF,(255,0,0),pod_pos,pod_pos + goalVec)

        FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
        pygame.draw.line(DISPLAYSURF,(255,255,0),pod_pos,pod_pos + FT)
        pygame.draw.line(DISPLAYSURF,(255,0,255),pod_pos,pod_pos + pod.v)

        if (pod_pos[0] > 0) & (pod_pos[1] > 0):
            angle = (pod.theta + np.pi/2.)/(2*np.pi)*360. # note that +np.pi/2. comes from the fact that we want an initial rotation of 90 deg of the original image
            image_pivot = (PODSIZE[0]/2.,PODSIZE[1]/2.) # around the point which we rotate, in the coordinate system of the image
            surface_pivot = pod_pos # the point around which we rotate, in the coordinate system of the screen
            Visualise.blitRotate(DISPLAYSURF, podImgList[i], surface_pivot, image_pivot, -angle)
    # print(t)
    pygame.display.update()
    fpsClock.tick(conf['FPS'])

pygame.display.quit()
pygame.quit()

fitnessScore = calculateFitness(fitness,tMax)
for ID,score in fitnessScore.items():
    print(f'{ID} has a score of {score}')

sys.exit()
print('done')
