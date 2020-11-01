import numpy as np
import sys
import pygame

import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
# from ..Models.conf1.Config import conf
import GamePhysicsSim.Visualise as Visualise
import GamePhysicsSim.NN as NN



def CheckPointCheck(pod,CheckPointSize):
    return np.linalg.norm(pod.dest - pod.pos) < CheckPointSize

def calculateFitness(fitness,tMax):
    fitnessScore = {}
    for ID,scoreData in fitness.items():
        fitnessScore[ID] = len(scoreData)-1 - Utils.sigmoid(scoreData[-1]/tMax)
    return fitnessScore

conf['FPS'] = 40 #make simul fast
dt = conf['dt']
AngularDrag = conf['AngularDrag']
AngularFriction = conf['AngularFriction']
Drag = conf['AirDrag']
Friction = conf['Friction']

N = 10 # number of pods
M = 10 # number of checkpoints

# Sizes
X = 1000
Y = 1000
PODSIZE = (30,30)
CheckPointSize = 15

# create checkpoints
# CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
# checkpointList = [np.array(c) for c in CheckGen]

fitness = {}
IDList = []
checkpointList = []

def reset_lists(IDList,fitness):
    fitness.clear()
    IDList[:] = []
    checkpointList[:] = []

def initiate_pods(modelList):
    # Initiate Pod
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)]) # random starting position
    theta0 = 3/2 * np.pi
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

    return podList

def setCheckpoints():
    CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
    checkpointList[:] = [np.array(c) for c in CheckGen]
    # initiate the first checkpoint for each pod:
    for pod in podList:
        pod.setDest(checkpointList[0])

# in case you want to watch the result:
imgFile = '/Users/bjornahlgren/Documents/Privat/Projects/codingames/csb/SpaceShip2.png'

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
tMax = 60
while t<tMax: # the main game loop
    for event in pygame.event.get():
        if event.type in [pygame.QUIT,pygame.WINDOWEVENT_CLOSE,]:
            pygame.display.quit()
            pygame.quit()
            sys.exit()

    t+=dt
    updateParamsNr+=1
    DISPLAYSURF.fill(conf['WHITE'])

    for checkpoint in checkpointList:
        pygame.draw.circle(DISPLAYSURF, (0,0,150), checkpoint, CheckPointSize)

    for i,pod in enumerate(podList):

        # Thrust,Torque = pod.NN_getSteering()
        # (Torque,Thrust),(v_desired,a_required,w_required,alpha_required) = pod.GetSteering(pod.dest,dt)
        Torque,(v_desired,a_required) = pod.GetSteering_PID(pod.dest,dt)
        dist = np.linalg.norm(pod.dest - pod.pos)
        thetaDiff = abs(np.arccos(np.dot(Utils.unit_vector([np.cos(pod.theta),np.sin(pod.theta)]),Utils.unit_vector(pod.dest - pod.pos))))
        if thetaDiff>(30/360.*2*np.pi):
            ThrustGauge = np.clip(np.cos(thetaDiff+30/360.*2*np.pi),0,1)
        else:
            ThrustGauge = np.clip(np.cos(thetaDiff),0,1)
        ThrustGauge = ThrustGauge * Utils.sigmoid(dist/300.) * np.clip((dist+30)/300.,0,1)
        print(f'dist = {dist}')
        Thrust = conf['ThrustMax'] * ThrustGauge
        # Thrust = 0
        # print(Torque)
        # print(pod.dest)
        # print(pod.pos)
        # print(dt)
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
    pygame.display.update()
    if updateParamsNr%20==0:
        print(t)
    fpsClock.tick(conf['FPS'])


pygame.display.quit()
pygame.quit()

fitnessScore = calculateFitness(fitness,tMax)
for ID,score in fitnessScore.items():
    print(f'{ID} has a score of {score}')

sys.exit()
print('done')


#########################


# filenamelist = [0]*5
# for i in range(5):
#     pygame.draw.rect(displaysurf, (0, 0, 0), pygame.Rect(0, 0, IMGSIZE, IMGSIZE), 0)
#     pygame.draw.rect(displaysurf, (255, 255, 255), pygame.Rect(i*20, i*20, 20, 20), 0)
#     #Save as PNG images on disk:
#     filenamelist[i] = "pic" + str(i) + ".png"
#     pygame.image.save(displaysurf, filenamelist[i])
#
# #Combine into a GIF using ImageMagick's "convert"-command (called using subprocess.call()):
# convertexepath = "C:/Program Files (x86)/ImageMagick-6.8.9-Q16/convert.exe"  # Hardcoded
# convertcommand = [convertexepath, "-delay", "10", "-size", str(IMGSIZE) + "x" + str(IMGSIZE)] + filenamelist + ["anim.gif"]
# subprocess.call(convertcommand)
#
# #Remove the PNG files (if they were meant to be temporary):
# for filename in filenamelist:
#     os.remove(filename)
