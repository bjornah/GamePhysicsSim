import GamePhysicsSim as GPS
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Visualise as Visualise
import numpy as np
import sys
import pygame
import random
import pandas as pd



conf['FPS'] = 30 #make simul fast
dt = conf['dt']
AngularDrag = conf['AngularDrag']
AngularFriction = conf['AngularFriction']
Drag = conf['AirDrag']
Friction = conf['Friction']

N = 20 # number of pods
M = 20 # number of checkpoints

# Sizes
X = 800
Y = 800
PODSIZE = (30,30)
CheckPointSize = 15

# create checkpoints
CheckGen = zip(np.random.randint(X,size=M),np.random.randint(Y,size=M))
checkpointList = [np.array(c) for c in CheckGen]

# Initiate Pod
podList = []
IDList = []

# Length of simulation in seconds
tMax = 60

# settings for PID controller
# K0 = 5
# f0 = 0.3
# Kp = K0 * 1
# Ki = 0.2 * Kp * f0
# Kd = Kp / (9*f0)
K0 = 16
f0 = 0.4
Kp = K0 * 1
Ki = 0.2 * Kp * f0
Kd = Kp / (15*f0)

for i in range(N):
    pod_pos_0 = np.array([np.random.randint(X),np.random.randint(Y)])
    theta0 = np.random.uniform(0,2*np.pi)
    pod = PodClass.Pod(pos0 = pod_pos_0, w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                        m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                        ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'], wMax = conf['wMax'])
    pod.PID_init(Kp,Ki,Kd)
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

# initiate x_train and y_train
x_train = []
y_train = []
# for pod in podList:
    # x_train.append(pod.getPodSteeringStats(dt))

updateParamsNr = 0
t = 0

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
        x_train.append(pod.getPodSteeringStats(dt))
        ### Steering
        Torque,(v_desired,a_required) = pod.GetSteering_PID(pod.dest,dt)
        dist = np.linalg.norm(pod.dest - pod.pos)
        thetaDiff = abs(np.arccos(np.dot(Utils.unit_vector([np.cos(pod.theta),np.sin(pod.theta)]),Utils.unit_vector(pod.dest - pod.pos))))
        if thetaDiff>np.pi/60:
            # ThrustGauge = np.clip(np.cos(abs(thetaDiff)),0,1) # + 10/360*2*np.pi #*(thetaDiff/(0.523-thetaDiff))
            ThrustGauge = np.clip(np.exp(-abs(thetaDiff/(np.pi/8))),0,1)
        else:
            ThrustGauge = 1
        # ThrustGauge = np.clip(np.cos(thetaDiff),0,1)
        ThrustGauge = ThrustGauge * Utils.sigmoid(dist/300.)**2 * np.clip((dist+70)/300.,0,1)
        Thrust = conf['ThrustMax'] * ThrustGauge

        pod.AddThrust(Thrust = Thrust)
        pod.AddTorque(Torque = Torque)

        y_train.append(np.array([Thrust,Torque]))
        pod.Rotate(AngularDrag = AngularDrag, AngularFriction = AngularFriction, dt = dt)
        pod.Move(Drag = Drag, Friction = Friction, dt = dt)
        # x_train.append(pod.getPodSteeringStats(delta_t=dt))

        if Utils.CheckPointCheck(pod,CheckPointSize):
            if pod.ID not in IDList:
                print(f'{pod.ID} not in IDList. wtf.')
            try:
                doneCheckPoints = ( len(fitness[pod.ID]) % M )
            except:
                print(fitness[pod.ID], sys.exc_info())
            # pod.setDest(checkpointList[doneCheckPoints])
            pod.setDest(random.choice(checkpointList))
            fitness[pod.ID].append(t)
            pod.pid.reset()


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
    fpsClock.tick(conf['FPS'])


pygame.display.quit()
pygame.quit()

x_train = np.array(x_train)
# x_train = x_train[:-N]
y_train = np.array(y_train)

df_x = pd.DataFrame(data = np.array(x_train).T)
df_y = pd.DataFrame(data = np.array(y_train).T)
df_x.to_csv('train_data_errors_v1.csv',index=False)
df_y.to_csv('train_data_response_v1.csv',index=False)
