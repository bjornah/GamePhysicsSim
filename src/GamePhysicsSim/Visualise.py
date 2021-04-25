import GamePhysicsSim.Utils as Utils
import GamePhysicsSim.Pod as PodClass
from GamePhysicsSim.Config import conf
import numpy as np
import sys
import pygame

#############



#############

def blitRotate(surf, image, pos, originPos, angle):
    # calcaulate the axis aligned bounding box of the rotated image
    w, h       = image.get_size()
    box        = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]
    min_box    = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box    = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    # calculate the translation of the pivot
    pivot        = pygame.math.Vector2(originPos[0], -originPos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move   = pivot_rotate - pivot
    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originPos[0] + min_box[0] - pivot_move[0], pos[1] - originPos[1] - max_box[1] + pivot_move[1])
    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    # rotate and blit the image
    surf.blit(rotated_image, origin)
    # draw rectangle around the image
    # pygame.draw.rect (surf, (255, 0, 0), (*origin, *rotated_image.get_size()),2)

def initiate_animation(imgFile,_DISPLAYSIZE = (800,640), PODSIZE = (40,40)):
    # Clock and FPS
    fpsClock = pygame.time.Clock()

    # Display
    DISPLAYSURF = pygame.display.set_mode(_DISPLAYSIZE, 0, 32)
    pygame.display.set_caption('Animation')

    # Pod
    if imgFile == None:
        podImg = pygame.Surface(PODSIZE)
        podImg.fill((100,100,255))
    else:
        podImg = pygame.image.load(imgFile).convert_alpha()
        podImg = pygame.transform.scale(podImg, PODSIZE)

    return DISPLAYSURF,podImg,fpsClock


def RunManualControl(imgFile,conf):

    DISPLAYSURF,podImg,fpsClock = initiate_animation(imgFile,_DISPLAYSIZE = conf['_DISPLAYSIZE'], PODSIZE = conf['PODSIZE'])
    # Initiate Pod
    theta0 = np.pi*1/2.
    pod = PodClass.Pod(pos0 = conf['pod_pos_0'], w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                        m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                        ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'], wMax = conf['wMax'])

    # Set initial steering parameters

    run = True
    updateParamsNr = 0

    while run: # the main game loop
        Thrust = pod.Thrust
        Torque = pod.Torque

        updateParamsNr += 1
        pod.Move(Drag = conf['AirDrag'], Friction = conf['Friction'], dt = conf['dt'])
        pod.Rotate(AngularDrag = conf['AngularDrag'], AngularFriction = conf['AngularFriction'], dt = conf['dt'])

        pod_pos = np.rint(pod.pos).astype(int)

        DISPLAYSURF.fill(conf['WHITE'])
        if (pod_pos[0] > 0) & (pod_pos[1] > 0):
            angle = (pod.theta + np.pi/2.)/(2*np.pi)*360. % 360 # note that +np.pi/2. comes from the fact that we want an initial rotation of 90 deg of the original image
            image_pivot = (conf['PODSIZE'][0]/2.,conf['PODSIZE'][1]/2.) # around the point which we rotate, in the coordinate system of the image
            surface_pivot = pod_pos # the point around which we rotate, in the coordinate system of the screen
            blitRotate(DISPLAYSURF, podImg, surface_pivot, image_pivot, -angle)

        for event in pygame.event.get():
            if event.type in [pygame.QUIT,pygame.WINDOWEVENT_CLOSE,]:
                pygame.display.quit()
                pygame.quit()
                run = False
                # sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    pygame.quit()
                    run = False
                    # sys.exit()
        if run:
            if updateParamsNr%conf['Delta']==0:
                keys_pressed = pygame.key.get_pressed()
                if keys_pressed[pygame.K_LEFT]:
                    Torque -= 3
                if keys_pressed[pygame.K_RIGHT]:
                    Torque += 3
                if keys_pressed[pygame.K_UP]:
                    Thrust += 300
                if keys_pressed[pygame.K_DOWN]:
                    Thrust -= 300
                if keys_pressed[pygame.K_z]:
                    Thrust = 0
                    pod.Break(conf['AirDrag'],conf['Friction'],conf['dt'])
                if keys_pressed[pygame.K_x]:
                    Torque = 0
                    pod.StopRotate(conf['AngularDrag'],conf['AngularFriction'],conf['dt'])
            pod.AddThrust(Thrust = Thrust)
            pod.AddTorque(Torque = Torque)


            if not any(np.isnan(pod.v)):
                pygame.draw.line(DISPLAYSURF,(255,0,255),pod_pos,pod_pos + pod.v)
            FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
            pygame.draw.line(DISPLAYSURF,(255,255,0),pod_pos,pod_pos + FT)

            pygame.display.update()
            fpsClock.tick(conf['FPS'])


def run_mouse_control_PID_steering(imgFile,conf):



    dt = conf['dt']
    CheckPointSize = 15
    DISPLAYSURF,podImg,fpsClock = initiate_animation(imgFile,_DISPLAYSIZE = conf['_DISPLAYSIZE'], PODSIZE = conf['PODSIZE'])
    # Initiate Pod
    pod_pos_0 = conf['pod_pos_0']
    theta0 = np.pi*1/2.

    pod = PodClass.Pod(pos0 = conf['pod_pos_0'], w0 = 0, v0 = conf['v0'], a0 = conf['a0'], alpha0 = 0, theta0 = theta0,
                        m=conf['m_pod'], I=conf['I_pod'], TorqueMax = conf['TorqueMax'],
                        ThrustMax = conf['ThrustMax'], vMin = conf['vMin'], wMin = conf['wMin'], wMax = conf['wMax'])

    # Set initial steering parameters
    Thrust = 0
    Torque = 0
    K0 = 16
    f0 = 0.4
    Kp = K0 * 1
    Ki = 0.2 * Kp * f0
    Kd = Kp / (15*f0)
    pod.PID_init(Kp,Ki,Kd)

    run = True
    updateParamsNr = 0

    clicked = False
    pos_click = pod_pos_0
    pod.setDest(pod_pos_0)

    while run: # the main game loop
        updateParamsNr += 1
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                pos_click = np.array(pygame.mouse.get_pos())
                clicked = True
                pod.setDest(pos_click)
            if event.type in [pygame.QUIT,pygame.WINDOWEVENT_CLOSE,]:
                pygame.display.quit()
                pygame.quit()
                run = False
                # sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    pygame.quit()
                    run = False
                    # sys.exit()
        if run:
            if not clicked:
                pos_click = pod.pos
            # if updateParamsNr%conf['Delta']==0:

            dist = np.linalg.norm(pod.dest - pod.pos)

            if dist>1:
                Torque,(v_desired,a_required) = pod.GetSteering_PID(pod.dest,dt)

            thetaDiff = abs(np.arccos(np.dot(Utils.unit_vector([np.cos(pod.theta),np.sin(pod.theta)]),Utils.unit_vector(pod.dest - pod.pos))))

            if thetaDiff>np.pi/60:
                ThrustGauge = np.clip(np.exp(-abs(thetaDiff/(np.pi/8))),0,1)
            else:
                ThrustGauge = 1
            ThrustGauge = ThrustGauge * Utils.sigmoid(dist/300.)**2 * np.clip((dist+70)/300.,0,1)
            Thrust = conf['ThrustMax'] * ThrustGauge

            pod.AddThrust(Thrust = Thrust)
            pod.AddTorque(Torque = Torque)

            pod.Move(Drag = conf['AirDrag'], Friction = conf['Friction'], dt = dt)
            pod.Rotate(AngularDrag = conf['AngularDrag'], AngularFriction = conf['AngularFriction'], dt = dt)

            pod_pos = np.rint(pod.pos).astype(int)

            if Utils.CheckPointCheck(pod,CheckPointSize):
                pod.StopRotate(AngularDrag = conf['AngularDrag'], AngularFriction = conf['AngularFriction'], dt = dt)
                pod.Break(Drag = conf['AirDrag'], Friction = conf['Friction'], dt = dt)
                pod.pid.reset()

            DISPLAYSURF.fill(conf['WHITE'])
            if clicked:
                pygame.draw.circle(DISPLAYSURF, (0,0,255), pos_click, 5)
                pygame.draw.line(DISPLAYSURF,(255,0,0),pod_pos,pod_pos + v_desired)
                pygame.draw.line(DISPLAYSURF,(0,255,0),pod_pos,pod_pos + a_required)
                FT = np.array([Thrust*np.cos(pod.theta),Thrust*np.sin(pod.theta)])
                pygame.draw.line(DISPLAYSURF,(255,255,0),pod_pos,pod_pos + FT)

            if (pod_pos[0] > 0) & (pod_pos[1] > 0):
                angle = (pod.theta + np.pi/2.)/(2*np.pi)*360. # note that +np.pi/2. comes from the fact that we want an initial rotation of 90 deg of the original image
                image_pivot = (conf['PODSIZE'][0]/2.,conf['PODSIZE'][1]/2.) # around the point which we rotate, in the coordinate system of the image
                surface_pivot = pod_pos # the point around which we rotate, in the coordinate system of the screen
                blitRotate(DISPLAYSURF, podImg, surface_pivot, image_pivot, -angle)

                if not any(np.isnan(pod.v)):
                    pygame.draw.line(DISPLAYSURF,(255,0,255),pod_pos,pod_pos + pod.v)

            pygame.display.update()
            fpsClock.tick(conf['FPS'])


if __name__ == "__main__":
    imgFile = '../images/SpaceShip2.png'
    run_mouse_control_PID_steering(imgFile,conf)
    # RunManualControl(imgFile,conf)
