import numpy as np
import os

ROOT_DIR = os.path.dirname('/'.join(os.path.abspath(__file__).split('/')[:-2]))


conf = {
# Clock and FPS
'FPS' : 30, # frames per second setting
'_DISPLAYSIZE' : (800,800),
# Colors
'WHITE' : (255, 255, 255),
# Pod
'PODSIZE' : (40,40),
'pod_pos_0' : np.array([150,150]),
'm_pod' : 30, # kg
'r_pod' : 0.1, # m
'TorqueMax' : 30, # N * m : kg * m^2 * s^−2
'ThrustMax' : 100000, # N : kg * m * s^−2
'AirDrag' : 0.47, # 0.47 # taken from https://asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
'AngularDrag' : 0.95, # 0.95 # taken from nowhere, the entire concept is flawed as done here. We hope that it looks good enough
'Friction' : 0.15,
'AngularFriction' : 0.95,
'vMin' : 10, # a pod slower than this with zero thrust will stop due to static friction
'vMax' : 1000, # this should not be needed, since friction and air drag will take care of this.
'wMin' : 0.15, # a pod with angular velocity slower than this with zero torque will stop due to static friction
'wMax' : 200./360*2*np.pi, # maximum angular velocity in radians per second
'v0'   : np.array([0,0]),
'a0'   : np.array([0,0]),
'conf_version': 'conf1'
}

conf['I_pod'] = 0.5 * conf['m_pod'] * conf['r_pod']**2
conf['dt'] = 1./conf['FPS']
conf['Delta'] = int(conf['FPS']/6.)
conf['delta_t'] = conf['Delta']*conf['dt'] # time step for input updates
