import numpy as np
import GamePhysicsSim.Utils as Utils
# import math

class RigidBody():
    '''
    Class for rigid body in 2D.

    dt is the time scale over which we calculate changes to physical variables
    '''
    def __init__(self,pos0=np.array([0,0]),v0=np.array([0,0]),a0=np.array([0,0]),m=1,I=1/2.,theta0=0,w0=0,alpha0=0):
        '''
        '''
        self.pos = pos0
        self.v = v0
        self.a = a0
        self.m = m
        self.I = I
        self.theta = theta0
        self.w = w0
        self.alpha = alpha0

    def CalcEnergy(self):
        self.Ek = 0.5 * self.m * self.v**2
        self.Ew = 0.5 * self.I * self.w**2
        self.Etot = self.Ek + self.Ew

    def CalcMomentum(self):
        self.p = self.v * self.m

    def ApplyForce(self,F):
        '''
        Assume for now that forces are applied only to the center of mass.
        '''
        self.a = self.a + F/self.m

    def ApplyTorque(self,Torque):
        self.alpha = self.alpha + Torque/self.I

    def Friction(self,friction=0.1):
        '''
        Note that the friction force is proportional to v, such that in the absence of other
        acceleration v = v0/(1+friction*t)
        '''
        self.ApplyForce(-self.v*friction)

    def Drag(self,Drag=0.1):
        '''
        Note that the drag force is proportional to v^2
        '''
        self.ApplyForce(-self.v*np.linalg.norm(self.v)*Drag)

    def AngularDrag(self,AngularDrag=0.1):
        '''
        Note that the drag force is proportional to v^2
        '''
        self.ApplyTorque(-np.sign(self.w)*self.w**2*AngularDrag)

    def AngularFriction(self,AngularFriction=0.1):
        '''
        Note that the drag force is proportional to v^2
        '''
        self.ApplyTorque(-self.w*AngularFriction)

    def DoMove(self,dt):
        self.v = self.v + self.a * dt
        if any(np.isnan(self.v)):
            self.v = np.zeros(self.v.shape)
        self.pos = self.pos + self.v * dt #+ self.a * dt**2 / 2.

    def DoRot(self,dt):
        self.w = self.w + self.alpha * dt
        if np.isnan(self.w):
            self.w = 0
        self.theta = self.theta + self.w * dt # + self.alpha * dt**2 / 2.
        self.theta = self.theta % (2*np.pi)
