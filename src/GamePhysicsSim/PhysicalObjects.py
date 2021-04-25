import numpy as np
import GamePhysicsSim.Utils as Utils
# import math

class RigidBody():
    '''
    Class for rigid body in 2D.

    dt is the time scale over which we consider the physical conditions to be constant.
    '''
    def __init__(self, pos0=np.array([0,0]), v0=np.array([0,0]), a0=np.array([0,0]), m=1, I=1/2., theta0=0, w0=0, alpha0=0):
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

    def ApplyForce(self, F):
        '''
        Sets acceleration based on input force, F.
        Assume for now that forces are applied only to the center of mass.
        '''
        self.a = self.a + F/self.m

    def ApplyTorque(self,Torque):
        '''
        Sets angular acceleration based on input Torque
        '''
        self.alpha = self.alpha + Torque/self.I

    def Friction(self, friction=0.1):
        '''
        Applies force in the negative v direction.
        Note that the friction force is proportional to v, such that in the absence of other
        acceleration, v = v0/(1+friction*t)
        '''
        self.ApplyForce(-self.v*friction)

    def Drag(self, Drag=0.1):
        '''
        Applies force in the negative v direction.
        Note that the drag force is proportional to v^2
        '''
        self.ApplyForce(-self.v*np.linalg.norm(self.v)*Drag)

    def AngularFriction(self, AngularFriction=0.1):
        '''
        Applies torque in the negative w direction.
        Note that the drag force is proportional to v
        '''
        self.ApplyTorque(-self.w*AngularFriction)

    def AngularDrag(self, AngularDrag=0.1):
        '''
        Applies torque in the negative w direction.
        Note that the drag force is proportional to v^2
        '''
        self.ApplyTorque(-np.sign(self.w)*self.w**2*AngularDrag)

    def DoMove(self, dt, vMax=None):
        '''
        Updates velocity based on current accelaration, then steps + v*dt. This is cleaner than updating assuming continuous acceleration during dt.
        '''
        # This is correct
        # self.pos = self.pos + self.v * dt + self.a * dt**2 / 2.
        # self.v = self.v + self.a * dt

        # This is faster and cleaner
        self.v = self.v + self.a * dt

        if vMax:
            self.v = np.array([np.sign(v_i) * min(abs(v_i),vMax) for v_i in self.v])

        self.pos = self.pos + self.v * dt  # + self.a * dt**2 / 2.


    def DoRot(self, dt, wMax=np.inf):
        '''
        Updates anugular velocity based on current angular accelaration, then steps + w*dt. This is cleaner than updating assuming continuous angular acceleration during dt.
        '''
        # this is the correct way of doing it
        # wIntegrated = (self.w + self.alpha * dt / 2.) # the effective angular velocity used for this time step, assuming constant torque over dt
        # AngVelocityRot = np.sign(wIntegrated)*min(abs(wIntegrated),wMax) # make effective angular velocity respect max value
        # self.theta = (self.theta + wIntegrated * dt) % (2*np.pi) # new angle is previous angle + effective angular velocity times time step, modulo 2pi
        # wNew = self.w + self.alpha * dt
        # self.w = np.sign(wNew)*min(abs(wNew),wMax)

        # this is the fast way of doing it (which becomes the same in the limit of small dt)
        wNew = self.w + self.alpha * dt
        self.w = np.sign(wNew)*min(abs(wNew),wMax)
        self.theta = (self.theta + self.w * dt) % (2*np.pi) # new angle is previous angle + effective angular velocity times time step, modulo 2pi
