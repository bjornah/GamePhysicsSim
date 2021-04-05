import numpy as np
import GamePhysicsSim.Utils as Utils
from GamePhysicsSim.PhysicalObjects import RigidBody
from IPython.core.debugger import set_trace
from simple_pid import PID

class Pod(RigidBody):
    '''
    Inherits from RigidBody class. Adds Thurst, Rotate, Turn, and Move functions.

    dt is the time scale over which we calculate changes to physical variables

    delta_t indicates the time scale during which we consider input forces to be constant (e.g. user input in terms of Thrust and Torque)
    '''

    def __init__(self, TorqueMax=1, ThrustMax=100, wMax=None, wMin = 0.1, vMin = 0.1, dest = None, model = None, podpatch = None,
                destLine = None, velLine = None, accLine = None, *args, **kwargs):
        super(Pod, self).__init__(*args, **kwargs)
        self.ThrustMax = ThrustMax
        self.TorqueMax = TorqueMax
        self.wMax      = wMax
        self.wMin      = wMin
        self.vMin      = vMin
        self.dest      = dest
        self.ID        = id(self)
        self.model     = model
        self.podpatch  = podpatch
        self.destLine  = destLine
        self.velLine   = velLine
        self.accLine   = accLine
        self.pid       = None

    def ApplyThrust(self,Thrust):
        '''
        Scale thrust to respect max thrust.

        We assume that the thrust is coming directly from the back of the ship.

        Calculate FT (i.e. the force on the rigid body) from the given thrust and current angle of ship
        '''
        Thrust = min(Thrust,self.ThrustMax)
        FT = np.array([Thrust*np.cos(self.theta),Thrust*np.sin(self.theta)])
        self.ApplyForce(FT)

    def Move(self,Thrust,Drag,Friction,dt):
        '''
        Thrust is scalar value between 0 and ThrustMax. This is applied in direction the pod is facing.
        This is translated into a pure force vector on the underlying RigidBody object.

        Note that we perform a new tallying of all forces applied to our pod every turn.
        This is why the acceleration is set to zero at the end. Otherwise we would need to deal with Delta_Force instead,
        which seems more cumbersome
        '''
        self.ApplyThrust(Thrust)
        self.Drag(Drag)
        self.Friction(Friction)
        self.DoMove(dt)
        if (Thrust==0) and (np.linalg.norm(self.v)<self.vMin):
            self.v = np.array([0.,0.])
        self.a = 0

    def Rotate(self,Torque,AngularDrag,AngularFriction,dt):
        '''
        Note that we perform a new tallying of all torques applied to our pod every turn.
        This is why the angular acceleration (alpha) is set to zero at the end.
        Otherwise we would need to deal with Delta_torque instead, which seems more cumbersome

        Note that we assume that the friction of the Rotating motion is proportional to the angular velocity squared
        (c.f. linear drag, see also e.g. https://physics.stackexchange.com/questions/304742/angular-drag-on-body)
        '''
        Torque = np.sign(Torque)*min(abs(Torque),self.TorqueMax)

        self.ApplyTorque(Torque)
        self.AngularDrag(AngularDrag)
        self.AngularFriction(AngularFriction)
        self.DoRot(dt)
        if (Torque==0) and (abs(self.w)<self.wMin):
            self.w = 0
        self.alpha = 0

    def StopRotate(self,AngularDrag,AngularFriction,dt):
        Torque = -self.I * self.w/dt
        Torque = np.sign(Torque)*min(abs(Torque),self.TorqueMax)
        self.ApplyTorque(Torque)
        # self.AngularDrag(AngularDrag)
        # self.AngularFriction(AngularFriction)
        self.DoRot(dt)
        self.alpha = 0

    def Break(self,Drag,Friction,dt):
        # set_trace()
        Break = np.linalg.norm(self.v) / dt
        Break = min(Break,self.ThrustMax)
        # print(f'self.v = {self.v}')
        # print(f'type(self.v) = {type(self.v)}')
        # v = self.v
        F_Break = -Break * Utils.unit_vector(self.v)
        self.ApplyForce(F_Break)
        self.Drag(Drag)
        self.Friction(Friction)
        self.DoMove(dt)
        # if (Thrust==0) and (np.linalg.norm(self.v)<self.vMin):
            # self.v = np.array([0.,0.])
        self.a = 0

    # def Turn(self,):
    #     '''
    #     To include a simplified way of turning for manual control. Automatically stops rotating once turning action is over
    #     (not immediately, but will automatically apply StopRotate function, probably).
    #     '''
    #     return

    def GetSteering(self,pos_dest,delta_t):
        '''
        Assumes we have a destination in mind, given by 2D vector pos_dest. We limit max acceleration and Torque to ThrustMax and TorqueMax, respectively.

        Returns Torque and Thrust to be put into Move and Rotate.

        This steering clearly needs more work. When using with torque rather than angular velocity to control turning we need something more intricate than this.
        '''
        v_desired = (pos_dest - self.pos)/delta_t
        a_required = (v_desired - self.v)/delta_t
        if np.linalg.norm(a_required)>0:
            # theta_required = np.arctan(a_required[0]/a_required[1])
            theta_required = Utils.full_angle(a_required,np.array([1,0]))
            delta_theta = theta_required - self.theta #- self.w*delta_t # this last term accounts for the fact that we will soon rotate
            if delta_theta > np.pi:
                delta_theta = delta_theta - 2*np.pi
            w_required = delta_theta/delta_t
        else:
            theta_required = self.theta
            delta_theta = 0.0001 # to avoid division by zero
            w_required = 0

        delta_w = w_required - self.w
        # print(delta_w)

        # if (self.w*delta_t/1./delta_theta<1):# and (self.w*delta_t/1./delta_theta>0):
            # alpha_required = 0
        # else:
        alpha_required = delta_w/delta_t

        # limit to max allowed values of torque and thrust
        Torque = np.sign(alpha_required)*min(abs(alpha_required*self.I),self.TorqueMax)
        Thrust = min(np.linalg.norm(a_required)*self.m,self.ThrustMax)

        return (Torque,Thrust),(v_desired,a_required,w_required,alpha_required) # apply acceleration and torque accordingly

    def PID_init(self,Kp,Ki,Kd):
        pid = PID(Kp, Ki, Kd, setpoint=0)
        pid.output_limits = (-self.TorqueMax, self.TorqueMax)
        self.pid = pid

    def GetSteering_PID(self,pos_dest,delta_t):
        '''
        Assumes we have a destination in mind, given by 2D vector pos_dest. We limit max acceleration and Torque to ThrustMax and TorqueMax, respectively.

        Returns Torque and Thrust to be put into Move and Rotate.

        This steering clearly needs more work. When using with torque rather than angular velocity to control turning we need something more intricate than this.
        '''
        v_desired = (pos_dest - self.pos)/delta_t
        a_required = (v_desired - self.v)/delta_t
        if np.linalg.norm(a_required)>0:
            theta_required = Utils.full_angle(a_required,np.array([1,0]))
            delta_theta = theta_required - self.theta
            # delta_theta = self.theta - theta_required #- self.w*delta_t # this last term accounts for the fact that we will soon rotate
            if delta_theta > np.pi:
                delta_theta = delta_theta - 2*np.pi
        # delta_theta = delta_theta/np.pi
        Torque = -self.pid(delta_theta,delta_t)
        # print(f'delta_theta = {delta_theta}\nTorque = {Torque}')

        return Torque,(v_desired,a_required) # apply torque accordingly

    def GetSteering_w(self,pos_dest,delta_t,wMax):
        '''
        Assumes we have a destination in mind, given by 2D vector pos_dest, and a highest allowed rotation of the pod, given by wMax.
        '''
        v_desired = (pos_dest - self.pos)/delta_t
        a_required = (v_desired - self.v)/delta_t
        if np.linalg.norm(a_required)>0:
            theta_required = Utils.full_angle(a_required,np.array([1,0])) # np.arctan(a_required[0]/a_required[1])
            delta_theta = theta_required - self.theta
            if delta_theta > np.pi:
                delta_theta = delta_theta - 2*np.pi
            w_required = delta_theta/delta_t
        else:
            theta_required = self.theta
            w_required = 0
        delta_w = w_required - self.w
        delta_w = np.sign(delta_w)*min(abs(delta_w),wMax)
        self.w = delta_w
        return v_desired,a_required,w_required

    def setDest(self,dest):
        self.dest = dest

    def getPodSteeringStats(self,delta_t):
        vx,vy           = self.v
        w               = self.w
        # theta           = self.theta
        try:
            delta_x,delta_y = self.dest - self.pos
            v_desired = (self.dest - self.pos)/delta_t
            a_required = (v_desired - self.v)/delta_t
            if np.linalg.norm(a_required)>0:
                theta_required = Utils.full_angle(a_required,np.array([1,0]))
                delta_theta = theta_required - self.theta
                if delta_theta > np.pi:
                    delta_theta = delta_theta - 2*np.pi
            else:
                delta_theta = 0
        except TypeError:
            print(f'pod {self.ID} appears to be missing a destination. Do not forget to set ut using pod.setDest(dest)', sys.exc_info()[0])
        # return np.array([vx,vy,w,theta,delta_x,delta_y,delta_theta])
        return np.array([vx,vy,w,delta_x,delta_y,delta_theta])


    def NN_getSteering(self,x,model=None):
        '''
        Returns Thrust and TorqueMax given current velocity, position and angle relative to the destination, and angular velocity
        Requires uncompiled model. This is a factor of ~30 faster than predicting for a singe data point using a compiled model
        '''
        if model==None:
            model = self.model
        Thrust,Torque = model(x)[0]

        return Thrust,Torque
