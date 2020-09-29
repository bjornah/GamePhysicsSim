import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def signed_angle(a1,a2):
    '''
    Calculates the signed angle between vectors a1 and a2. returns float in range [-pi,pi], in a positive rotation from x to y axis
    '''
    angle = np.arctan2(a1[1],a1[0]) - np.arctan2(a2[1],a2[0])
    if angle<-np.pi:
        angle = angle+np.pi
    return angle
