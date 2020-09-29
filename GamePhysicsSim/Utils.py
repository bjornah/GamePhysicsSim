import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    # print(vector)
    # print(type(vector))
    # if any(np.isnan(vector)):
        # vector = np.zeros(vector.shape)
    norm = np.linalg.norm(vector)
    # print(norm)
    # print(norm)
    # print(np.isnan(norm))
    # if not np.isnan(norm):
    if (norm>0):
        return vector / np.linalg.norm(vector)
    else:
        return np.zeros(vector.shape)

def signed_angle(a1,a2):
    '''
    Calculates the signed angle between vectors a1 and a2. returns float in range [-pi,pi], in a positive rotation from x to y axis
    '''
    angle = np.arctan2(a1[1],a1[0]) - np.arctan2(a2[1],a2[0])
    if angle<-np.pi:
        angle = angle+np.pi
    return angle

def full_angle(a1,a2):
    '''
    Calculates the signed angle between vectors a1 and a2. returns float in range [0,2*pi], in a positive rotation from x to y axis
    '''
    angle = np.arctan2(a1[1],a1[0]) - np.arctan2(a2[1],a2[0])
    if angle<0:
        angle = angle+2*np.pi
    return angle

def sigmoid(X):
    return 1./(1+np.exp(-X))
