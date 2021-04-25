import numpy as np
import cProfile
import pstats
from functools import wraps


def profile(output_file=None, sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """
    Taken from https://towardsdatascience.com/how-to-profile-your-code-in-python-e70c834fad89
    A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + '.prof'
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, 'w') as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


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

def CheckPointCheck(pod,CheckPointSize):
    return np.linalg.norm(pod.dest - pod.pos) < CheckPointSize
    
