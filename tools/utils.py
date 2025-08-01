import numpy as np


def gcd(a_lon, a_lat, b_lon, b_lat):
    """
    Great Circle Distance
    """
    a = np.sin((b_lat - a_lat)/2)**2 + np.cos(a_lat)*np.cos(b_lat)*np.sin((a_lon - b_lon)/2)**2
    c = 2*np.asin(np.sqrt(a))
    return c
