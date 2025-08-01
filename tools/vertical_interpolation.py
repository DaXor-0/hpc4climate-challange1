import numpy as np


def plev_interpolation(field, pres, plev, ncells):
    pidx = np.sum(pres < plev, axis=1, keepdims=True)
    pidx = np.minimum(pidx, pres.shape[1]-2)
    pres_pidx = np.take_along_axis(pres, pidx, axis=1)[:, 0, ..., None].reshape((-1,), order="F")[:ncells]
    pres_pidx1 = np.take_along_axis(pres, pidx+1, axis=1)[:, 0, ..., None].reshape((-1,), order="F")[:ncells]
    w = np.divide((plev-pres_pidx), (pres_pidx1-pres_pidx))
    field_pidx = np.take_along_axis(field, pidx, axis=1)[:, 0, ..., None].reshape((-1,), order="F")[:ncells]
    field_pidx1 = np.take_along_axis(field, pidx+1, axis=1)[:, 0, ..., None].reshape((-1,), order="F")[:ncells]
    return (1-w)*field_pidx + w*field_pidx1
