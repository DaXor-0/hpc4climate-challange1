import comin
import numpy as np
from mpi4py import MPI
from scipy.spatial import KDTree


def lonlat2xyz(lon, lat):
    clat = np.cos(lat)
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
jg = 1  # we only support domain 1 for now
nproma = comin.descrdata_get_global().nproma

# build a KDTree of the cell centers
domain = comin.descrdata_get_domain(jg)
clon = np.asarray(domain.cells.clon)
clat = np.asarray(domain.cells.clat)
xyz = np.c_[lonlat2xyz(clon.ravel(), clat.ravel())]
tree = KDTree(xyz)


def extend_mask(mask, radius):
    """
    Extends the mask by a given radius (given in (great circle degrees).
    Thought to be used for mask that have globally only a few positive points.

    mask can be a tuple of list of indices in dimensions nproma, nblks or a boolean mask of shape (nproma, nblks)

    return a tuple of lists of indices in nproma and nblks dimension that can be used to slice variables.
    """
    coords = np.vstack([clon[mask], clat[mask]])
    coords = host_comm.allgather(coords)
    coords = np.hstack(coords)
    xyz_coords = np.vstack(lonlat2xyz(*coords)).T

    assert radius < 90, "radius must be smaller than 90°"
    xyz_radius = np.sqrt(2)*np.abs(np.sin(np.deg2rad(radius)))

    neighbors = np.concatenate(tree.query_ball_point(xyz_coords, xyz_radius))
    return neighbors % nproma, neighbors // nproma
