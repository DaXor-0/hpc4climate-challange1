import comin
import numpy as np
from mpi4py import MPI
from scipy.spatial import KDTree


def lonlat2xyz(lon, lat):
    clat = np.cos(lat)
    return clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)


host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
n_dom = comin.descrdata_get_global().n_dom
nproma = comin.descrdata_get_global().nproma

# build a KDTree of the cell centers
domains = [comin.descrdata_get_domain(jg) for jg in range(1, n_dom+1)]
ncells = [domain.cells.ncells for domain in domains]
clons = [np.reshape(domain.cells.clon, shape=(-1,), order="F")[:ncell]
         for domain, ncell in zip(domains, ncells)]
clats = [np.reshape(domain.cells.clat, shape=(-1,), order="F")[:ncell]
         for domain, ncell in zip(domains, ncells)]
trees = [KDTree(np.c_[lonlat2xyz(clon.ravel(), clat.ravel())])
         for clon, clat in zip(clons, clats)]


def find_close_cells(lon, lat, *, radius, jg=1):
    assert radius < 90, "radius must be smaller than 90°"
    xyz_radius = 2*np.sin(np.deg2rad(radius)/2)

    xyz_coords = np.vstack(lonlat2xyz(lon, lat)).T
    return np.concatenate(list(np.asarray(i, dtype=int) for i in trees[jg-1].query_ball_point(xyz_coords, xyz_radius)))


def find_nearest_cells(lon, lat, *, jg=1):
    xyz_coords = np.stack(lonlat2xyz(lon, lat), axis=-1)
    return trees[jg-1].query(xyz_coords)[1]
