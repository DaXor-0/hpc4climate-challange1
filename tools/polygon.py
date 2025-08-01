import comin

from shapely import multipolygons, union, union_all, simplify, Polygon, MultiPolygon, Point
from mpi4py import MPI
import numpy as np
from .cell_search import find_close_cells
from .utils import gcd


nproma = comin.descrdata_get_global().nproma
host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())


def mask_to_polygon(cell_mask, *, jg=1, root=0, tolerance=None):
    domain = comin.descrdata_get_domain(jg)
    ncells = domain.cells.ncells
    if len(cell_mask.shape) == 2:
        cell_mask = cell_mask.reshape((-1,), order="F")[:ncells]
    if cell_mask.dtype == np.bool_:
        cell_mask = cell_mask.nonzero()
    v_idx = np.reshape(domain.cells.vertex_idx, (-1, 3), order="F")[cell_mask, :] - 1
    v_blk = np.reshape(domain.cells.vertex_blk, (-1, 3), order="F")[cell_mask, :] - 1
    vlon = np.asarray(domain.verts.vlon)[v_idx, v_blk]
    vlat = np.asarray(domain.verts.vlat)[v_idx, v_blk]
    cell_polygons = multipolygons(np.stack([vlon, vlat], axis=-1))
    polygon = union_all(cell_polygons)
    if tolerance is not None:
        polygon = simplify(polygon, tolerance=tolerance)

    # reduce over all processes
    if root is None:
        polygon = host_comm.allreduce(polygon, op=union)
        if tolerance is not None:
            polygon = simplify(polygon, tolerance=tolerance)
    else:
        polygon = host_comm.reduce(polygon, op=union, root=root)
        if root == host_comm.rank and tolerance is not None:
            polygon = simplify(polygon, tolerance=tolerance)

    return polygon


def check_contour_criteria(center, r, condition, jg=1):
    """Checks the contour criteria as described in

    Ullrich, P. A. and Zarzycki, C. M.: TempestExtremes: a framework
    for scale-insensitive pointwise feature tracking on unstructured
    grids, Geosci. Model Dev., 10, 1069–1090,
    https://doi.org/10.5194/gmd-10-1069-2017, 2017.

    Section 2.6. But it we use a different algorithm because of the
    domain decomposed setting we have.

    condition must be a predicate that is evaluated for each cell.
    """
    lon, lat = center
    disc_cells = find_close_cells(lon, lat, radius=r + 0.5, jg=jg)  # +0.5 to ensure that all centers are found
    satisfied = ~condition(disc_cells)
    p = mask_to_polygon(satisfied, root=None)
    if type(p) is Polygon:
        p = MultiPolygon([p])
    for g in p.geoms:
        if g.contains(Point(*center)):
            for point in g.convex_hull.boundary.coords:
                if gcd(*point, lon, lat) > r/180*np.pi:
                    return False, None
            return True, g
    comin.print_warning("Cannot apply contour criteria. No polygon contains the center.")
    return False, None
