import comin
import numpy as np
from mpi4py import MPI

# get descriptive data
domain = comin.descrdata_get_domain(1)   # domain ID 1
ncells_glb = domain.cells.ncells_global
ncells_local = domain.cells.ncells
global_idx = np.asarray(domain.cells.glb_index) - 1

# Use ComIn's so-called host MPI communicator which is the MPI communicator
# that comprises all MPI tasks of the ICON simulation which are involved in
# the ComIn callbacks.
host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())

# Helper function to assemble the gathered field
def assemble(gathered):
    idx, val = zip(*gathered)
    result = np.empty(ncells_glb, dtype='f4')
    result[np.concatenate(idx)] = np.concatenate(val)
    return result


# Constructor: Access the field `tot_prec` (surface pressure) from ICON:
@comin.EP_SECONDARY_CONSTRUCTOR
def secondary_constructor():
    global tot_prec
    tot_prec = comin.var_get(
        [comin.EP_ATM_WRITE_OUTPUT_BEFORE], 
        ("tot_prec", 1), 
        comin.COMIN_FLAG_READ
    )


# Callback function: performs an MPI gather operation on process 0.
@comin.EP_ATM_WRITE_OUTPUT_BEFORE
def gather_field():
    tot_prec_local = np.asarray(tot_prec).ravel('F')[:ncells_local]
    tot_prec_gathered = host_comm.gather((global_idx, tot_prec_local), root=0)
    tot_prec_global = assemble(tot_prec_gathered) if host_comm.rank == 0 else None
