import comin
from mpi4py import MPI

# initialization
host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())

domain = comin.descrdata_get_domain(1)
nlev = domain.nlev
ncells = domain.cells.ncells


@comin.EP_SECONDARY_CONSTRUCTOR
def sec_ctr():
    # comin which variables you want to access.
    ...


@comin.EP_ATM_WRITE_OUTPUT_AFTER
def detect():
    # the detection code here
    ...
