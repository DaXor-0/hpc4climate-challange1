import comin
from yac import (YAC,
                 CloudGrid,
                 Location,
                 Field,
                 TimeUnit,
                 InterpolationStack,
                 Action,
                 Reduction,
                 def_calendar,
                 Calendar,
                 NNNReductionType,
                 )
from mpi4py import MPI
import numpy as np

nproma = comin.descrdata_get_global().nproma
host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
domain = comin.descrdata_get_domain(1)
ncells = domain.cells.ncells
ncells_global = domain.cells.ncells_global
nverts = domain.verts.nverts


class YACSmootherFactory():
    def __init__(self):
        def_calendar(Calendar.PROLEPTIC_GREGORIAN)
        self.yac = YAC(comm=host_comm)
        self.yac.def_datetime("0000-01-01", "9999-12-31")  # long enough
        self.comp = self.yac.def_comp("smoother")
        self.source_grid = CloudGrid(
            "smoother_source_grid",
            np.reshape(domain.cells.clon, (-1,), order="F")[:ncells],
            np.reshape(domain.cells.clat, (-1,), order="F")[:ncells],
        )
        self.source_points = self.source_grid.def_points(
            np.reshape(domain.cells.clon, (-1,), order="F")[:ncells],
            np.reshape(domain.cells.clat, (-1,), order="F")[:ncells],
        )
        self.target_grid = CloudGrid(
            "smoother_target_grid",
            np.reshape(domain.cells.clon, (-1,), order="F")[:ncells],
            np.reshape(domain.cells.clat, (-1,), order="F")[:ncells],
        )
        self.target_points = self.target_grid.def_points(
            np.reshape(domain.cells.clon, (-1,), order="F")[:ncells],
            np.reshape(domain.cells.clat, (-1,), order="F")[:ncells],
        )

    class _Smoother:
        def __init__(self, source_field, target_field):
            self.source_field = source_field
            self.target_field = target_field

        def smooth(self, data, data_out=None):
            self.source_field.put(data)
            data, info = self.target_field.get(buf=data_out)
            assert info == Action.COUPLING
            return data

    def create_smoother(self, n, name="smoother", collection_size=1):
        source_field = Field.create(name+"source",
                                    self.comp,
                                    self.source_points,
                                    collection_size,
                                    "1",
                                    TimeUnit.MILLISECOND,
                                    )

        target_field = Field.create(name+"target",
                                    self.comp,
                                    self.target_points,
                                    collection_size,
                                    "1",
                                    TimeUnit.MILLISECOND,
                                    )

        istack = InterpolationStack()
        ##istack.add_nnn(NNNReductionType.GAUSS, n=n, max_search_distance=0.0, scale=1)
        istack.add_nnn(NNNReductionType.AVG, n=n, max_search_distance=0.0, scale=1)
        self.yac.def_couple("smoother", "smoother_source_grid", name+"source",
                            "smoother", "smoother_target_grid", name+"target",
                            "1", TimeUnit.MILLISECOND,
                            Reduction.TIME_NONE,
                            istack)

        self.yac.enddef()
        return YACSmootherFactory._Smoother(source_field, target_field)
