import numpy as np
import zarr
import isodate
from mpi4py import MPI
import comin
import sys
jg = 1
domain = comin.descrdata_get_domain(jg)
ncells_glb = domain.cells.ncells_global
host_comm = MPI.Comm.f2py(comin.parallel_get_host_mpi_comm())
ncells_local = domain.cells.ncells
global_idx = np.asarray(domain.cells.glb_index) - 1
owner_mask = np.reshape(domain.cells.decomp_domain, shape=(-1,), order="F")[:ncells_local] == 0


def get_global_coordinates():
    lat_local = np.rad2deg(domain.cells.clat).ravel('F')[:ncells_local]
    lon_local = np.rad2deg(domain.cells.clon).ravel('F')[:ncells_local]

    lat_gathered = host_comm.gather((global_idx, lat_local), root=0)
    lon_gathered = host_comm.gather((global_idx, lon_local), root=0)

    if host_comm.rank == 0:
        def assemble(gathered):
            idx, val = zip(*gathered)
            flat_idx = np.concatenate(idx)
            flat_val = np.concatenate(val)
            result = np.empty(ncells_glb, dtype='f4')
            result[flat_idx] = flat_val
            return result

        lat_global = assemble(lat_gathered)
        lon_global = assemble(lon_gathered)
    else:
        lat_global = None
        lon_global = None

    return lat_global, lon_global


lat_global, lon_global = get_global_coordinates()


class EventWriter:
    def __init__(self, interval_iso, path, chunks=(1, 10000), dtype='f4'):
        self.interval_iso = interval_iso
        self.path = path
        self.chunks = chunks
        self.dtype = dtype

        self.lat_global = lat_global
        self.lon_global = lon_global
        self.host_comm = host_comm
        self.ncells_glb = ncells_glb

        interval = comin.descrdata_get_simulation_interval()
        self.exp_start = np.datetime64(interval.exp_start)
        self.exp_stop = np.datetime64(interval.exp_stop)

        self.fields = []  # list of field names
        self.zarr_group = None
        self.ntimesteps = None
        self.z_idx = 0  # TODO: in a restarted run this needs to be computed from run_start
        self.nlevel = {}

    def register_output_field(self, name, nlevel=None):
        """Register the name of a field to be written."""
        self.fields.append(name)
        self.nlevel[name] = nlevel

    def initialize_output(self):
        """Create the Zarr group and datasets after fields are registered."""
        interval_ms = int(isodate.parse_duration(self.interval_iso).total_seconds() * 1000)
        interval_np = np.timedelta64(interval_ms, 'ms')
        self.ntimesteps = int((self.exp_stop - self.exp_start) / interval_np)

        if self.host_comm.rank == 0:
            group = zarr.open_group(self.path, mode='w')

            for name in self.fields:
                nlevel = self.nlevel[name]
                zdim = [] if nlevel is None else [nlevel]
                zchunk = [] if nlevel is None else [5]  # TODO: make it configurable
                arr = group.create_dataset(
                    name,
                    shape=(self.ntimesteps + 1, *zdim, self.ncells_glb),
                    chunks=(self.chunks[0], *zchunk, self.chunks[1]),
                    dtype=self.dtype,
                    fill_value=np.nan
                )
                zdimname = [] if nlevel is None else [f"level{zdim[0]}"]
                arr.attrs["_ARRAY_DIMENSIONS"] = ("time", *zdimname, "cell")

            group.create_dataset("lat", shape=(self.ncells_glb,), dtype="f4", data=self.lat_global).attrs["_ARRAY_DIMENSIONS"] = ("cell",)
            group.create_dataset("lon", shape=(self.ncells_glb,), dtype="f4", data=self.lon_global).attrs["_ARRAY_DIMENSIONS"] = ("cell",)
            group.create_dataset("time", shape=(self.ntimesteps + 1,), dtype="datetime64[ns]").attrs["_ARRAY_DIMENSIONS"] = ("time",)

        self.host_comm.barrier()
        self.zarr_group = zarr.open_group(self.path, mode='a')

    def is_time_to_write(self):
        """Check if it's time to write based on simulation time and interval."""
        current_time = np.datetime64(comin.current_get_datetime())
        interval_ns = int(isodate.parse_duration(self.interval_iso).total_seconds() * 1e9)
        elapsed_ns = (current_time - self.exp_start).astype('timedelta64[ns]').astype(int)
        return (elapsed_ns % interval_ns) == 0

    def write(self, mask_or_indices, **field_data):
        """Gather field values and write them to the Zarr output.

        Parameters
        ----------
        mask_or_indices : np.array
            Either a boolean mask of length ncells_local or a list/array of indices
        **field_data:
            fields to be written to zarr in the shape how comin provides them (nproma, [nlevel,] nblks, ....)
        """
        unregistered = set(field_data) - set(self.fields)
        if unregistered:
            raise ValueError(f"Unregistered fields: {unregistered}", file=sys.stderr)

        if mask_or_indices.dtype == bool:
            indices = (mask_or_indices & owner_mask).nonzero()[0]
        else:
            indices = indices[owner_mask[indices]]

        def slice_field(name, data):
            if self.nlevel[name] is None:
                return np.reshape(data, (-1,), order="F")[indices]
            else:
                nlev = self.nlevel[name]
                return np.swapaxes(data, 0, 1).reshape((nlev, -1), order="F")[:, indices]

        glb_idx = global_idx[indices]
        gathered = {
            name: self.host_comm.gather((glb_idx, slice_field(name, data)), root=0)
            for name, data in field_data.items()
        }

        if self.host_comm.rank == 0:
            for name, gathered_data in gathered.items():
                idx_list, val_list = zip(*gathered_data)
                all_indices = np.concatenate(idx_list)
                all_values = np.concatenate(val_list, axis=-1)
                if self.nlevel[name] is None:
                    self.zarr_group[name][self.z_idx, all_indices] = all_values
                else:
                    self.zarr_group[name][self.z_idx, :, all_indices] = all_values

            self.zarr_group["time"][self.z_idx] = comin.current_get_datetime()
            self.z_idx += 1
