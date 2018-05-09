"""
Module for wrapping an :class:`ncobj.Group` to make it as far as
possible appear as if it were a :class:`netCDF4.Dataset`.

This provides the ability for Iris to load and save ncobj data *as if* it were
dealing with regular `netCDF4.Dataset` objects.  This provides loading from,
and saving to, an `ncobj.Group` rather than an actual netcdf file.
This will allow Iris to use generalised netcdf data which can be manipulated
by code rather than it having to exist in an actual file.

The netCDF4 emulation is limited to what Iris netcdf load/save actually needs,
so no more general compatibility with the netCDF4 API is guaranteed.

"""

from collections import OrderedDict

import netCDF4 as nc
from ncobj.nc_dataset import write as ncobj_write


class Nc4ComponentMimic(object):
    """Abstract class providing general methods for all mimic object types."""
    def __init__(self, nco_component, parent_grp=None):
        """Create a mimic object wrapping a :class:`nco.Ncobj` component."""
        #: contained :class:`ncobj.NcObj` element.
        self.ncobj = nco_component
        # parent group object (mimic)
        self._parent_group_ncobj = parent_grp

    @property
    def name(self):
        return self.ncobj.name

    def group(self):
        return self._parent_group_ncobj

    def __eq__(self, other):
        return self.ncobj == other.ncobj

    def __ne__(self, other):
        return not self == other


def _name_as_string(obj_or_string):
    return (obj_or_string.name
            if hasattr(obj_or_string, 'name')
            else obj_or_string)


class DimensionMimic(Nc4ComponentMimic):
    """
    A Dimension object mimic wrapper.

    Dimension properties: name, length, unlimited, (+ parent-group)

    """
    @property
    def size(self):
        return 0 if self.isunlimited() else self.ncobj.length

    def __len__(self):
        return self.size

    def isunlimited(self):
        return self.ncobj.unlimited or not self.ncobj.length


class Nc4ComponentAttrsMimic(Nc4ComponentMimic):
    """An abstract class for an Nc4ComponentMimic with attribute access."""
    def ncattrs(self):
        return map(_name_as_string, self.ncobj.attributes)

    def getncattr(self, attr_name):
        if attr_name in self.ncobj.attributes.names():
            result = self.ncobj.attributes[attr_name].value
        else:
            raise AttributeError()
        return result

    def __getattr__(self, attr_name):
        return self.getncattr(attr_name)


class VariableMimic(Nc4ComponentAttrsMimic):
    """
    A Variable object mimic wrapper.

    Variable properties:
        name, dimensions, dtype, data (+ attributes, parent-group)
        shape, size, ndim

    """
    @property
    def dtype(self):
        return self.ncobj.data.dtype

    @property
    def datatype(self):
        return self.dtype

    @property
    def dimensions(self):
        return tuple(map(_name_as_string, self.ncobj.dimensions))

    def __getitem__(self, keys):
        if self.ndim == 0:
            return self.ncobj.data
        else:
            return self.ncobj.data[keys]

    @property
    def shape(self):
        return self.ncobj.data.shape

    @property
    def ndim(self):
        return self.ncobj.data.ndim

    @property
    def size(self):
        return self.ncobj.data.size


class GroupMimic(Nc4ComponentAttrsMimic):
    """
    A Group object mimic wrapper.

    Group properties:
        name, dimensions, variables, (sub)groups (+ attributes, parent-group)

    """
    def __init__(self, *args, **kwargs):
        super(GroupMimic, self).__init__(*args, **kwargs)

        self.dimensions = OrderedDict(
            [(dim.name, DimensionMimic(dim, parent_grp=self))
             for dim in self.ncobj.dimensions])

        self.variables = OrderedDict(
            [(var.name, VariableMimic(var, parent_grp=self))
             for var in self.ncobj.variables])

        self.groups = OrderedDict(
            [(grp.name, GroupMimic(grp, parent_grp=self))
             for grp in self.ncobj.groups])


class Nc4DatasetMimic(GroupMimic):
    def __init__(self, file_path=None, file_mode='r', *args, **kwargs):
        super(GroupMimic, self).__init__(*args, **kwargs)
        self._filepath = file_path
        self._file_mode = file_mode

    def close(self):
        # For read, nothing.  For write, create a dataset and save.
        if 'w' in self._file_mode:
            with nc.Dataset(filename=self.file_path, mode=self.file_mode,
                            **kwargs) as ds:
                ncobj_write(ds, self.ncobj)


def fake_readable_nc4python_dataset(ncobj_group):
    """
    Make a wrapper around an :class:`ncobj.Group` object to emulate a
    :class:`netCDF4.Dataset'.

    The resulting :class:`GroupMimic` supports the essential properties of a
    read-mode :class:`netCDF4.Dataset', enabling an arbitrary netcdf data
    structure in memory to be "read" as if it were a file
    (i.e. without writing it out to disk first).

    In particular, variable data access is delegated to the original,
    underlying :class:`ncobj.Group` object :  This provides deferred, sectional
    data access on request, in the usual way, avoiding the need to read in all
    the variable data.

    """
    return Nc4DatasetMimic(ncobj_group)


def fake_writeable_nc4python_dataset(file_path, file_mode='r'):
    """
    Make a wrapper around an :class:`ncobj.Group` object to emulate a
    :class:`netCDF4.Dataset'.

    The resulting :class:`GroupMimic` supports the essential properties of a
    write-mode :class:`netCDF4.Dataset', enabling an arbitrary netcdf data
    structure in memory to be "written" to it as if it were a file.

    In particular, variable data access is delegated to the original,
    underlying :class:`ncobj.Group` object :  This provides deferred, sectional
    data access on request, in the usual way, avoiding the need to read in all
    the variable data.

    """
    return Nc4DatasetMimic(file_path=file_path, file_mode=file_mode)
