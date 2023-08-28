import warnings

import mrcfile
import numpy as np
from mrcfile.mrcfile import MrcFile

class MrcFileStack(MrcFile):
    DTYPE_FROM_MODE = {0: np.int8,
                       1: np.int16,
                       2: np.float32,
                       4: np.complex64,
                       6: np.uint16
                       }

    def __init__(self, name, mode='r', overwrite=False, permissive=False,
                 map_function=None, **kwargs):
        """Initialise a new :class:`MrcFile` object.

        The given file name is opened in the given mode. For mode ``r`` or
        ``r+`` the header, extended header and data are read from the file. For
        mode ``w+`` a new file is created with a default header and empty
        extended header and data arrays.

        Args:
            name: The file name to open, as a string or pathlib Path.
            mode: The file mode to use. This should be one of the following:
                ``r`` for read-only, ``r+`` for read and write, or ``w+`` for a
                new empty file. The default is ``r``.
            overwrite: Flag to force overwriting of an existing file if the
                mode is ``w+``. If :data:`False` and a file of the same name
                already exists, the file is not overwritten and an exception is
                raised. The default is :data:`False`.
            permissive: Read the file in permissive mode. (See
                :class:`mrcfile.mrcinterpreter.MrcInterpreter` for details.)
                The default is :data:`False`.
            map_function: A function to be applied to each image. None if no function is going to be applied


        Raises:
            :exc:`ValueError`: If the mode is not one of ``r``, ``r+`` or
                ``w+``.
            :exc:`ValueError`: If the file is not a valid MRC file and
                ``permissive`` is :data:`False`.
            :exc:`ValueError`: If the mode is ``w+``, the file already exists
                and overwrite is :data:`False`.
            :exc:`OSError`: If the mode is ``r`` or ``r+`` and the file does
                not exist.

        Warns:
            RuntimeWarning: If the file appears to be a valid MRC file but the
                data block is longer than expected from the dimensions in the
                header.
            RuntimeWarning: If the file is not a valid MRC file and
                ``permissive`` is :data:`True`.
            RuntimeWarning: If the header's ``exttyp`` field is set to a known
                value but the extended header's size is not a multiple of the
                number of bytes in the corresponding dtype.
        """
        super().__init__(name, mode=mode, overwrite=overwrite, permissive=permissive,
                         header_only=True, **kwargs)

        self.fname = name
        self.f = self._iostream

        self.shape = tuple(self.header[ax] for ax in ["ny", "nx"])
        self.n = int(self.header["nz"])
        self.map_function = map_function if map_function is not None else self._identity
        self.dtype = MrcFileStack.DTYPE_FROM_MODE[int(self.header['mode'])]
        self.image_nelem = np.product(self.shape).astype(np.int64) #np.long
        self.elem_size = np.ones(1, dtype=np.float32).itemsize
        self.image_stride = self.image_nelem * self.elem_size
        self._data = None

    def _identity(self, x):
        return x

    def close(self):
        self.f.close()
        super().close()

    def __len__(self):
        return self.n
    def _get_offset(self, idx):
        offset = 1024 + idx * self.image_stride
        return offset

    def get_idx(self, idx):
        self.f.seek(self._get_offset(idx))
        try:
            image = np.fromfile(self.f, dtype=self.dtype, count=self.image_nelem).reshape(self.shape)
        except ValueError as e:
            raise ValueError(str(e)+f" file {self.fname}")
        if self.map_function is not None:
            image = self.map_function(image)
        return image

    def __getitem__(self, idx):
        assert idx >= 0, "Error, negative indexes are currently not suported"
        return self.get_idx(idx)


    def get_header_prop(self, name):
        assert name in self.header.dtype.names
        return self.header[name]

    @property
    def data(self):
        if self._data  is None:
            warnings.warn(
                "You have requested to load all the data into memory. This could cause Out of Memory given file sizes")
            self._data = mrcfile.read(self.fname)
        return self._data


    @staticmethod
    def dump_npImages_from_iterator(stackFname, particlesIterator,
                               nParticles, particle_shape, sampling_rate, overwrite=False,
                               batch_size:int=1000):

        with mrcfile.new_mmap(stackFname, shape=(nParticles, *particle_shape), mrc_mode=2,
                                   overwrite=overwrite) as mrc_out:

            mrc_out.voxel_size = sampling_rate
            # Iterate through the particles in batches
            for start_idx in range(0, nParticles, batch_size):
                end_idx = min(start_idx + batch_size, nParticles)
                # Iterate through the batch and write the images to the memory-mapped file
                for idx in range(start_idx, end_idx):

                    mrc_out.data[idx] = next(particlesIterator)


    @staticmethod
    def dump_npImages(stackFname, npImages, sampling_rate, overwrite=False):
        with mrcfile.new(stackFname, overwrite=overwrite) as mrc_out:
            mrc_out.set_data(npImages.astype(np.float32))
            mrc_out.voxel_size = sampling_rate