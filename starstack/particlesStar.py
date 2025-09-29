import re
from os import PathLike

import numpy as np
import starfile
import pandas as pd
import os.path as osp
from typing import Optional, Union, Iterator, List, Dict, Any, Tuple, TypeVar
from queue import Queue
from threading import Thread
from collections import defaultdict
from .constants import RELION_ANGLES_NAMES, RELION_SHIFTS_NAMES, RELION_PRED_POSE_CONFIDENCE_NAME
from .mrcFileStack import MrcFileStack

T = TypeVar('T', bound='ParticlesStarSet')
class ParticlesStarSet():
    """
    A class for handling particle data stored in RELION star files along with their associated MRC image stacks.

    This class provides functionality to read, manipulate, and save particle metadata and images from RELION
    experiments. It handles both the metadata stored in star files and the particle images stored in MRC
    stack files.

    Attributes:
        starFname (str): Path to the input star file
        particlesDir (Optional[str]): Directory containing particle MRC stack files
        starts_at_1 (bool): Whether particle numbering starts at 1 (RELION standard) or 0
        optics_md (pd.DataFrame): Optics metadata from the star file
        particles_md (pd.DataFrame): Particle metadata from the star file
        partNum_fname (List[Tuple[int, str]]): List of tuples mapping particle numbers to filenames
        _imgStackFileHandlers (Dict[str, MrcFileStack]): Cache of MRC file handlers
    """
    def __init__(self, starFname: Union[PathLike, str, Dict[str, pd.DataFrame]], particlesDir: Optional[str] = None,
                 starts_at_1: bool = True):
        """

        :param starFname: The star filename with the metadata, or a dictionary like the one returned by starfile.read
        :param particlesDir: The directory of the particles .mrcs file(s). If None, it assumes that starFname
        imageName paths for .mrcs are referred from the current working directory
        :param starts_at_1: Whether the first particle at the starfile is named 0 or 1. In Relion it starts with 1
        """
        self.starFname = starFname
        self.particlesDir = particlesDir
        self.starts_at_1 = starts_at_1
        self.read(starFname)

    def read(self, starFname: Union[PathLike, str, Dict[str, pd.DataFrame]]) -> None:
        """
        Read particle metadata from a star file.

        :param starFname: The star filename with the metadata
        """

        if isinstance(starFname, dict):
            data = starFname
        elif isinstance(starFname, (str, PathLike)):
            data = starfile.read(starFname)
        else:
            raise RuntimeError(f"Not valid input {starFname}")

        self._init_md(data)

    def _init_md(self, data: Dict[str, pd.DataFrame]):
        if "particles" in data:
            self.optics_md = data["optics"]
            self.particles_md = data["particles"]
        else:
            self.particles_md = data
            self.optics_md = None

        self.particles_md.set_index("rlnImageName", inplace=True, drop=False)
        self.partNum_fname = self._compute_partNumFname()

        self._imgStackFileHandlers = {}

    def _compute_partNumFname(self):
        if self.particlesDir:
            _get_fname = lambda d, b: osp.join(self.particlesDir, d, b)
        else:
            _get_fname = lambda d, b: osp.join(d, b) if d else b

        def get_fname(dirname, basename):
            fullFname = _get_fname(dirname, basename)
            if not osp.isfile(fullFname) and isinstance(self.starFname, (str, PathLike)):
                # try to get it from the same directory as the starFname
                fullFname = osp.join(osp.dirname(self.starFname), basename)
            return fullFname

        partNum_dirname_basename_list = [(int(d["partNum"]), d["dirname"] if d["dirname"] else "",
                                          d["basename"]) for d in
                                         self.particles_md["rlnImageName"].map(lambda x: split_particle_and_fname(x))]
        return [(int(pn) - bool(self.starts_at_1), get_fname(d, b)) for pn, d, b in
                partNum_dirname_basename_list]

    def shuffle(self):
        self.particles_md = self.particles_md.sample(frac=1)
        self.partNum_fname = self._compute_partNumFname()

    def emptySet(self):
        """Reset the instance to an empty state by clearing all data."""
        self.particles_md = None
        self.optics_md = None
        self.partNum_fname = []
        self._imgStackFileHandlers = {}

    @property
    def particle_shape(self) -> Tuple[int, int]:
        """
        Get the shape of particles in the dataset.

        Returns:
            Tuple containing the height and width of particles
        """
        partNum, fname = self.partNum_fname[0]
        img_stack_handler = self.getImgStackFilehandler(fname)
        particle_shape = img_stack_handler[partNum].shape
        return particle_shape

    @property
    def sampling_rate(self) -> float:
        """
        Get the pixel size (sampling rate) of the particles.

        Returns:
            Pixel size in Angstroms per pixel
        """
        return self.optics_md["rlnImagePixelSize"].unique().item()

    @staticmethod
    def createEmptySet():
        """
        Create an empty ParticlesStarSet instance.

        Returns:
            Empty ParticlesStarSet instance
        """
        pset = ParticlesStarSet.__new__(ParticlesStarSet)
        pset.emptySet()
        return pset

    @staticmethod
    def createFromPdNp(newStarFname: str, opticsDf: pd.DataFrame, particlesDf: pd.DataFrame,
                       npImages: Union[np.ndarray, Iterator[np.ndarray]],
                       overwrite: bool = False, basenameInStarImageName: bool = True):
        """

        :param newStarFname: The starfile name to save the results. The stack of particles will have the same name  as newStarFname but ending with .mrcs
        :param opticsDf: The optics datafile
        :param particlesDf: The particles metadata as a pandas dataframe
        :param npImages: The images, as a numpy array of shape (Nx3) to store or as an iterator of np images.
        :param overwrite: True if the files are to be overwritten.
        :param basenameInStarImageName: True if the mrcs filenames are going to be stored as basenames
        :return:
        """

        # assert "rlnImageName" in particlesDf, ("rlnImageName will be automatically assigned when the stack is created, "
        #                                        "it needs to be removed from particlesDf")

        newStackName = osp.splitext(newStarFname)[0] + ".mrcs"
        _newStackName = newStackName if not basenameInStarImageName else osp.basename(newStackName)
        particlesDf["rlnImageName"] = ["%06d@%s" % (i, _newStackName) for i in range(1, 1 + len(particlesDf))]

        if isinstance(npImages, np.ndarray):
            assert npImages.shape[-1] == opticsDf["rlnImageSize"].unique().item(), \
                "Error, image size mismatch between images and metadata"
            assert npImages.shape[0] == particlesDf.shape[0], "Error, mismatch in the number of particles"
            MrcFileStack.dump_npImages(newStackName, npImages,
                                       sampling_rate=opticsDf["rlnImagePixelSize"].unique().item(),
                                       overwrite=overwrite)
        else:
            MrcFileStack.dump_npImages_from_iterator(newStackName, npImages, nParticles=len(particlesDf),
                                                     particle_shape=(opticsDf["rlnImageSize"].unique().item(),) * 2,
                                                     sampling_rate=opticsDf["rlnImagePixelSize"].unique().item(),
                                                     overwrite=overwrite, batch_size=1000)

        starfile.write(dict(optics=opticsDf, particles=particlesDf), newStarFname, overwrite=overwrite)
        return ParticlesStarSet(newStarFname)

    def createSubset(self, start: Optional[int] = None, end: Optional[int] = None,
                     idxs: Optional[List[int]] = None) -> T: #TODO: Enable ids for subseters
        """
        Create a new ParticlesStarSet containing a subset of particles.

        Args:
            start: Starting index for slicing
            end: Ending index for slicing
            idxs: List of specific indices to include. Only start and end, or idxs can be provided

        Returns:
            New ParticlesStarSet containing only the specified particles

        Raises:
            AssertionError: If both slice indices and specific indices are provided
        """

        pset = ParticlesStarSet.createEmptySet()
        pset.optics_md = self.optics_md.copy()
        if idxs is not None:
            assert start is None and end is None, "Error, if idxs are provided, start and end should not be provided"
            pset.particles_md = self.particles_md.iloc[idxs].copy()
            pset.partNum_fname = [self.partNum_fname[i] for i in idxs]
        else:
            assert start is not None and end is not None, "Error, if idxs are not provided, start and end should be provided"
            pset.particles_md = self.particles_md.iloc[start:end, :].copy()
            pset.partNum_fname = self.partNum_fname[start:end]
        pset._imgStackFileHandlers = {}
        pset.starFname = None
        pset.particlesDir = self.particlesDir
        pset.starts_at_1 = self.starts_at_1
        return pset

    def copy(self):
        return self.createSubset(start=0, end=len(self))

    def getImgStackFilehandler(self, fname):
        if fname not in self._imgStackFileHandlers:
            self._imgStackFileHandlers[fname] = MrcFileStack(fname)
        return self._imgStackFileHandlers[fname]

    def save(self, starFname: str, stackFname: Optional[str] = None, overwrite: bool = False, batch_size: int = 1000,
             basenameOnlyForNewStar: bool = True):

        # Create a copy of the particles metadata
        new_particles_md = self.particles_md.copy()

        # If stackFname is provided, copy the particles to the new stack and update the image names
        if stackFname:
            # Determine the shape of the particles

            MrcFileStack.dump_npImages_from_iterator(stackFname, (p for p, md in self),
                                                     nParticles=len(self.partNum_fname),
                                                     particle_shape=self.particle_shape,
                                                     sampling_rate=self.optics_md["rlnImagePixelSize"].iloc[0],
                                                     overwrite=overwrite, batch_size=batch_size)

            # Update the image names in the new particles metadata to refer to the new stack file
            if basenameOnlyForNewStar:
                stackFname = osp.basename(stackFname)
            new_particles_md["rlnImageName"] = ["%06d@%s" % (i + 1, stackFname) for i in range(len(new_particles_md))]

        # Create the new starfile
        star_data = {}
        if self.optics_md is not None:
            star_data["optics"] = self.optics_md
        star_data["particles"] = new_particles_md

        starfile.write(star_data, starFname, overwrite=True)

    def updateMd(self, *, ids: Optional[List[str]]=None, idxs: Optional[List[int]]=None,
                 colname2change: Dict[str, Any]=None):
        assert colname2change is not None
        for col, vals in colname2change.items():
            if ids is not None:
                self.particles_md.loc[ids, col] = vals
            elif idxs is not None:
                try:
                    col_idx =  self.particles_md.columns.get_loc(col)
                    self.particles_md.iloc[idxs, col_idx] = vals
                except KeyError:
                    self.particles_md[col] = np.nan * np.ones(self.particles_md.shape[0], dtype=vals.dtype)
                    self.particles_md.iloc[idxs, self.particles_md.shape[-1]-1] = vals
            else:
                self.particles_md[col] = vals

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            imgs, mds = zip(*[self.__getitem(ii) for ii in range(*idx.indices(len(self)))])
            return np.stack(imgs, 0), pd.DataFrame(mds)
        elif isinstance(idx, (list, tuple)):
            imgs, mds = zip(*[self.__getitem(ii) for ii in idx])
            return np.stack(imgs, 0), pd.DataFrame(mds)
        else:
            return self.__getitem(idx)

    def __getitem(self, idx):
        partNum, fname = self.partNum_fname[idx]
        return self.getImgStackFilehandler(fname)[partNum], self.particles_md.iloc[idx, :]

    def __len__(self):
        return len(self.particles_md)

    def getPose(self, idx: Union[int,List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the pose (Euler angles and shifts) for specified particle(s).

        Args:
            idx: Single index or list of indices of particles

        Returns:
            Tuple containing:
                - np.ndarray: Euler angles in degrees
                - np.ndarray: X/Y shifts in Angstroms
        """
        md = self.particles_md.iloc[idx, :]
        return self.getPoseFromMd(md)

    def setPose(self, idx: Union[int,List[int]], eulerDegs: Optional[np.ndarray] = None, shiftsAngst: Optional[np.ndarray] = None,
                confidence: Optional[np.ndarray] = None):
        """
        Set the pose parameters for specified particle(s).

        Args:
            idx: Single index or list of indices of particles to update
            eulerDegs: Euler angles in degrees (rot, tilt, psi)
            shiftsAngst: X/Y shifts in Angstroms
            confidence: Pose confidence scores
        """

        if eulerDegs is not None:
            self.particles_md.iloc[idx,
                                [self.particles_md.columns.get_loc(col) for col in RELION_ANGLES_NAMES]] = eulerDegs

        if shiftsAngst is not None:
            self.particles_md.iloc[idx,
                                [self.particles_md.columns.get_loc(col) for col in RELION_SHIFTS_NAMES]] = shiftsAngst

        if confidence is not None:
            if RELION_PRED_POSE_CONFIDENCE_NAME not in self.particles_md.columns:
                self.particles_md[RELION_PRED_POSE_CONFIDENCE_NAME] = -1

            self.particles_md.iloc[idx,
                                self.particles_md.columns.get_loc(RELION_PRED_POSE_CONFIDENCE_NAME)] = confidence


    @classmethod
    def getPoseFromMd(cls, md):
        anglesDegs = [md[name] for name in RELION_ANGLES_NAMES]
        xyShiftAngs = [md[name] for name in RELION_SHIFTS_NAMES]
        anglesDegs = np.array(anglesDegs).T
        xyShiftAngs = np.array(xyShiftAngs).T
        return anglesDegs, xyShiftAngs

    def getCtfParamsFromMd(self, md):  # TODO: Move the names to constants
        if isinstance(md, pd.Series):
            md = pd.DataFrame([md])
        elif not isinstance(md, pd.DataFrame):
            raise RuntimeError("Error, md is not a pd.DataFrame")
        voltage = self.optics_md.rlnVoltage
        amplitude_contrast = self.optics_md.rlnAmplitudeContrast
        spherical_aberration = self.optics_md.rlnSphericalAberration
        sampling_rate = self.optics_md.rlnImagePixelSize
        defocus_u = md.rlnDefocusU
        defocus_v = md.rlnDefocusV
        defocus_angle = md.rlnDefocusAngle
        params = dict(defocus_u=defocus_u, defocus_v=defocus_v, defocus_angle=defocus_angle, voltage=voltage,
                      sampling_rate=sampling_rate, amplitude_contrast=amplitude_contrast,
                      spherical_aberration=spherical_aberration)
        return {k: v.values for k, v in params.items()}

    def __add__(self, other):
        ps = self.createEmptySet()
        ps.particles_md = pd.concat([self.particles_md, other.particles_md])
        ps.partNum_fname += other.partNum_fname
        ps._imgStackFileHandlers.update(other._imgStackFileHandlers)

        # Combine optics metadata if available
        if self.optics_md is not None and other.optics_md is not None:
            # Determine the next available optics group ID
            next_optics_group_id = self.optics_md['rlnOpticsGroup'].max() + 1

            # Update the optics group IDs in the other optics DataFrame
            other_optics_md = other.optics_md.copy()
            other_optics_md['rlnOpticsGroup'] = next_optics_group_id

            suffix = f"_{next_optics_group_id}"
            other_optics_md['rlnOpticsGroupName'] = other_optics_md['rlnOpticsGroupName'].apply(lambda x: x + suffix)
            # Combine the optics DataFrames
            combined_optics = pd.concat([self.optics_md, other_optics_md])
            combined_optics.drop_duplicates(ignore_index=True, inplace=True)
            ps.optics_md = combined_optics
        return ps

    def _group_particles_by_file(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Group particles by their MRC file and organize them for efficient reading.
        Orders particles within each file by their MRC index for contiguous reads.
        """
        file_groups = defaultdict(list)
        for particle_idx, (mrc_idx, fname) in enumerate(self.partNum_fname):
            file_groups[fname].append((particle_idx, mrc_idx))

        return {
            fname: sorted(indices, key=lambda x: x[1])
            for fname, indices in file_groups.items()
        }

    def _read_block(self, fname: str, particle_indices: List[Tuple[int, int]]) -> Tuple[List[int], np.ndarray]:
        """Read a block of particles from a specific file."""
        mrc_handler = self.getImgStackFilehandler(fname)
        particle_idxs, mrc_idxs = zip(*particle_indices)
        images = mrc_handler.get_block(mrc_idxs)
        return particle_idxs, images

    def ori_iter_particles(self,
                       block_size: int = 128,
                       num_threads: int = 1,
                       queue_size: int = 4
                       ) -> Iterator[Tuple[np.ndarray, pd.Series]]:
        """
        Iterate over particles using a fixed worker pool for efficient reading.

        Args:
            block_size: Number of particles per block
            num_threads: Number of worker threads in the pool
            queue_size: Size of the results queue

        Yields:
            Tuple[np.ndarray, pd.Series]: (particle_image, particle_metadata)
        """
        file_groups = self._group_particles_by_file()

        # Create work and results queues
        work_queue: Queue = Queue()
        results_queue: Queue = Queue(maxsize=queue_size)

        def worker():
            """Worker function that processes blocks from any file"""
            while True:
                try:
                    task = work_queue.get()
                    if task is None:  # Shutdown signal
                        work_queue.task_done()
                        break

                    block_id, fname, indices = task
                    try:
                        particle_idxs, images = self._read_block(fname, indices)
                        results_queue.put((block_id, particle_idxs, images))
                    except Exception as e:
                        results_queue.put((block_id, "error", e))
                    work_queue.task_done()
                except:  # Catch all exceptions to ensure proper shutdown
                    break

        # Start worker threads
        workers = []
        active_workers = num_threads
        for _ in range(num_threads):
            t = Thread(target=worker, daemon=True)
            t.start()
            workers.append(t)

        try:
            # Queue all work
            block_id = 0
            for fname, particle_indices in file_groups.items():
                for start_idx in range(0, len(particle_indices), block_size):
                    end_idx = min(start_idx + block_size, len(particle_indices))
                    block = particle_indices[start_idx:end_idx]
                    work_queue.put((block_id, fname, block))
                    block_id += 1

            # Process results in order
            blocks_to_process = block_id
            next_block_id = 0
            pending_results = {}

            while blocks_to_process > 0:
                if active_workers == 0:
                    raise RuntimeError("All workers died")

                try:
                    block_id, particle_idxs, result = results_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue

                if isinstance(particle_idxs, str) and particle_idxs == "error":
                    raise result

                if block_id == next_block_id:
                    # Yield this block's particles
                    for i, particle_idx in enumerate(particle_idxs):
                        yield result[i], self.particles_md.iloc[particle_idx]
                    next_block_id += 1
                    blocks_to_process -= 1

                    # Process any pending blocks that are now ready
                    while next_block_id in pending_results:
                        p_idxs, p_images = pending_results.pop(next_block_id)
                        for i, particle_idx in enumerate(p_idxs):
                            yield p_images[i], self.particles_md.iloc[particle_idx]
                        next_block_id += 1
                        blocks_to_process -= 1
                else:
                    # Store out-of-order results
                    pending_results[block_id] = (particle_idxs, result)

        except Exception as e:
            # Make sure we clean up on any error
            raise e

        finally:
            # Shutdown workers
            for _ in range(num_threads):
                work_queue.put(None)

            # Wait for workers with timeout
            for w in workers:
                w.join(timeout=1.0)


    def iter_particles_batch(self,
                             batch_size: int = 32,
                             block_size: Optional[int] = None,
                             num_threads: int = 1,
                             queue_size: int = 4,
                             shuffle_batch: bool = False
                             ) -> Iterator[Tuple[np.ndarray, pd.DataFrame]]:
        """
        Iterate over particles in batches using a fixed worker pool for efficient reading.

        Args:
            batch_size: Number of particles per batch returned
            block_size: Number of particles to read in each block (must be >= batch_size)
                       If None, defaults to batch_size * 4
            num_threads: Number of worker threads in the pool
            queue_size: Size of the results queue
            shuffle_batch: Whether to shuffle particles within each batch

        Yields:
            Tuple[np.ndarray, pd.DataFrame]: (batch_images, batch_metadata)
        """
        if block_size is None:
            block_size = batch_size * 4

        if block_size < batch_size:
            raise ValueError(f"block_size ({block_size}) must be >= batch_size ({batch_size})")

        file_groups = self._group_particles_by_file()

        # Create work and results queues
        work_queue: Queue = Queue()
        results_queue: Queue = Queue(maxsize=queue_size)

        def worker():
            """Worker function that processes blocks from any file"""
            while True:
                try:
                    task = work_queue.get()
                    if task is None:  # Shutdown signal
                        work_queue.task_done()
                        break

                    block_id, fname, indices = task
                    try:
                        particle_idxs, images = self._read_block(fname, indices)
                        results_queue.put((block_id, particle_idxs, images))
                    except Exception as e:
                        results_queue.put((block_id, "error", e))
                    work_queue.task_done()
                except:  # Catch all exceptions to ensure proper shutdown
                    break

        # Start worker threads
        workers = []
        active_workers = num_threads
        for _ in range(num_threads):
            t = Thread(target=worker, daemon=True)
            t.start()
            workers.append(t)

        try:
            # Queue all work
            block_id = 0
            for fname, particle_indices in file_groups.items():
                for start_idx in range(0, len(particle_indices), block_size):
                    end_idx = min(start_idx + block_size, len(particle_indices))
                    block = particle_indices[start_idx:end_idx]
                    work_queue.put((block_id, fname, block))
                    block_id += 1

            # Process results in order
            blocks_to_process = block_id
            next_block_id = 0
            pending_results = {}
            current_batch_images = []
            current_batch_metadata = []

            while blocks_to_process > 0:
                if active_workers == 0:
                    raise RuntimeError("All workers died")

                try:
                    block_id, particle_idxs, result = results_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue

                if isinstance(particle_idxs, str) and particle_idxs == "error":
                    raise result

                if block_id == next_block_id:
                    # Process this block's particles
                    for i, particle_idx in enumerate(particle_idxs):
                        current_batch_images.append(result[i])
                        current_batch_metadata.append(self.particles_md.iloc[particle_idx])

                        if len(current_batch_images) == batch_size:
                            batch_array = np.stack(current_batch_images)
                            batch_md = pd.DataFrame(current_batch_metadata)

                            if shuffle_batch:
                                shuffle_idx = np.random.permutation(batch_size)
                                batch_array = batch_array[shuffle_idx]
                                batch_md = batch_md.iloc[shuffle_idx]

                            yield batch_array, batch_md
                            current_batch_images = []
                            current_batch_metadata = []

                    next_block_id += 1
                    blocks_to_process -= 1

                    # Process any pending blocks that are now ready
                    while next_block_id in pending_results:
                        p_idxs, p_images = pending_results.pop(next_block_id)
                        for i, particle_idx in enumerate(p_idxs):
                            current_batch_images.append(p_images[i])
                            current_batch_metadata.append(self.particles_md.iloc[particle_idx])

                            if len(current_batch_images) == batch_size:
                                batch_array = np.stack(current_batch_images)
                                batch_md = pd.DataFrame(current_batch_metadata)

                                if shuffle_batch:
                                    shuffle_idx = np.random.permutation(batch_size)
                                    batch_array = batch_array[shuffle_idx]
                                    batch_md = batch_md.iloc[shuffle_idx]

                                yield batch_array, batch_md
                                current_batch_images = []
                                current_batch_metadata = []

                        next_block_id += 1
                        blocks_to_process -= 1
                else:
                    # Store out-of-order results
                    pending_results[block_id] = (particle_idxs, result)

            # Yield remaining particles if any
            if current_batch_images:
                batch_array = np.stack(current_batch_images)
                batch_md = pd.DataFrame(current_batch_metadata)

                if shuffle_batch and len(current_batch_images) > 1:
                    shuffle_idx = np.random.permutation(len(current_batch_images))
                    batch_array = batch_array[shuffle_idx]
                    batch_md = batch_md.iloc[shuffle_idx]

                yield batch_array, batch_md

        finally:
            # Shutdown workers
            for _ in range(num_threads):
                work_queue.put(None)

            # Wait for workers with timeout
            for w in workers:
                w.join(timeout=1.0)


    def iter_particles(self,
                       block_size: int = 128,
                       num_threads: int = 1,
                       queue_size: int = 4
                       ) -> Iterator[Tuple[np.ndarray, pd.Series]]:
        """
        Iterate over individual particles.

        This is implemented as a wrapper over iter_particles_batch for consistency.

        Args:
            block_size: Size of internal batches for reading
            num_threads: Number of worker threads
            queue_size: Size of the results queue

        Yields:
            Tuple[np.ndarray, pd.Series]: (particle_image, particle_metadata)
        """
        for batch_images, batch_metadata in self.iter_particles_batch(
                batch_size=block_size,
                num_threads=num_threads,
                queue_size=queue_size,
                shuffle_batch=False
        ):
            for i in range(len(batch_images)):
                yield batch_images[i], batch_metadata.iloc[i]


def split_particle_and_fname(fname, pattern=re.compile(r"(\d+@)?(.*/)*(.*)")):
    matchObj = re.match(pattern, fname)
    return dict(partNum=matchObj.group(1)[:-1], dirname=matchObj.group(2), basename=matchObj.group(3))