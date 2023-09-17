import re

import numpy as np
import starfile
import pandas as pd
import os.path as osp
from typing import Optional, Union, Iterator, List, Dict, Any
from .mrcFileStack import MrcFileStack


class ParticlesStarSet():

    def __init__(self, starFname, particlesDir: Optional[str]=None, starts_at_1:bool=True):
        """

        :param starFname: The star filename with the metadata
        :param particlesDir: The directory of the particles .mrcs file(s). If None, it assumes that starFname
        imageName paths for .mrcs are referred from the current working directory
        :param starts_at_1: Whether the first particle at the starfile is named 0 or 1. In Relion it starts with 1
        """
        self.starFname = starFname
        self.particlesDir = particlesDir
        self.starts_at_1 = starts_at_1
        self.read(starFname, particlesDir, starts_at_1=starts_at_1)

    def read(self, starFname, particlesDir: Optional[str], starts_at_1:bool=True):
        """

        :param starFname: The star filename with the metadata
        :param particlesDir: The directory of the particles .mrcs file(s). If None, it assumes that starFname
        imageName paths for .mrcs are refered from the current working directory
        :param starts_at_1: Whether the first particle at the starfile is named 0 or 1. In Relion it starts with 1
        """

        data = starfile.read(starFname)
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
            if not osp.isfile(fullFname):
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
        self.particles_md = None
        self.optics_md = None
        self.partNum_fname = []
        self._imgStackFileHandlers = {}

    @property
    def particle_shape(self):
        partNum, fname = self.partNum_fname[0]
        img_stack_handler = self.getImgStackFilehandler(fname)
        particle_shape = img_stack_handler[partNum].shape
        return particle_shape

    @property
    def sampling_rate(self):
        return self.optics_md["rlnImagePixelSize"].unique().item()
    @staticmethod
    def createEmptySet():
        pset = ParticlesStarSet.__new__(ParticlesStarSet)
        pset.emptySet()
        return pset

    @staticmethod
    def createFromPdNp(newStarFname:str, opticsDf:pd.DataFrame, particlesDf:pd.DataFrame,
                       npImages:Union[np.ndarray, Iterator[np.ndarray]],
                       overwrite:bool=False, basenameInStarImageName:bool=True):
        """

        :param newStarFname: The starfile name to save the results. The stack of particles will have the same name  as newStarFname but ending with .mrcs
        :param opticsDf: The optics datafile
        :param particlesDf: The particles metadata as a pandas dataframe
        :param npImages: The images, as a numpy array of shape (Nx3) to store or as an iterator of np images.
        :param overwrite: True if the files are to be overwritten.
        :param basenameInStarImageName: True if the mrcs filenames are going to be stored as basenames
        :return:
        """


        assert "rlnImageName" in particlesDf, ("rlnImageName will be automatically assigned when the stack is created, "
                                             "it need to be removed from starDf")

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
                                                particle_shape=(opticsDf["rlnImageSize"].unique().item(),)*2,
                                                sampling_rate=opticsDf["rlnImagePixelSize"].unique().item(),
                                                overwrite=overwrite, batch_size=1000)

        starfile.write(dict(optics=opticsDf, particles=particlesDf), newStarFname, overwrite=overwrite)
        return ParticlesStarSet(newStarFname)

    def createSubset(self, start:Optional[int]=None, end:Optional[int]=None, idxs:Optional[List[int]]=None):

        pset = ParticlesStarSet.createEmptySet()
        pset.optics_md = self.optics_md.copy()
        if idxs is not None:
            assert start is  None and end is None, "Error, if idxs are provided, start and end should not be provided"
            pset.particles_md = self.particles_md.iloc[idxs]
            pset.partNum_fname = [self.partNum_fname[i] for i in idxs]
        else:
            assert start is not None and end is not None, "Error, if idxs are not provided, start and end should be provided"
            pset.particles_md = self.particles_md.iloc[start:end, :]
            pset.partNum_fname = self.partNum_fname[start:end]
        pset._imgStackFileHandlers = {}
        return pset

    def getImgStackFilehandler(self, fname):
        if fname not in self._imgStackFileHandlers:
            self._imgStackFileHandlers[fname] = MrcFileStack(fname)
        return self._imgStackFileHandlers[fname]


    def save(self, starFname: str, stackFname: Optional[str] = None, overwrite:bool=False, batch_size:int=1000,
             basenameOnlyForNewStar:bool=True):

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
            new_particles_md["rlnImageName"] = ["%6d@%s" % (i + 1, stackFname) for i in range(len(new_particles_md))]

        # Create the new starfile
        star_data = {}
        if self.optics_md is not None:
            star_data["optics"] = self.optics_md
        star_data["particles"] = new_particles_md


        starfile.write(star_data, starFname, overwrite=True)

    def updateMd(self, ids:List[str], colname2change:Dict[str, Any]):

        for col, vals in colname2change.items():
            self.particles_md.loc[ids, col] = vals

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            img_md_list = [self.__getitem(ii) for ii in range(*idx.indices(len(self)))]
            return img_md_list
        else:
            return self.__getitem(idx)

    def __getitem(self, idx):
        partNum, fname = self.partNum_fname[idx]
        return self.getImgStackFilehandler(fname)[partNum], self.particles_md.iloc[idx, :]

    def __len__(self):
        return len(self.particles_md)

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
            other_optics_md['rlnOpticsGroup'] += next_optics_group_id

            suffix = f"_{next_optics_group_id}"
            other_optics_md['rlnOpticsGroupName'] = other_optics_md['rlnOpticsGroupName'].apply(lambda x: x + suffix)
            # Combine the optics DataFrames
            combined_optics = pd.concat([self.optics_md, other_optics_md])
            combined_optics.drop_duplicates(ignore_index=True, inplace=True)
            ps.optics_md = combined_optics
        return ps



def split_particle_and_fname(fname, pattern = re.compile("(\d+@)?(.*/)*(.*)")):
    matchObj = re.match(pattern, fname)
    return dict(partNum=matchObj.group(1)[:-1], dirname=matchObj.group(2), basename=matchObj.group(3))