## starstack: CryoEM Particle stars and stack Management in Python


#O verview
The ParticlesStarSet class provides an efficient way to manage a set of particles in cryo-electron microscopy (cryoEM). 
It allows for easy reading and manipulation of STAR files and associated .mrcs image stacks. 
The library also offers utilities for creating subsets and saving them back to disk.

# Features

Read STAR files and associated .mrcs particle stacks.
Create new STAR files and .mrcs particle stacks.
Support for creating subsets of particles.
Support for particle stack file handling.

# Dependencies
mrcfile
numpy
pandas
starfile

# Installation
pip install git+https://github.com/rsanchezgarc/starstack

# Usage
Reading STAR Files

```
from starstack import ParticlesStarSet
pset = ParticlesStarSet("my_file.star", particlesDir="Particles/")
```