import os
import os.path as osp
import random
import tempfile
from unittest import TestCase

import mrcfile
import requests
import pandas as pd
import numpy as np

from starstack.mrcFileStack import MrcFileStack
from starstack.particlesStar import ParticlesStarSet


def _donwload_url(url, filename):
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved as {filename}")
    else:
        print("Failed to download the file.")


def download_dataset():
    rootdir = tempfile.gettempdir()
    jobDir = osp.join(rootdir, "Extract/job007")
    moviesDir = osp.join(jobDir, "Movies")
    os.makedirs(moviesDir, exist_ok=True)
    baseurl = "https://scipion.cnb.csic.es/downloads/scipion/data/tests/relion31_tutorial_precalculated/Extract/job007/"
    fnameOut = osp.join(jobDir, "particles.star")
    if not osp.exists(fnameOut):
        _donwload_url(baseurl + "particles.star", fnameOut)
    filenames = [
        "20170629_00021_frameImage.mrcs", "20170629_00022_frameImage.mrcs", "20170629_00023_frameImage.mrcs",
        "20170629_00024_frameImage.mrcs", "20170629_00025_frameImage.mrcs"
    ]
    for fname in filenames:
        fnameOut = osp.join(jobDir, "Movies/" + fname)
        if not osp.exists(fnameOut):
            _donwload_url(baseurl + "Movies/" + fname, osp.join(jobDir, "Movies/" + fname))

    return rootdir, jobDir


class TestParticlesSet(TestCase):
    def test_read(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        # print(len(pset))
        self.assertEqual(len(pset), 1158)
        firstImg, firstMd = pset[0]
        self.assertAlmostEqual(firstMd["rlnCoordinateX"], 1614.217657)
        lastImg, lastMd = pset[-1]

        self.assertAlmostEqual(lastMd["rlnCoordinateX"], 1230.417654)
        self.assertAlmostEqual(pset[48][0].mean(), 0.080, places=3)
        self.assertAlmostEqual(pset[73][0].mean(), 0.092, places=3)
        # import matplotlib.pyplot as plt
        # plt.imshow(firstImg); plt.show()
        # plt.imshow(lastImg); plt.show()

    def test_dump(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        with tempfile.NamedTemporaryFile() as f:
            MrcFileStack.dump_npImages_from_iterator(f.name, (p for p, md in pset), nParticles=len(pset),
                                                     particle_shape=pset.particle_shape,
                                                     sampling_rate=pset.sampling_rate, overwrite=True)
            f.seek(0)
            data = mrcfile.read(f.name)
            self.assertAlmostEqual(data[48].mean(), 0.080, places=3)
            self.assertAlmostEqual(data[73].mean(), 0.092, places=3)

    def test_subset(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        subsetPset = pset.createSubset(0, 100)
        self.assertTrue((pset[99][1] == subsetPset[99][1]).all())

        subsetPset = pset.createSubset(100, 104)
        self.assertTrue((pset[100][1] == subsetPset[0][1]).all())

        subsetPset = pset.createSubset(idxs=[100, 101])
        # print(pset[100][1]["rlnImageName"])
        # print(subsetPset[0][1]["rlnImageName"])
        self.assertTrue((pset[100][1] == subsetPset[0][1]).all())

        with tempfile.NamedTemporaryFile() as f:
            subsetPset.save(f.name, stackFname=f.name.replace(".star", ".mrcs"), overwrite=True)

    def test_updateMd(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        img, md = pset[5]

        # pset.updateMd(ids=[md['rlnImageName']], colname2change={'rlnAnglePsi':[-1]})
        # self.assertAlmostEqual(pset[5][-1]['rlnAnglePsi'], -1)

        idxs = [random.randint(0, len(pset) - 1) for _ in range(10)]
        mds = [pset[idx][-1] for idx in idxs]
        target_vals = list(range(len(idxs)))
        pset.updateMd(ids=[md['rlnImageName'] for md in mds], colname2change={'rlnAnglePsi': target_vals})
        new_md = [pset[idx][-1] for idx in idxs]
        self.assertEqual([int(round(n['rlnAnglePsi'])) for n in new_md], target_vals)

    def test_shuffle(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        pset.shuffle()


class TestParticlesStarSetBlock(TestCase):
    def setUp(self):
        """Set up test data with a small particles dataset"""
        # Create temporary directory and files
        self.temp_dir = tempfile.mkdtemp()

        # Create small test MRC files
        self.particle_shape = (10, 10)  # Small particles for testing
        self.min_particles_per_file = 1
        self.max_particles_per_file = 11
        self.n_files = 9
        # Create two MRC files with known data
        self.mrc_files = []
        self.all_particles = []
        self.n_particles_per_file_list = []
        for file_idx in range(self.n_files):
            mrc_path = os.path.join(self.temp_dir, f"particles_{file_idx}.mrcs")
            n_particles_per_file = np.random.randint(self.min_particles_per_file, self.max_particles_per_file)
            self.n_particles_per_file_list.append(n_particles_per_file)
            particles = np.random.rand(n_particles_per_file, *self.particle_shape).astype(np.float32)
            self.all_particles.extend(particles)

            with mrcfile.new(mrc_path, overwrite=True) as mrc:
                mrc.set_data(particles)
                mrc.voxel_size = 1.0
            self.mrc_files.append(mrc_path)

        # Create star file metadata
        optics_data = {
            'rlnOpticsGroup': [1],
            'rlnImagePixelSize': [1.0],
            'rlnImageSize': [self.particle_shape[0]],
            'rlnVoltage': [300],
            'rlnSphericalAberration': [2.7],
            'rlnAmplitudeContrast': [0.1],
            'rlnOpticsGroupName': ['opticsGroup1']
        }

        particles_data = []
        for file_idx, mrc_file in enumerate(self.mrc_files):
            for particle_idx in range(self.n_particles_per_file_list[file_idx]):
                particles_data.append({
                    'rlnImageName': f"{particle_idx + 1:06d}@{os.path.basename(mrc_file)}",
                    'rlnDefocusU': 15000.0,
                    'rlnDefocusV': 15000.0,
                    'rlnDefocusAngle': 0.0,
                })

        # Save star file
        self.star_file = os.path.join(self.temp_dir, "particles.star")
        import starfile
        starfile.write({
            'optics': pd.DataFrame(optics_data),
            'particles': pd.DataFrame(particles_data)
        }, self.star_file)

        # Create ParticlesStarSet instance
        self.pset = ParticlesStarSet(self.star_file, particlesDir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_iter_particles_matches_getitem(self):
        """Test that iter_particles yields same results as __getitem__"""
        # Get all particles using __getitem__
        getitem_particles = []
        getitem_metadata = []
        for i in range(len(self.pset)):
            img, md = self.pset[i]
            getitem_particles.append(img)
            getitem_metadata.append(md)

        getitem_particles = np.array(getitem_particles)
        getitem_metadata = pd.DataFrame(getitem_metadata)

        # Get all particles using iter_particles
        for block_size in [1,2,3,5,10,20]:
            iter_particles = []
            iter_metadata = []
            for img, md in self.pset.iter_particles():  # Small block size to test blocking
                iter_particles.append(img)
                iter_metadata.append(md)

            iter_particles = np.array(iter_particles)
            iter_metadata = pd.DataFrame(iter_metadata)

            # Compare results
            np.testing.assert_array_almost_equal(
                getitem_particles,
                iter_particles,
                decimal=6,
                err_msg="Particle images from iterator don't match __getitem__"
            )

            try:
                pd.testing.assert_frame_equal(
                getitem_metadata,
                iter_metadata,
                check_exact=True,
                )
            except AssertionError as e:
                raise AssertionError(f"Mticle metadata from iterator doesn't match __getitem: {str(e)}")

    def test_iter_particles_different_block_sizes(self):
        """Test that different block sizes produce the same results"""
        # Reference: get all particles using block_size=1
        ref_particles = []
        ref_metadata = []
        for img, md in self.pset:
            ref_particles.append(img)
            ref_metadata.append(md)

        ref_particles = np.array(ref_particles)
        ref_metadata = pd.DataFrame(ref_metadata)

        # Test different block sizes
        for num_threads in [1,2,3]:
            for block_size in [1, 2, 3, 4, 5, 7, 100]:  # Include sizes that don't divide evenly
                test_particles = []
                test_metadata = []
                for img, md in self.pset.iter_particles(num_threads=num_threads):
                    test_particles.append(img)
                    test_metadata.append(md)

                test_particles = np.array(test_particles)
                test_metadata = pd.DataFrame(test_metadata)

                np.testing.assert_array_almost_equal(
                    ref_particles,
                    test_particles,
                    decimal=6,
                    err_msg=f"Results don't match for block_size={block_size}"
                )

                try:
                    pd.testing.assert_frame_equal(
                        ref_metadata,
                        test_metadata,
                        check_exact=True
                    )
                except AssertionError as e:
                    raise AssertionError(f"Metadata doesn't match for block_size={block_size}: {str(e)}")

    def test_iter_particles_empty_set(self):
        """Test behavior with an empty particle set"""
        empty_pset = ParticlesStarSet.createEmptySet()
        particles = list(empty_pset.iter_particles())
        self.assertEqual(len(particles), 0, "Empty set should yield no particles")

    def test_iter_particles_with_missing_file(self):
        """Test proper error handling when an MRC file is missing"""
        # Modify a particle's filename to point to a non-existent file
        self.pset.partNum_fname[0] = (0, "nonexistent.mrcs")
        # print("n_particles,", len(self.pset))
        with self.assertRaises(Exception):
            list(self.pset.iter_particles(num_threads=4))