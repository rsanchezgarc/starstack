import os
import os.path as osp
import random
import tempfile
from unittest import TestCase

import mrcfile
import requests

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
        _donwload_url(baseurl+"particles.star", fnameOut)
    filenames =[
        "20170629_00021_frameImage.mrcs", "20170629_00022_frameImage.mrcs", "20170629_00023_frameImage.mrcs",
        "20170629_00024_frameImage.mrcs", "20170629_00025_frameImage.mrcs"
    ]
    for fname in filenames:
        fnameOut = osp.join(jobDir, "Movies/"+fname)
        if not osp.exists(fnameOut):
            _donwload_url(baseurl+"Movies/"+fname, osp.join(jobDir, "Movies/"+fname))

    return rootdir, jobDir

class TestParticlesSet(TestCase):
    def test_read(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        print(len(pset))
        self.assertEqual(len(pset), 1158)
        firstImg, firstMd = pset[0]
        self.assertAlmostEqual(firstMd["rlnCoordinateX"], 1614.217657)
        lastImg, lastMd = pset[-1]

        self.assertAlmostEqual(lastMd["rlnCoordinateX"], 1230.417654)
        self.assertAlmostEqual(pset[48][0].mean(), 0.080, places=3)
        self.assertAlmostEqual(pset[73][0].mean(), 0.092, places=3)

    def test_dump(self):

        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        with tempfile.NamedTemporaryFile() as f:
            MrcFileStack.dump_npImages_from_iterator(f.name, (p for p,md in pset), nParticles=len(pset),
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
        subsetPset = pset.createSubset(0,100)
        self.assertTrue((pset[99][1] == subsetPset[99][1]).all())

        subsetPset = pset.createSubset(100,104)
        self.assertTrue((pset[100][1] == subsetPset[0][1]).all())

        subsetPset = pset.createSubset(idxs=[100,101])
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


        idxs = [random.randint(0, len(pset)-1) for _ in range(10)]
        mds = [pset[idx][-1] for idx in idxs]
        target_vals = list(range(len(idxs)))
        pset.updateMd(ids=[md['rlnImageName'] for md in mds], colname2change={'rlnAnglePsi':target_vals})
        new_md = [pset[idx][-1] for idx in idxs]
        self.assertEqual([int(round(n['rlnAnglePsi'])) for n in new_md], target_vals)


    def test_shuffle(self):
        rootdir, jobDir = download_dataset()
        os.listdir(jobDir)
        pset = ParticlesStarSet(starFname=osp.join(jobDir, "particles.star"), particlesDir=rootdir)
        pset.shuffle()