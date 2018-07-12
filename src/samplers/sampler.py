#!/usr/bin/env python3
__doc__ = """

Sampling class for joint synapse and vesicle training

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""
import os

import dataprovider as dp
import h5py
import tifffile as tif
import numpy as np


def read_file(fname):

    f = h5py.File(fname)
    d = f["/main"].value
    f.close()

    return d


class Sampler(object):

    def __init__(self, datadir, dsets=[], mode="train", patchsz=(10, 128, 128),
                 inputsz=(20, 160, 160)): # (18, 160, 160)

        assert mode in ["train", "val", "test"]

        datadir = os.path.expanduser(datadir)

        volnames = ["input", "soma_label"]
        # spec = {name: patchsz for name in volnames}
        spec = {name: inputsz for name in volnames}

        self.dp = self.build_data_provider(datadir, spec, mode, dsets)

    def __call__(self, **kwargs):
        return self.dp("random", **kwargs)

    def build_data_provider(self, datadir, spec, mode, dsets):

        vdp = dp.VolumeDataProvider()

        dsets = ["Z25_Y12_3250-3349",
                 "Z25_Y13_2800-2899",
                 "Z19_Y07_2200-2299",
                 "Z15_Y13_1900-1999"]
        # dsets = ["Z25_Y12_3250-3349"]

        # dsets.append(self.build_dataset(datadir, spec, "Z25_Y12_3250-3349"))
        # dsets.append(self.build_dataset(datadir, spec, "Z25_Y13_2800-2899"))
        # dsets.append(self.build_dataset(datadir, spec, "Z19_Y07_2200-2299"))
        # dsets.append(self.build_dataset(datadir, spec, "Z15_Y13_1900-1999"))

        for dset in dsets:
            vdp.add_dataset(self.build_dataset(datadir, spec, dset))

        vdp.set_sampling_weights()

        vdp.set_augmentor(self._aug(mode))
        vdp.set_postprocessor(self._post())

        return vdp

    def build_dataset(self, datadir, spec, dset_name):

        print(dset_name)
        # img = tif.imread("/data/gornet/unet-training/GADH2DGFP_substack.tif")
        # lbl = tif.imread("/data/gornet/unet-training/GADH2DGFP_cell_markup_subvolume.tif").astype("float32")

        # img = read_file(os.path.join(datadir, dset_name + "_raw.h5"))
        # lbl = read_file(os.path.join(datadir, dset_name + "_labels.h5")).astype("float32")

        img = read_file(os.path.join(datadir, dset_name + ".h5"))
        lbl = read_file(os.path.join(datadir, dset_name + "-segmentation.h5")).astype("float32")

        img = dp.transform.divideby(img, val=np.max(img), dtype="float32")
        lbl[lbl != 0] = 1

        vd = dp.VolumeDataset()

        vd.add_raw_data(key="input", data=img)
        vd.add_raw_data(key="soma_label", data=lbl)

        # img = read_file(os.path.join(datadir, "Z25_Y13_2800-2899.h5"))
        # lbl = read_file(os.path.join(
        #     datadir, "Z25_Y13_2800-2899-segmentation.h5")).astype("float32")

        # img = dp.transform.divideby(img, val=np.max(img), dtype="float32")
        # lbl[lbl != 0] = 1

        # vd.add_raw_data(key="input",       data=img)
        # vd.add_raw_data(key="soma_label",  data=lbl)

        # img = read_file(os.path.join(datadir, "Z19_Y07_2200-2299.h5"))
        # lbl = read_file(os.path.join(
        #     datadir, "Z19_Y07_2200-2299-segmentation.h5")).astype("float32")

        # img = dp.transform.divideby(img, val=np.max(img), dtype="float32")
        # lbl[lbl != 0] = 1

        # vd.add_raw_data(key="input",       data=img)
        # vd.add_raw_data(key="soma_label",  data=lbl)

        vd.set_spec(spec)
        return vd

    def _aug(self, mode):

        aug = dp.Augmentor()
        if mode == "train":
            aug.append('occlusion')
            aug.append('duplicate')
            aug.append('stitch')
            aug.append('brightness')
            aug.append('crop', size=(10, 128, 128))

        return aug

    def _post(self):
        post = dp.Transformer()
        return post
