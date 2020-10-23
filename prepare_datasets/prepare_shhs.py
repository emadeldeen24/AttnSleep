import os
import numpy as np

import argparse
import glob
import math
import ntpath

import shutil
import urllib
# import urllib2

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import dhedfreader
import xml.etree.ElementTree as ET

###############################
EPOCH_SEC_SIZE = 30

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/abc/shhs/polysomnography/edfs/shhs1",
                        help="File path to the PSG files.")
    parser.add_argument("--ann_dir", type=str, default="/home/abc/shhs/polysomnography/annotations-events-profusion/shhs1",
                        help="File path to the annotation files.")
    parser.add_argument("--output_dir", type=str, default="/home/abc/output_npz/shhs",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-A1",
                        help="The selected channel")
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ids = pd.read_csv("selected_shhs1_files.txt", header=None, names='a')
    ids = ids['a'].values.tolist()

    edf_fnames = [os.path.join(args.data_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir,  i + "-profusion.xml") for i in ids]

    edf_fnames.sort()
    ann_fnames.sort()

    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)


    for file_id in range(len(edf_fnames)):
        if os.path.exists(os.path.join(args.output_dir, edf_fnames[file_id].split('/')[-1])[:-4]+".npz"):
            continue
        print(edf_fnames[file_id])

        raw = read_raw_edf(edf_fnames[file_id], preload=True, stim_channel=None, verbose=None)
        sampling_rate = raw.info['sfreq']
        ch_type = args.select_ch.split(" ")[0]
        select_ch = [s for s in raw.info["ch_names"] if ch_type in s][0]

        raw_ch_df = raw.to_data_frame(scaling_time=sampling_rate)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))



    ###################################################
        labels = []
        # Read annotation and its header
        t = ET.parse(ann_fnames[file_id])
        r = t.getroot()
        faulty_File = 0
        for i in range(len(r[4])):
            lbl = int(r[4][i].text)
            if lbl == 4:  # make stages N3, N4 same as N3
                labels.append(3)
            elif lbl == 5:  # Assign label 4 for REM stage
                labels.append(4)
            else:
                labels.append(lbl)
            if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                faulty_File = 1

        if faulty_File == 1:
            print( "============================== Faulty file ==================")
            continue

        labels = np.asarray(labels)

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values
        print(raw_ch.shape)

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        print(x.shape)
        print(y.shape)
        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != 0)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Saving as numpy files
        filename = os.path.basename(edf_fnames[file_id]).replace(".edf",  ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print(" ---------- Done this file ---------")



if __name__ == "__main__":
    main()
