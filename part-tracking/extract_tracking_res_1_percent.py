# extract tracking results for 1% of the frames that we have GT for them
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', default="0002_white_floor_05_02_2019_08_19_17_47", required=True)
parser.add_argument('-f', help='name of furniture from [Kallax_Shelf_Drawer, Lack_Side_Table, Lack_Coffee_Table, Lack_TV_Bench]', required=True)
parser.add_argument('-root', help='path of GT up to name of directory before furniture name', required=True)

args = parser.parse_args()

gt_fid = open(os.path.join(args.root, args.f, args.s) + '/dev3/tracking_gt.txt')
gt_samples = str.split(gt_fid.read(), '\n')[0:-1]

frames = [int(str.split(s, ',')[0]) for s in gt_samples]
frames = np.unique(frames)
input = 'tracklets_' + args.s + '.txt'
output = 'tracking_1_percent_' + args.s + '.txt'
seq_dets = np.loadtxt(input, delimiter=' ')  # load detections
output_file = open(output, 'w')
for seq in seq_dets:
    if seq[-1] in frames:
        str_to_srite = '%d,%d,%f,%f,%f,%f,-1,-1,-1\n' % (seq[5], seq[4], seq[0], seq[1], seq[2], seq[3])
        output_file.write(str_to_srite)
output_file.close()



