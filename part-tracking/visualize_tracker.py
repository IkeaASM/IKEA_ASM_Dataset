# Run visualization for tracklets of each video sample
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import cv2
import glob
import random
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', help='name of sample, like: <0007_oak_floor_01_01_2019_08_14_17_17>', required=True)
    parser.add_argument('-root', help='path to the images up to a directory before furniture name, '
                                      'consider the path like this '
                                      '/ikea_data/Lack_Coffee_Table/0002_white_floor_05_02_2019_08_19_17_47/dev3/images'
                                      ' the root will be /ikea_data/', required=True)
    parser.add_argument('-f', help='name of furniture from [Kallax_Shelf_Drawer, Lack_Side_Table, Lack_Coffee_Table, Lack_TV_Bench]', default="Lack_Coffee_Table", required=True)
    args = parser.parse_args()

    input = 'tracklets_' + args.s + '.txt'

    fid = open(input, 'r')
    lines = fid.read()
    lines = str.split(lines, '\n')[0:-1]

    if not os.path.exists('tracking_results_' + args.s):
        os.mkdir('tracking_results_' + args.s)

    frames = glob.glob(os.path.join(args.root, args.f, args.s) + '/dev3/images/*')
    frames.sort()
    frames_dic = {}
    exist_ids = {}

    for line in lines:
        items = str.split(line, ' ')
        id = int(items[-2])
        x = float(items[0])
        y = float(items[1])
        w = float(items[2])
        h = float(items[3])
        frame = int(items[-1])
        if frame not in frames_dic:
            frames_dic[frame] = []
        frames_dic[frame].append([x, y, w, h, id])

    for key, value in frames_dic.items():
        I = cv2.imread(frames[key])
        for v in value:
            if v[-1] not in exist_ids.keys():
                r = lambda: random.randint(0, 255)
                exist_ids[v[-1]] = (r(), r(), r())
                cv2.rectangle(I, (int(v[0]), int(v[1])), (int(v[0] + v[2]), int(v[1] + v[3])), exist_ids[v[-1]], 2)
                cv2.putText(I, str(v[-1]), (int(v[0]) + 10, int(v[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.rectangle(I, (int(v[0]), int(v[1])), (int(v[0] + v[2]), int(v[1] + v[3])), exist_ids[v[-1]], 2)
                cv2.putText(I, str(v[-1]), (int(v[0]) + 10, int(v[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imwrite('tracking_results_' + args.s + '/' + str.split(frames[key], '/')[-1], I)
        print('frame ', key, ' has been written!')

