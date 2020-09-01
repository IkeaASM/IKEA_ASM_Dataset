# Run visualization of the output of inference
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import cv2
import numpy as np
import json
import os
from detectron2.config import get_cfg
import argparse

colors = [(129, 0, 70), (220, 120, 0), (255, 100, 220), (6, 231, 255), (89, 0, 130), (251, 221, 64), (5, 5, 255)]
parser = argparse.ArgumentParser()
parser.add_argument('-s', help='name of sample such as, [0002_white_floor_05_02_2019_08_19_17_47]', required=True)
parser.add_argument('-root', default='/path/to/dataset/', required=True)


args = parser.parse_args()

if __name__ == '__main__':

    cfg = get_cfg()
    data_json = json.load(open(os.path.join(cfg.OUTPUT_DIR, args.s + '.json'), 'rb'))

    class_dict = {}
    for c in data_json['categories']:
        class_dict[c['id']] = c['name']
    img_list = data_json['images']
    segment_list = data_json['annotations']

    for im in range(len(img_list)):
        img_id = img_list[im]['id']
        file_name = img_list[im]['file_name']
        fname = str.split(file_name, '/')[-1]
        idx = [j for j in range(len(segment_list)) if segment_list[j]['image_id'] == img_id]
        file_n = str.split(file_name, '/')[-1]
        I = cv2.imread(os.path.join(args.root, file_name))

        for x in range(len(idx)):
            contours = []
            length = len(data_json['annotations'][idx[x]]['segmentation'])
            bbox = contour = data_json['annotations'][idx[x]]['bbox']
            for i in range(length):
                id = 0
                contour = data_json['annotations'][idx[x]]['segmentation'][i]
                cnt = len(contour)
                c = np.zeros((int(cnt / 2), 1, 2), dtype=np.int32)
                for j in range(0, cnt, 2):
                    c[id, 0, 0] = contour[j]
                    c[id, 0, 1] = contour[j + 1]
                    id = id + 1
                contours.append(c)

            color_cat = colors[data_json['annotations'][idx[x]]['category_id']-1]
            cv2.drawContours(I, contours, -1, color_cat, -1)
            I = cv2.rectangle(I, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0), 2)
            x1, y1 = bbox[:2]
            cv2.putText(I, class_dict[data_json['annotations'][idx[x]]['category_id']], (int(x1) - 10, int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 0, 0), 1)

        if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, args.s)):
            os.mkdir(os.path.join(cfg.OUTPUT_DIR, args.s))

        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, args.s) + '/' + fname, I)
        print('Image written for ', fname)
