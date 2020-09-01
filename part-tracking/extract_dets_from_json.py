# Convert detections of json format to standard tracking format
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import json
import argparse


# Find image name from image_id
def find_name(image_id, test_data):
    for item in test_data['images']:
        if item['id'] == image_id:
            return item['file_name']
    return -1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default="0002_white_floor_05_02_2019_08_19_17_47", required=True)
    args = parser.parse_args()

    output = 'dets_' + args.s + '.txt'
    input = args.s + '.json'

    # Select high confident predictions of the model to track
    score_thresh = 0.9
    fid = open(output, 'w')

    test_results = json.load(open(input, 'r'))
    all_boxes_dict = {}
    for item in test_results['annotations']:
        if item['image_id'] not in all_boxes_dict:
            all_boxes_dict[item['image_id']] = []
        all_boxes_dict[item['image_id']].append(item)

    for key, val in all_boxes_dict.items():
        image_name = find_name(key, test_results)
        fname = str.split(image_name, '/')[-1]
        fname_id = str.split(fname, '.')[0]
        for j in range(len(val)):
            if val[j]['score'] > score_thresh:
                # Write each detection in the standard format for tracking
                fid.write(fname_id + ',' + '-1' + ',' + str(val[j]['bbox'][0]) + ',' + str(val[j]['bbox'][1])
                          + ',' + str(val[j]['bbox'][2] - val[j]['bbox'][0]) + ','
                          + str(val[j]['bbox'][3] - val[j]['bbox'][1])+'\n')
    print(output + ' file created for tracking!')






