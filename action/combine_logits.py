# Author: Yizhak Ben-Shabat (Itzik), 2020
# A script to combine logits from different methods

import numpy as np
import os
import json
import utils
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset

list_of_method_paths = [
                        # './clip_based/c3d_and_p3d/log/p3d/results/',
                        # './clip_based/i3d/log/dev1/results/',
                        # './clip_based/i3d/log/dev2/results/',
                        './clip_based/i3d/log/dev3/results/',
                         # './clip_based/i3d/log/depth/results/',
                        './pose_based/log/HCN_32/results'
                        ]

output_path = './log/combined/combined_top_view_pose/results/'
pred_output_filename = os.path.join(output_path, 'pred.npy')
json_output_filename = os.path.join(output_path, 'action_segments.json')
os.makedirs(output_path, exist_ok=True)

methods_output_dilename = os.path.join(output_path, 'methods.json')
with open(methods_output_dilename, 'w') as outfile:
    json.dump({'methods': list_of_method_paths}, outfile)



for i, method in enumerate(list_of_method_paths):
    results_npy = os.path.join(method, 'pred.npy')

    # load the predicted data
    pred_data = np.load(results_npy, allow_pickle=True).item()
    logits = np.array(pred_data['logits'])
    combined_logits = logits if i == 0 else combined_logits + logits

combined_logits = combined_logits / len(list_of_method_paths)
combined_prediction = []
for video in combined_logits:
    combined_prediction.append(np.argmax(video, axis=1))

np.save(pred_output_filename, {'pred_labels': combined_prediction, 'logits': combined_logits.tolist(),
                               'methdos': list_of_method_paths})

test_dataset = Dataset(dataset_path='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/',
                       db_filename='ikea_annotation_db_full', test_filename='test_cross_env.txt',
                       train_filename='train_cross_env.txt', set='test',  resize=None, mode='img')

utils.convert_frame_logits_to_segment_json(combined_logits, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)

os.system('python3 ./evaluation/evaluate.py --results_path {} --mode img'.format(output_path))
