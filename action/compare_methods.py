# Author: Yizhak Ben-Shabat (Itzik), 2020
# script to compare between multiple methods' resutls visually (generates image)

import numpy as np
import os
import utils
from IKEAActionDataset import IKEAActionDataset as Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'
methods_name = ['I3D side', 'I3D front', 'I3D top', 'I3D depth', 'combined_views', 'HCN (pose)', 'ST_GCN (pose)', 'combined RGB+Pose','combined all']
list_of_method_paths = [
                        # './frame_based/log/resnet18/results',
                        # './frame_based/log/resnet50/results',
                        './clip_based/i3d/log/dev1/results/',
                        './clip_based/i3d/log/dev2/results/',
                        './clip_based/i3d/log/dev3/results/',
                        './clip_based/i3d/log/depth/results/',
                        # './clip_based/c3d_and_p3d/log/c3d/results',
                        # './clip_based/c3d_and_p3d/log/p3d/results',
                        './log/combined/combined_views_i3d/results/',
                        './pose_based/log/HCN_32/results',
                        './pose_based/log/ST_GCN_64/results',
                        './log/combined/combined_RGB_pose/results/',
                        './log/combined/combined_all/results/',
                        ]

output_path = './log/compare_multiview_baselines/'
pred_output_filename = os.path.join(output_path, 'pred.npy')
json_output_filename = os.path.join(output_path, 'action_segments.json')

gt_json_path = os.path.join(dataset_path, 'gt_segments.json')
dataset = Dataset(dataset_path, action_segments_filename=gt_json_path)
gt_labels = dataset.action_labels
gt_labels = [np.argmax(vid_gt_labels, 1) for vid_gt_labels in gt_labels]
os.makedirs(output_path, exist_ok=True)

output_dict = {}
for i, method in enumerate(list_of_method_paths):
    results_npy = os.path.join(method, 'pred.npy')

    # load the predicted data
    pred_data = np.load(results_npy, allow_pickle=True).item()
    logits = np.array(pred_data['logits'])
    combined_logits = logits if i == 0 else combined_logits + logits
    pred_labels = [np.argmax(vid_logits,  axis=1) for vid_logits in logits]

    # get per cladd accuract
    c_matrix = confusion_matrix(np.concatenate(gt_labels[0:len(pred_labels)]).ravel(), np.concatenate(pred_labels).ravel(),
                                labels=range(dataset.num_classes)) #assums test videos are first in the json file
    class_acc = c_matrix.diagonal() / c_matrix.sum(1)
    output_dict[method] = class_acc

combined_logits = combined_logits / len(list_of_method_paths)
combined_prediction = []
for video in combined_logits:
    combined_prediction.append(np.argmax(video, axis=1))

c_matrix = confusion_matrix(np.concatenate(gt_labels[0:len(pred_labels)]).ravel(), np.concatenate(combined_prediction).ravel(),
                            labels=range(dataset.num_classes)) #assums test videos are first in the json file
class_acc = c_matrix.diagonal() / c_matrix.sum(1)
output_dict['combined'] = class_acc


# np.save(pred_output_filename, {'pred_labels': combined_prediction, 'logits': combined_logits.tolist()})

# methods_name = list(output_dict.keys())

# methods_name = ['ResNet18', 'ResNet50', 'I3D', 'C3D', 'P3D']

acc_mat = np.array(list(output_dict.values()))[:-1]
fig, ax = utils.plot_class_acc_comparison(acc_mat,
                          class_names=dataset.action_list,
                          methods_name=methods_name,
                          title=None,
                          cmap=None)

plt.savefig(os.path.join(output_path, 'method_comparison.png'))
