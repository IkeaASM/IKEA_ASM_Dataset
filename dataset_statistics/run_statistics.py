# this script runs the statistics on the dataset
import os
import sys
import stat_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
sys.path.append('../')
from action.IKEAActionDataset import IKEAActionDataset

matplotlib.rcParams.update({'font.size': 20})

dataset_path = '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/'

db_file = 'ikea_annotation_db_full'
train_file = 'train_cross_env.txt'
test_file = 'test_cross_env.txt'
action_list_file = 'atomic_action_list.txt'
action_object_relation_file = 'action_object_relation_list.txt'
fps = 25
dataset = IKEAActionDataset(dataset_path, db_file, action_list_file, action_object_relation_file, train_file, test_file)
show_action_names = False
output_path = './dataset_statistics/'
include_threshold = 20  # number of examples to include the action class in the dataset
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Comptue video duration statistics
n_videos, total_n_frames, frames_per_video_list = stat_utils.compute_video_duration_stats(dataset)
total_duration = total_n_frames / fps

plt.figure()
plt.hist(frames_per_video_list, bins=20)
plt.title("Distribution of video duration")
plt.xlabel('Temporal length (frames)')
plt.ylabel('# videos')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'video_duration_dist.pdf'))
print('Total number of videos: {}, {} per device.'.format(n_videos, n_videos/3))
print('Total number of frames in the dataset: {}, Total duration:{} hours'.format(total_n_frames, total_duration/(60*60)))
print('Average video number of frames: {} (~= {} min)'.format(np.mean(frames_per_video_list),
                                                                      np.mean(frames_per_video_list)/(fps*60)))

############ Compute action duration statistics #############################
frames_per_action_list = stat_utils.compute_action_duration_stats(dataset)

plt.figure()
plt.hist(frames_per_action_list, bins=50)
plt.title("Distribution of action duration")
plt.xlabel('Temporal length (frames)')
plt.ylabel('# action clips')
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'action_duration_dist.pdf'))
print('\nThe average action duration: {} frames (~{} sec)'.format(np.mean(frames_per_action_list),
                                                                  np.mean(frames_per_action_list)/fps))

matplotlib.rcParams.update({'font.size': 22})
############################################ compute action occurances in train and test split ########################

train_action_count, test_action_count = stat_utils.compute_action_train_test_proportions(dataset)
sum_counts = train_action_count + test_action_count
print("Taxonomy \n")
for i, action in enumerate(dataset.action_list):
    print("{} & {}".format(i, action))
action_names = dataset.squeeze_action_names()[1:]

#sort by frequency
sum_counts, action_names, train_action_count, test_action_count, order = (list(t) for t in zip(*sorted(list(zip(sum_counts[1:], action_names, train_action_count[1:], test_action_count[1:], range(len(action_names)))))))
print(sum_counts)
order = [str(elem + 1) for elem in order]
x_labels = action_names if show_action_names else order
rotation = 'vertical' if show_action_names else 'horizontal'
plt.figure(figsize=[15, 5])
p1 = plt.bar(x_labels, train_action_count, 0.75, color='C0')
p2 = plt.bar(x_labels, test_action_count, 0.75, bottom=train_action_count, color='orange')
plt.xticks(x_labels, rotation=rotation)
plt.xlim([0.5, len(action_names)-0.5 ])
plt.title("Actions train/test distribution", fontsize=38)
plt.xlabel('Action label',  fontsize=38)
plt.ylabel('# action clips',  fontsize=38)
plt.legend((p1[0], p2[0]), ('train', 'test'), bbox_to_anchor=(0.0, 1.0), loc="lower left",  fontsize=20)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(output_path, 'action_train_test_dist.pdf'))
print('Total number of actions {} - train:{}, test:{}'.format(np.sum(train_action_count) + np.sum(test_action_count),
                                                              np.sum(train_action_count), np.sum(test_action_count)))


####################### compute action instance occurances in train and test split ################################3
train_action_count, test_action_count = stat_utils.compute_occurrence_action_train_test_proportions(dataset)
sum_counts = train_action_count + test_action_count
action_names = dataset.squeeze_action_names()[1:]
#sort by frequency
sum_counts, action_names, train_action_count, test_action_count, order = (list(t) for t in zip(*sorted(list(zip(sum_counts[1:], action_names, train_action_count[1:], test_action_count[1:], range(len(action_names)))))))
order = [str(elem + 1) for elem in order]
x_labels = action_names if show_action_names else order
rotation = 'vertical' if show_action_names else 'horizontal'
plt.figure(figsize=[15, 5])
p1 = plt.bar(x_labels, train_action_count, 0.75, color='C0')
p2 = plt.bar(x_labels, test_action_count, 0.75, bottom=train_action_count, color='orange')
plt.xticks(x_labels, rotation=rotation)
plt.xlim([0.5, len(action_names)-0.5 ])
plt.title("Actions occurrence train/test distribution", fontsize=38)
plt.xlabel('Action label', fontsize=38)
plt.ylabel('# action clips', fontsize=38)
# plt.legend((p1[0], p2[0]), ('train', 'test'), bbox_to_anchor=(0.0, 1.0), loc="lower left", fontsize=20)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(output_path, 'action_occurrence_train_test_dist.pdf'))
print('Total number of action occurences {} - train:{}, test:{}'.format(np.sum(train_action_count) + np.sum(test_action_count),
                                                              np.sum(train_action_count), np.sum(test_action_count)))
