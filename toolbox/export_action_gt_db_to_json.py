import numpy as np
from torch.utils import data
import sys
sys.path.append('../action/')
from IKEAActionDataset import IkeaAllSingleVideoActionDataset as Dataset
import os
import utils

gpu_idx = 1
batch_size = 1024
dataset_path = "/mnt/IronWolf/Datasets/ANU_ikea_dataset_smaller/"
test_filename = 'test_cross_env.txt'
train_filename = 'train_cross_env.txt'
output_path = dataset_path
os.makedirs(output_path, exist_ok=True)


def convert_action_gt_db_to_json():

    # Generators
    dataset = {"testing": Dataset(dataset_path=dataset_path, set='test', transform=None, test_filename=test_filename),
               "training": Dataset(dataset_path=dataset_path, set='train', transform=None, test_filename=test_filename)}
    loader = {"testing": data.DataLoader(dataset["testing"], batch_size=batch_size, shuffle=False, num_workers=8),
              "training": data.DataLoader(dataset["training"], batch_size=batch_size, shuffle=False, num_workers=8)}

    gt_output_filename = os.path.join(output_path, 'gt_action.npy')
    json_output_filename = os.path.join(output_path, 'gt_segments.json')
    # switch to evaluate mode


    gt_labels_per_video = []
    subset = []
    for phase in ["testing", "training"]:
        gt_labels_accum = np.array([], dtype=np.int8)
        last_vid_idx = 0

        for step, (im_data, im_class, _, vid_idx) in enumerate(loader[phase]):
            print('Processing video batch [{}/{}]'.format(step, len(loader[phase])))
            gt_labels = im_class.detach().cpu().numpy().astype(np.int8)
            vid_idx = vid_idx.detach().cpu().numpy()

            # accumulate results per video
            if not vid_idx[0] == vid_idx[-1]:  # video index changes in the middle of the batch
                print('Finished video {}'.format(vid_idx[0]))
                idx_change = np.where(vid_idx[:-1] != vid_idx[1:])[0][0] + 1  # find where the video index changes
                gt_labels_accum = np.concatenate([gt_labels_accum, gt_labels[:idx_change]])
                gt_labels_per_video.append(gt_labels_accum)
                gt_labels_accum = np.array(gt_labels[idx_change:], dtype=np.int8)
            elif not last_vid_idx == vid_idx[0]:  # video index changes after a full batch
                print('Finished video {}'.format(last_vid_idx))
                gt_labels_per_video.append(gt_labels_accum)
                gt_labels_accum = np.array(gt_labels, dtype=np.int8)
            elif step == len(loader[phase])-1:  # last batch
                print('Finished video {} - the last batch'.format(last_vid_idx))
                gt_labels_accum = np.concatenate([gt_labels_accum, gt_labels])
                gt_labels_per_video.append(gt_labels_accum)
            else:
                gt_labels_accum = np.concatenate([gt_labels_accum, gt_labels])

            last_vid_idx = vid_idx[-1]
        subset = subset + [phase]*len(dataset[phase].video_list)

    file_list = dataset["testing"].video_list + dataset["training"].video_list
    np.save(gt_output_filename, {'scan_name': file_list, 'gt_labels': gt_labels_per_video})
    utils.convert_db_to_segment_json(gt_labels_per_video, json_output_filename, file_list,
                                               dataset["testing"].action_list, mode="labels", subset=subset)


if __name__ == '__main__':
    convert_action_gt_db_to_json()