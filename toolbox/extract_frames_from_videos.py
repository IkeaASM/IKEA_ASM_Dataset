import argparse
import tb_utils as utils
import os
import multiprocessing
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed_frames',
                    help='path to output location of the frames extracted from the video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev1', 'dev2', 'dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()


category_path_list, scan_list, rgb_path_list, depth_path_list, depth_params_files, \
rgb_params_files, normals_path_list = utils.get_scan_list(args.dataset_path, devices=args.devices)

print('Video dataset path: ' + args.dataset_path)
print('Individual frames dataset will be saved to ' + args.output_path)


os.makedirs(args.output_path, exist_ok=True)
for scan_list in [rgb_path_list, depth_path_list, normals_path_list]:
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(utils.extract_frames)(scan, args.dataset_path,
                                                            args.output_path) for _, scan in enumerate(scan_list))
    # # Non-parallel implementation
    # for scan in scan_list:
    #     utils.extract_frames(scan, args.dataset_path, args.output_path)

