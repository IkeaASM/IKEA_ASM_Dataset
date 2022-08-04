# compress depth scans (depth image dirs) within the dataset into a single zip file for transport
import os
import argparse
import tb_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/mnt/IronWolf/Datasets/ANU_ikea_dataset/',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/mnt/IronWolf/Datasets/ANU_ikea_dataset_compressed_depth/',
                    help='path to output location of the frames extracted from the video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()

category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files, \
rgb_params_files, _ = tb_utils.get_scan_list(args.dataset_path, devices=args.devices)


compressed_path_list = []
for i, scan in enumerate(depth_path_list):
    # frame_list = get_files(scan, file_type='.png')
    out_dirname = scan.replace(args.dataset_path, args.output_path)
    out_filename = os.path.join(out_dirname, 'depth_frames.tar.gz')
    scan_relative_path = scan.replace(args.dataset_path, '')
    if not os.path.isdir(out_dirname) :
        os.makedirs(out_dirname)
    os.system('cd {} && tar -czvf {} {}/*.png '.format(args.dataset_path, out_filename, scan_relative_path))
    compressed_path_list.append(scan_relative_path)

os.system('cd {} && tar -czvf {} {}'.format(args.output_path, 'all_depth_frames.tar.gz', ' '.join(compressed_path_list)))