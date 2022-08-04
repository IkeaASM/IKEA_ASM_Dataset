# compress depth scans (depth image dirs) within the dataset into a single zip file for transport
import os
import argparse
import tb_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/mnt/IronWolf/Datasets/temp/extracted_depth/',
                    help='path to the ANU IKEA assembly video dataset')
parser.add_argument('--output_path', type=str, default='/mnt/IronWolf/Datasets/temp/compressed_depth/',
                    help='path to output location of the frames extracted from the video dataset')
parser.add_argument('--devices', nargs='+',  default=['dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
args = parser.parse_args()

os.makedirs(args.dataset_path, exist_ok=True)

# os.system('cd {} && tar -xvf {} -C {} && rm {}'.format(args.output_path, 'all_depth_frames.tar.gz', args.dataset_path,'all_depth_frames.tar.gz'))
os.system('cd {} && tar -xvf {} -C {}'.format(args.output_path, 'all_depth_frames.tar.gz', args.dataset_path))

compressed_file_list = []
for path, subdirs, files in os.walk(args.dataset_path):
    for name in files:
        if '.tar.gz' in name:
            compressed_file_list.append(os.path.join(path, name))

for i, tarfilename in enumerate(compressed_file_list):
    os.system('cd {} && tar -xvf {} -C {} && rm {}'.format(args.dataset_path, tarfilename, args.dataset_path, tarfilename))

