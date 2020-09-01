import tarfile
from tqdm import tqdm
import argparse
import tb_utils as utils
import os

#Flags for postprocessing exportss
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/media/sitzikbs/6TB/ANU_ikea_dataset/', help='path to the ANU IKEA dataset')
parser.add_argument('--output_path', type=str, default='/media/sitzikbs/6TB/ANU_ikea_individually compressed/', help='path to the ANU IKEA dataset')
FLAGS = parser.parse_args()

INPUT_PATH = FLAGS.input_path
OUTPUT_PATH = FLAGS.output_path
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
category_path_list, scan_path_list, _, _, _, _, _ = utils.get_scan_list(INPUT_PATH)

# compress each scan individually and maintain directory tree structure
with tqdm(total=len(scan_path_list)) as pbar:
    for scan in scan_path_list:
        pbar.update(1)
        scan_output_path = scan.replace(INPUT_PATH, OUTPUT_PATH)

        categoty_path = os.path.dirname(scan_output_path)
        if not os.path.exists(categoty_path):
            os.makedirs(categoty_path)

        output_filename = scan_output_path + '.tar.gz'
        if not os.path.exists(output_filename):
            with tarfile.open(output_filename, "w:gz") as tar:
                tar.add(scan, arcname=os.path.basename(scan))
