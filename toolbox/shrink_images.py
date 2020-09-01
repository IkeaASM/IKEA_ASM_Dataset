from tqdm import tqdm
import argparse
import tb_utils as utils
import os
from PIL import Image
from multiprocessing import Pool
import cv2
import numpy as np

#Flags for postprocessing exports
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset/', help='path to the ANU IKEA dataset')
parser.add_argument('--output_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_smaller_depth/', help='path to the ANU IKEA dataset')
parser.add_argument('--shrink_factor', type=int, default=1.8, help='factor to shrink the imagesm recommended setting 4 to rgb and 1.8 to depth')
parser.add_argument('--mode', type=str, default='rgb', help='depth | rgb, indicating if to shrink rgb or depth images')
FLAGS = parser.parse_args()

INPUT_PATH = FLAGS.input_path
OUTPUT_PATH = FLAGS.output_path
SHRINK_FACTOR = FLAGS.shrink_factor
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

if FLAGS.mode == 'depth':
    file_list = utils.get_list_of_all_files(INPUT_PATH, file_type='.png')
    file_list = [file for file in file_list if 'dev3' in file]
else:
    file_list = utils.get_list_of_all_files(INPUT_PATH, file_type='.jpg')

print(0)

def shrink_and_save(filename):
    # saves and shrinks a file from the dataset.
    # if the file already exists in the output directory - skip it.

    output_filename = filename.replace(INPUT_PATH, OUTPUT_PATH)
    output_dir_path = os.path.dirname(os.path.abspath(output_filename))
    if not os.path.exists(output_filename):
        os.makedirs(output_dir_path, exist_ok=True)

        if FLAGS.mode == 'depth':
            img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            img = img * 255 / 4500
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, dsize=(int(img.shape[0] / SHRINK_FACTOR), int(img.shape[1] / SHRINK_FACTOR)))
            cv2.imwrite(output_filename, img)
        else:
            # shrink and save the data
            img = Image.open(filename)
            img = img.resize((int(img.size[0] / SHRINK_FACTOR), int(img.size[1] / SHRINK_FACTOR)))
            img.save(output_filename)


# compress each scan individually and maintain directory tree structure - parallel to speed it up
with Pool(8) as p:
  list(tqdm(p.imap(shrink_and_save, file_list), total=len(file_list)))
