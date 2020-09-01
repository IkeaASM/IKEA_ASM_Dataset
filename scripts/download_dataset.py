# Author: Yizhak Ben-Shabat (Itzik), 2020
# This script fails for large files due to google drive limitations
# A script to download all elements of the dataset and extract them into a single dataset dir
# it allows to download all or part of the data
# By default it will download calibration data, indexing files data, RGB top view videos and action annotations
# after downloading the videos extract the frames using "extract_frames_from_videos.py" in the toolbox dir.

import argparse
# import gdown
import requests
import zipfile
import os


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def get_flags(args):
    """

    Parameters
    ----------
    args : input arguments

    Returns
    -------
    flag_dict : dictionary of boolean values indicating which part of the dataset to download
    """
    flag_dict = {'data': {'top_view': False, 'utility': True, 'camera_params': True, 'multi-view': False, 'depth': False},
                 'annotations': {'action': False, 'pose': False, 'segmentation': False},
                 'pt_model': {'action': False, 'pose': False, 'segmentation': False}}

    if args.download_all:
        for data_type in flag_dict:
            for key in flag_dict[data_type].keys():
                flag_dict[data_type][key] = True
    # set to get the data

    if args.top_view_data:
        flag_dict['data']['top_view'] = True
    if args.multi_view_data:
        flag_dict['data']['multi-view'] = True
    if args.depth_data:
        flag_dict['data']['depth'] = True

    # set to get the annotations and pre-trained models
    if args.action_ann:
        flag_dict['annotations']['action'] = True
        if args.pre_trained_models:
            flag_dict['pt_models']['action'] = True
    if args.pose_ann:
        flag_dict['annotations']['pose'] = True
        if args.pre_trained_models:
            flag_dict['pt_models']['pose'] = True
    if args.instance_segmentation_ann:
        flag_dict['annotations']['segmentation'] = True
        if args.pre_trained_models:
            flag_dict['pt_models']['segmentation'] = True

    return flag_dict

parser = argparse.ArgumentParser()
parser.add_argument('--output_dataset_path', type=str, default='../../ikea_asm_dataset/',
                    help='path to the output dir where all of the data files will be downloaded and extracted,'
                         ' requies approximately 400GB for the full video dataset and 2TB after extracting the frames')
parser.add_argument('--download_all', action="store_true",
                    help='flag to download all data, annotations and trained models')

parser.add_argument('--top_view_data', action="store_true", help='flag to download top view RGB data')
parser.add_argument('--multi_view_data',  action="store_true", help='flag to download side and front view RGB data')
parser.add_argument('--depth_data',  action="store_true", help='flag to download depth data')

parser.add_argument('--action_ann', action="store_true",
                    help='flag to download action annotations and optionally trained models')
parser.add_argument('--pose_ann', action="store_true",
                    help='flag to download pose annotations and optionally trained models')
parser.add_argument('--instance_segmentation_ann', action="store_true",
                    help='flag to download instance segmentation annotations and optionally trained models')

parser.add_argument('--pre_trained_models', action="store_true",
                    help='flag to download pretrained models for the flagged data')
args = parser.parse_args()


os.makedirs(args.output_dataset_path, exist_ok=True)

flag_dict = get_flags(args)

# flag_dict = {'data': {'top_view': False, 'utility': False, 'camera_params': False, 'multi-view': False, 'depth': False},
#              'annotations': {'action': False, 'pose': False, 'segmentation': False},
#              'pt_model': {'action': False, 'pose': True, 'segmentation': False}}

file_id_dict = {'data': {'utility': '11D7d8XBRg-CPIxMroviQEaaMhw3EaGnB',
                         'camera_params': '1BRq9HJQeEJFbhnCwGwY3eXe1587TybCe',
                         'top_view': '1CFOH-W-6N50AVA_NqHnm06GUsfpcka0L',
                         'multi-view': '1eCbrIuw--16xCmI3RtBhRJ-r9K_FVkL6',
                         'depth': '18FKRSzoUiO3EV_J2WmQyvmPGiHJcH28S',
                         },
                'annotations': {'action': '1SwBNLViktSpk99jhh3sMXVGTMVr6tpju',
                                'pose': '1RE7Ya1gwogqJtJIi5WeYOH4_Cs1RuTx7',
                                'segmentation': '1_jRCcLAz9zhXTnNnslBUJcu2sZjp9dVV'},
                'pt_model': {'action': '1QksK_Uvty6pTYoGmBGWYYG3scvM_NX2X',
                                'pose': '1SMoYC-PTHr6Y2StKKT8j_-gSYcwhTHKb',
                                'segmentation': '1lLNiWU6ILFCgg104FDwWvRMV0iQaGKyp'}
                }
file_name_dict = {'data': {'utility': 'utility_files.zip',
                           'camera_params': 'camera_params.zip',
                           'top_view': 'ikea_asm_dataset_RGB_top.zip',
                           'multi-view': 'ikea_asm_dataset_RGB_multiview.zip',
                           'depth': 'ikea_asm_dataset_depth.zip'},
                  'annotations': {'action': 'action_annotation.zip',
                                 'pose': 'pose_annotations.zip',
                                 'segmentation': 'segmentation_tracking_annotation.zip'},
                  'pt_model': {'action': 'pt_models_action_recognition.zip',
                               'pose': 'pt_models_pose_estimation.zip',
                               'segmentation': 'pt_models_instance_segmentation.zip'}
                  }

# md5s = {'data': {'utility': 'be7bf16936aa9777d1c023583459d098',
#                'camera_params': '51742d35cdd8f9eba8620a43ce53814f',
#                'top_view': 'f380157bf1ece925e1f8f5374da540df',
#                'multi-view': '72ed317addae2307ee7701c1fa50cdca',
#                'depth': 'bce97a1ee3512c48e5c6dea316894d76'},
#       'annotations': {'action': '719c5b0a1cf20ed56ffe90543e2fdba8',
#                      'pose': '1092e342a5ae91de63cdf9d7b445c7dd',
#                      'segmentation': '95569f751e9782494a872efb92c7a977'},
#       'pt_model': {'action': '3b238583772f37ae68dd2ccb86a46923',
#                    'pose': '35ab1161cc98ce665e7118edbf4e927d',
#                    'segmentation': 'd0f6eb63cb9e177e5317068ac8911cd0'}
#                   }

for data_type in file_id_dict.keys():
    for benchmark_type in file_id_dict[data_type].keys():
        if flag_dict[data_type][benchmark_type]:
            file_id = file_id_dict[data_type][benchmark_type]
            destination = os.path.join(args.output_dataset_path, file_name_dict[data_type][benchmark_type])
            print('Downloading ' + benchmark_type + ' ' + data_type + ' files. This may take a while...')

            download_file_from_google_drive(file_id, destination)

            # url = 'https://drive.google.com/uc?id=' + file_id
            # gdown.download(url, destination, quiet=False)

            # md5 = md5s[data_type][benchmark_type]
            # gdown.cached_download(url, destination, md5=md5, postprocess=gdown.extractall)

            print('Unzipping file')
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(args.output_dataset_path)

            print('Deleting zip file')
            os.remove(destination)

            print('Done processing ' + benchmark_type + ' ' + data_type + ' files')