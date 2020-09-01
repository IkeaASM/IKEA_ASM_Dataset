import argparse
import tb_utils as utils
import os

#Flags for postprocessing exportss
parser = argparse.ArgumentParser()
parser.add_argument('--point_clouds', type=int, default=False, help='export point clouds')
parser.add_argument('--normal_vectors', type=int, default=False, help='export normal vectors images')
parser.add_argument('--pose_img', type=int, default=False, help='export pose images from gt json')
parser.add_argument('--seg_img', type=int, default=False, help='export object segmentation images from gt json')
parser.add_argument('--perception_demo_img', type=int, default=False, help='export perception demo images')

parser.add_argument('--rgb_vid', type=int, default=False, help='export rgb video file')
parser.add_argument('--depth_vid', type=int, default=True, help='export depth video file')
parser.add_argument('--normals_vid', type=int, default=False, help='export normals video file')
parser.add_argument('--pose_vid', type=int, default=False, help='export pose video file')
parser.add_argument('--seg_vid', type=int, default=False, help='export object segmentation video file')
parser.add_argument('--perception_demo_vid', type=int, default=False, help='export object segmentation video file')
parser.add_argument('--montage_vid', type=int, default=False, help='export video montage file')

parser.add_argument('--input_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset/',
                    help='path to the ANU IKEA dataset')
parser.add_argument('--output_path', type=str, default='/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_processed/',
                    help='path to the output dir')
# parser.add_argument('--scan_name', type=str, default='Lack_Side_Table/0007_oak_floor_01_01_2019_08_14_17_17',
#                     help='name of single scan to export') # preception demo video
parser.add_argument('--scan_name', type=str, default='Lack_Coffee_Table/0044_oak_table_10_04_2019_08_28_15_19',
                    help='name of single scan to export')
parser.add_argument('--skeleton_type', type=str, default='openpose', help='openpose | keypoint_rcnn formats')
parser.add_argument('--devices', nargs='+',  default=['dev3'],
                    help='dev1 | dev2 | dev3 list of device to export')
FLAGS = parser.parse_args()

EXPORT_PC = FLAGS.point_clouds
EXPORT_N = FLAGS.normal_vectors
EXPORT_POSE = FLAGS.pose_img
EXPORT_SEG = FLAGS.seg_img
EXPORT_PERCEPTION_DEMO = FLAGS.perception_demo_img
EXPORT_VID_RGB = FLAGS.rgb_vid
EXPORT_VID_DEPTH = FLAGS.depth_vid
EXPORT_VID_NORMALS = FLAGS.normals_vid
EXPORT_VID_POSE = FLAGS.pose_vid
EXPORT_VID_MONTAGE = FLAGS.montage_vid
EXPORT_VID_SEG = FLAGS.seg_vid
EXPORT_VID_PERCEPTION_DEMO = FLAGS.perception_demo_vid

SKELETON_TYPE = FLAGS.skeleton_type
DEVICES = FLAGS.devices

INPUT_PATH = FLAGS.input_path
OUTPUT_PATH = FLAGS.output_path
SCAN_NAME=FLAGS.scan_name

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    print("Output files will be saved to " + OUTPUT_PATH)


    if EXPORT_PC:
        utils.export_point_clouds(INPUT_PATH, output_path=OUTPUT_PATH)
    if EXPORT_N:
        utils.export_normal_vectors(INPUT_PATH, output_path=OUTPUT_PATH)
    if EXPORT_POSE:
        utils.export_pose_images(INPUT_PATH, output_path=OUTPUT_PATH,
                                 scan_name=SCAN_NAME, mode='skeleton', skeleton_type=SKELETON_TYPE)
    if EXPORT_SEG:
        utils.export_seg_images(INPUT_PATH, output_path=OUTPUT_PATH,
                                     scan_name=SCAN_NAME)
    if EXPORT_PERCEPTION_DEMO:
        utils.export_perception_demo_images(INPUT_PATH, output_path=OUTPUT_PATH,
                                     scan_name=SCAN_NAME, mode='skeleton')
    # change when all pose labels are available
    if EXPORT_VID_RGB or EXPORT_VID_DEPTH or EXPORT_VID_NORMALS or EXPORT_VID_POSE or EXPORT_VID_SEG or EXPORT_VID_PERCEPTION_DEMO:
        utils.export_videos(INPUT_PATH, rgb_flag=EXPORT_VID_RGB, depth_flag=EXPORT_VID_DEPTH,
                            normals_flag=EXPORT_VID_NORMALS, pose_flag=EXPORT_VID_POSE, seg_flag=EXPORT_VID_SEG,
                            perception_demo_flag=EXPORT_VID_PERCEPTION_DEMO,
                            output_path=OUTPUT_PATH, vid_scale=1, vid_format='avi', devices=DEVICES)
    if EXPORT_VID_MONTAGE:
        utils.export_video_montage(INPUT_PATH, OUTPUT_PATH)
