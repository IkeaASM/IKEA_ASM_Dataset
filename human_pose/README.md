# Human Pose Estimation Baselines

### Installation

First, install the baseline code implementations from their respective GitHub repositories. Refer to their `README.md` files for installation details.
- OpenPose \[PyTorch implementation\] ([link](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation))
- MaskRCNN/KeypointRCNN (provided, `./keypointrcnn/main.py`)
- STAF ([link](https://github.com/soulslicer/openpose/tree/staf)): use the `staf` branch
- HMMR ([link](https://github.com/akanazawa/human_dynamics))
- VP3D ([link](https://github.com/facebookresearch/VideoPose3D))
- VIBE ([link](https://github.com/mkocabas/VIBE))


### Training and Testing

**OpenPose:**

Copy the additional required files (modified for the IKEA dataset) to the OpenPose directory. For example, if the OpenPose directory was cloned to `./openpose_pytorch`, use
```
cp -vr openpose/* openpose_pytorch
```

For training, we use a flattened directory structure, divided into test and train folders, and downsize the images to 640x360.
```
<root>
    --train
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.png
    ----...
    --test
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.png
    ----...
```
This can be created using the following command (may take a while, requires 3.8GB):
```
python ikea_pose_dataset.py --orig_dataset_dir <ORIGINAL DATASET DIRECTORY> --dataset_dir <NEW FLATTENED DATASET DIRECTORY>
```
Replace the `DATA_DIR` string in `./train/train_VGG19_ikea.py` and `./train/train_VGG19_ikea_ft.py` with the new directory location.

The model can be trained using the following command:
```
python train/train_VGG19_ikea_ft.py --batch-size 32 --log_dir experiments_ikea/ --lr 1.0 --epochs 40
```

For testing, use the following command:
```
python run_inference.py --dataset_dir <DATASET LOCATION> --run_openpose_pytorch --openpose_pytorch_dir <OPENPOSE DIRECTORY> --device <GPU ID>
```

**MaskRCNN/KeypointRCNN:**

For training, we use a flattened directory structure, divided into test and train folders, and downsize the images to 640x360.
```
<root>
    --train
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.png
    ----...
    --test
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.png
    ----...
```
This can be created using the following command (may take a while, requires 3.8GB):
```
python ikea_pose_dataset.py --orig_dataset_dir <ORIGINAL DATASET DIRECTORY> --dataset_dir <NEW FLATTENED DATASET DIRECTORY>
```

The model can be trained using the following command:
```
python keypointrcnn.py --gpu 0 --lr 0.01 --logdir ./logs/ <DATASET LOCATION>
```

For testing, we use the standard structured dataset directory without test/train splits.
```
python keypointrcnn.py --gpu 0 --save_results frames --resume <CHECKPOINT FILE LOCATION> <DATASET LOCATION>
```

**STAF:**

For testing, use the following command:
```
python run_inference.py --dataset_dir <DATASET LOCATION> --run_openpose_staf --openpose_staf_dir <STAF DIRECTORY> --device <GPU ID>
```

**HMMR:**

For testing, use the following command (activating the virtual environment first, if used):
```
source <HMMR DIRECTORY>/venv_hmmr/bin/activate
python run_inference.py --dataset_dir <DATASET LOCATION> --run_hmmr --hmmr_dir <HMMR DIRECTORY> --device <GPU ID>
deactivate
```

**VP3D:**

For testing, use the following command:
```
python vp3d.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin --dataset_dir <DATASET DIRECTORY> --vp3d_dir <VP3D DIRECTORY>
```

Requires OpenPose 2D detections per video, in the subfolder `predictions/pose2d/openpose/`.

**VIBE:**

For testing, use the following command (activating the virtual environment first, if used):
```
source <VIBE DIRECTORY>/vibe-env/bin/activate
python run_inference.py --dataset_dir <DATASET LOCATION> --run_vibe --vibe_dir <VIBE DIRECTORY> --device <GPU ID>
deactivate
```

### Evaluation

To evaluate a method, run
```
python eval.py --dataset_dir <DATASET LOCATION> <EVAL FLAG>
```
with \<EVAL FLAG\> set to the relevant method:
```
--eval_openpose_pt
--eval_openpose_ft
--eval_openpose_staf
--eval_keypoint_rcnn_pt
--eval_keypoint_rcnn_ft
--eval_hmmr
--eval_vp3d
--eval_vibe
```
This will run the evaluation metrics, save the results in the `./results/` folder, and display the results to the screen in LaTeX table format.

### Results

The results for 2D and 3D human pose baselines are reported in the tables below. The methods are evaluated with respect to the ground truth 2D human joint annotations (1% of all frames) and the pseudo ground truth 3D human joint annotations (1% of all frames).

| Method        | Input | MPJPE &#8595; | PCK &#8593; | AUC &#8593; |
|---------------|-------|-----------|-----------|-----------|
| OpenPose-pt	| Image | 17.3		| 46.9		| 78.1      |
| OpenPose-ft	| Image | 11.8		| 57.8		| 87.7      |
| MaskRCNN-pt	| Image | 15.5		| 51.9		| 78.2      |
| MaskRCNN-ft	| Image | **7.6**	| **77.6**	| **92.1**  |
| STAF-pt		| Video | 21.4		| 41.8		| 75.3      |

**Table 1:** 2D human pose results (train set). The Mean Per Joint Position Error (MPJPE) in pixels and the Percentage of Correct Keypoints (PCK) @ 10 pixels (0.5% image width) are reported. Pretrained models are denoted 'pt' and models fine-tuned on the training data are denoted 'ft'.

| Method        | Input | MPJPE &#8595; | PCK &#8593; | AUC &#8593; |
|---------------|-------|-----------|-----------|-----------|
| OpenPose-pt	| Image | 16.5              | 46.7             | 77.8             |
| OpenPose-ft	| Image | 13.9              | 52.6             | 85.6             |
| MaskRCNN-pt	| Image | 16.1              | 51.5             | 79.2             |
| MaskRCNN-ft	| Image | **11.5**	  	   | **64.3** 		  | **87.8** 		 |
| STAF-pt		| Video | 19.7              | 41.1             | 75.4             |

**Table 2:** 2D human pose results (test set). The Mean Per Joint Position Error (MPJPE) in pixels and the Percentage of Correct Keypoints (PCK) @ 10 pixels (0.5% image width) are reported. Pretrained models are denoted 'pt' and models fine-tuned on the training data are denoted 'ft'.

| Method | Input | MPJPE &#8595; | MPJPE (PA) &#8595; | mPJPE &#8595; | mPJPE (PA) &#8595; | PCK &#8593; | PCK (PA) &#8593; |
|---|---|---|---|---|---|---|---|
| HMMR | Video | 589 | **501** | 189 | 96 | 32 | 54 |
| VP3D | Video | **546** | 518 | **111** | 87 | **63** | 70 |
| VIBE | Video | 568 | 517 | 139 | **81** | 55 | **74** |

**Table 3:** 3D human pose results (train set). The Mean Per Joint Position Error (MPJPE) in millimeters, the median PJPE (mPJPE), and the Percentage of Correct Keypoints (PCK) @ 150mm are reported, with and without Procrustes alignment (PA). Only confident ground-truth annotations are used and only detected joints contribute to the errors.

| Method | Input | MPJPE &#8595; | MPJPE (PA) &#8595; | mPJPE &#8595; | mPJPE (PA) &#8595; | PCK &#8593; | PCK (PA) &#8593; |
|---|---|---|---|---|---|---|---|
| HMMR | Video | 1012 | 951 | 369 | 196 | 25 | 40 |
| VP3D | Video | **930** | **913** | 212 | 179 | **44** | 47 |
| VIBE | Video | 963 | 940 | **199** | **153** | 43 | **50** |

**Table 4:** 3D human pose results (test set). The Mean Per Joint Position Error (MPJPE) in millimeters, the median PJPE (mPJPE), and the Percentage of Correct Keypoints (PCK) @ 150mm are reported, with and without Procrustes alignment (PA). Only confident ground-truth annotations are used and only detected joints contribute to the errors.

### Miscellaneous

**Calibration:**
To find the intrinsic and extrinsic camera parameters from the chessboard calibration images in `./Calibration/`, use the `calibration.py` script.
```
calibration.py --dataset_dir <DATASET LOCATION>
```

**Triangulation:**
To estimate the pseudo ground truth 3D joint annotations, run triangulation on the 3-view 2D detections with the camera parameters found during calibration, using the `triangulation.py` script. Assumes 2D predictions are available for all frames with ground truth 2D annotations.
```
triangulation.py --dataset_dir <DATASET LOCATION>
```