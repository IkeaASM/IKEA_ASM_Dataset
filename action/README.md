# IKEA ASM Dataset - Action recognition baselines

We provide three types of action recognition baselines: 
* Frame based - using each frame separately to predict per frame labels.
* Clip based -  using short sequences to predict per frame action labels.
* Pose based- using the pose labels to predict per frame action labels  

The results for the frame based and clip based methods are presented in the table below: 


|    method     | **top 1** | **top 3** | **macro** |  **mAP**   |
|---------------|-----------|-----------|-----------|------------|
|   ResNet18    |   27.06   |   55.14   |   21.95   |   11.69    |
|   ResNet50    |   30.38   |   56.1    |   20.03   |   9.47     |
|     C3D       |   45.73   |   69.56   |   32.48   |   21.98    |
|     P3D       | **60.4**  | **81.07** | **45.21** | **29.86**  |
|     I3D       |   57.58   |   76.72   |   39.03   |   28.77    |


The results for different views clip based I3D methods and posed based methods are presented in the table below:


| Data type| View/method    |   top1   |   top3   |  macro  |   mAP   | 
|----------|----------------|----------|----------|---------|---------|
|    RGB   |    top view    |   57.58  |   76.72  |  39.03  |  28.77  |
|    RGB   |   front view   |   60.75  |   79.3   |  42.67  |  32.73  |
|    RGB   |   side view    |   52.16  |   72.21  |  36.59  |  26.76  | 
|    RGB   | combined views |   63.24  |   80.59  |  45.38  |  32.58  |
|----------|----------------|----------|----------|---------|---------|
|   pose   |      HCN       |   39.15  |   65.37  |  28.18  |  22.32  | 
|   pose   |     ST-GCN     |   43.40  |   66.29  |  26.54  |  18.56  |
| RGB+pose |    top+HCN     |   57.77  |   76.45  |  40.00  |  29.47  |
| RGB+pose | multiview+HCN  |   64.25  |   80.58  |  46.33  |  33.08  |
|   Depth  |    top view    |   35.43  |   59.48  |  21.37  |  14.4   |
|    all   |                |   64.02  |   81.45  |  44.61  |  31.45  |

## Training and testing
Each baseline is provided in a designated directory with a `train.py` and `test.py` files. 
The test script outputs a `.json` file to `./log/model_name/results` subdirectory. 
 
Note that the input dataset for the frame based baselines is not the raw images. 
It is first resized to (480x270) and then saved in Pytorch's `ImageFolder` structure by running `shrink_images.py` and then `rearange_dataset_images_to_ImageFolder.py` scripts from the toolbox. 

`python3 ../toolbox/shrink_images.py --input_path path_to_raw_dataset --output_path path_to_smaller_dataset_destination --shrink_factor 4 --mode rgb`

`python3 ../toolbox/rearange_dataset_images_to_ImageFolder.py --dataset_path path_to_smaller_dataset_destination --output_path path_to_destination_ImageFolder`
This conversion reduces train and test time significantly. 

We provide a script for combining methods logits `combine_logits.py`. edit the `list_of_method_paths` variable
 in the script to specidy different methods directories.
 
 
 C3D original pretrained model (Sports1M) can be downloaded from: https://github.com/DavideA/c3d-pytorch
 
 P3D original pretrained model (Kinetics) can be downloaded from: https://github.com/naviocean/pseudo-3d-pytorch
 
## Evaluation
To evaluate a method run
```
python3 ./evaluate/evaluate.py --results_path ./log/model_name/results --dataset_path /path_to_dataset 
--testset_filename test_cross_env.txt 
```

This will run the evaluation metrics and  output the resutls to `scores.txt` in the resutls directory. 
