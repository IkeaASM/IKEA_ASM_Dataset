# IKEA_asm_dataset-dev

This repo contains code for the "IKEA assembly dataset". This is a dev repo, after cleanup, it will be publicly available on Github. 

[Link to raw dataset on cloudstor](https://cloudstor.aarnet.edu.au/plus/s/66mxSGe2f6ZsFut)  ~2TB

[Link to google drive video dataset](https://drive.google.com/file/d/1X0So9X_LQZQcCGC5DagMp3S1qy3I_XXn/view?usp=sharing) ~240GB

[Link to project website](https://ikeaasm.github.io/)

**The IKEA ASM Dataset**: Understanding People Assembling Furniture through Actions, Objects and Pose
---


### Introduction
This is the code for processing the IKEA assembly dataset.

This work will be presented in (insert name of conference/journal here). 

Abstract: 
The availability of a large labeled dataset is a key requirement for applying deep learning methods to solve various computer vision tasks. In the context of understanding human activities, existing public datasets, while large in size, are often limited to a single RGB camera and provide only per-frame or per-clip action annotations. To enable richer analysis and understanding of human activities, we introduce IKEA ASM---a three million frame, multi-view, furniture assembly video dataset that includes depth, atomic actions, object segmentation, and human pose. Additionally, we benchmark prominent methods for video action recognition, object segmentation and human pose estimation tasks on this challenging dataset. The dataset enables the development of holistic methods, which integrate multi-modal and multi-view data to better perform on these tasks.

### Citation
If you find this dataset useful in your research, please cite our work:
[Preprint](https://arxiv.org/abs/2007.00394):

    @article{ben2020ikea,
      title={The IKEA ASM Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose},
      author={Ben-Shabat, Yizhak and Yu, Xin and Saleh, Fatemeh Sadat and Campbell, Dylan and Rodriguez-Opazo, Cristian and Li, Hongdong and Gould, Stephen},
      journal={arXiv preprint arXiv:2007.00394},
      year={2020}
    }

### Installation
Please first download the dataset using the provided script `./scripts/download_dataset.py`.  By default it will download calibration data, indexing files data, RGB top view videos and action annotations. 
To download the full dataset run
 
`./scripts/download_dataset.py --download_all`

You can selectively download portions of the dataset (see `download_dataset.py` help).
  
Alternatively, you can manually download using the shared [GoogleDrive folder](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq?usp=sharing).

After downloading the video data, extract the individual frames using `./toolbox/extract_frames_from_videos.py`

For depenencies see Requirements.txt.

### Benchmarks
Please refer to the `README.md` file in the individual benchmark dirs for further details on training, testing and evaluating the different benchmarks (action recognition, pose estiamtion, intance segmentation, and part tracking)
Note that all pre-trained models are provided in the [GoogleDrive folder](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq?usp=sharing).

### License
Our code is released under MIT license (see license file).