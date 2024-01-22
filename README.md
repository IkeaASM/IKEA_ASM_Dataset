# IKEA Assembly Dataset

This repo contains code for the "IKEA assembly dataset". This is a dev repo, after cleanup, it will be publicly available on Github. 


[Link to google drive video dataset](https://drive.google.com/file/d/1X0So9X_LQZQcCGC5DagMp3S1qy3I_XXn/view?usp=sharing) ~240GB

[Link to project website](https://ikeaasm.github.io/)

**The IKEA ASM Dataset**: Understanding People Assembling Furniture through Actions, Objects and Pose
---


### Introduction
This is the code for processing the IKEA assembly dataset.

This work will be presented in WACV 2021. 

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

[WACV2021](https://openaccess.thecvf.com/content/WACV2021/html/Ben-Shabat_The_IKEA_ASM_Dataset_Understanding_People_Assembling_Furniture_Through_Actions_WACV_2021_paper.html): 

    @inproceedings{ben2021ikea,
      title={The ikea asm dataset: Understanding people assembling furniture through actions, objects and pose},
      author={Ben-Shabat, Yizhak and Yu, Xin and Saleh, Fatemeh and Campbell, Dylan and Rodriguez-Opazo, Cristian and Li, Hongdong and Gould, Stephen},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      pages={847--859},
      year={2021}
    }
    
### Installation
Please first download the dataset using the provided links: 
[Full dataset download](https://drive.google.com/drive/folders/1xkDp--QuUVxgl4oJjhCDb2FWNZTkYANq?usp=sharing)

Alternatively, you can download only the relevant parts:  
* [utility files](https://drive.google.com/file/d/11D7d8XBRg-CPIxMroviQEaaMhw3EaGnB/view?usp=sharing)
* [camera parameters](https://drive.google.com/file/d/1BRq9HJQeEJFbhnCwGwY3eXe1587TybCe/view?usp=sharing)
* [Data - RGB top view](https://drive.google.com/file/d/1CFOH-W-6N50AVA_NqHnm06GUsfpcka0L/view?usp=sharing)
* [Data - RGB multi-view](https://drive.google.com/file/d/1eCbrIuw--16xCmI3RtBhRJ-r9K_FVkL6/view?usp=sharing)
* [Data - Depth](https://drive.google.com/file/d/18FKRSzoUiO3EV_J2WmQyvmPGiHJcH28S/view?usp=sharing)
* [Annotations - action](https://drive.google.com/file/d/1SwBNLViktSpk99jhh3sMXVGTMVr6tpju/view?usp=sharing)
* [Annotations - pose](https://drive.google.com/file/d/1RE7Ya1gwogqJtJIi5WeYOH4_Cs1RuTx7/view?usp=sharing)
* [Annotations - segmetnation and tracking](https://drive.google.com/file/d/1_jRCcLAz9zhXTnNnslBUJcu2sZjp9dVV/view?usp=sharing)
* [Pretrained models - action](https://drive.google.com/file/d/1QksK_Uvty6pTYoGmBGWYYG3scvM_NX2X/view?usp=sharing)
* [Pretrained models - pose](https://drive.google.com/file/d/1SMoYC-PTHr6Y2StKKT8j_-gSYcwhTHKb/view?usp=sharing)
* [Pretrained models - segmentation and tracking](https://drive.google.com/file/d/1lLNiWU6ILFCgg104FDwWvRMV0iQaGKyp/view?usp=sharing)

 
After downloading the video data, extract the individual frames using `./toolbox/extract_frames_from_videos.py`
For further processing of the data refer to the individual benchmarks `README.md` files.

For depenencies see `requirements.txt`.

### Benchmarks
We provide several benchmarks: 
* Action recognition
* Pose Estimation
* Part segmentation and tracking

Please refer to the `README.md` file in the individual benchmark dirs for further details on training, testing and evaluating the different benchmarks (action recognition, pose estiamtion, intance segmentation, and part tracking).
Make sure to download the relevant pretrained models from the links above.

### License
Our code is released under MIT license (see `LICENCE.txt` file).
