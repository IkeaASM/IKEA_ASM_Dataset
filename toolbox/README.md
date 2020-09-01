**IKEA Dataset** : toolbox
---
### Introduction
This toolbox allows to perform post-processing on the ANU IKEA dataset and generate point clouds, normal vectors, and AVI videos (rgb and depth).
It also contains scripts to make the training faster: extract the frames from the video data, shrink images, rearrange into an `ImageFolder` structure, and create acustom train-test split. 
It also contains scripts to export action ground truth from sqlite databse to .json format.

### Installation
 For the normal vectors you will need `CGAL`. (recommend to install the python binding using conda)
 
 For AVI video you will need `opencv`.
 
 For saving point clouds as `.ply` files `plyfile`
 
 For parallel processing `multiprocessing` and `joblib`
 
 For rendering the human model you will need `smplx` and `pyrender`
