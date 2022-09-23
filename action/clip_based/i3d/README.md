# I3D models trained on Kinetics, Charades and IKEA asm Dataset

## Overview

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.
This code is based on [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d.git) which itself is based on Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d). 

Including PyTorch versions of their models.

## Note
This code was written for Python 3.7 and PyTorch 1.3. 

## train
run `train_i3d.py -dataset_path 'path/to/ikea/dataset' -db_file 'path/to/ikea/dataset/database/with/annotations/' ` set the ikea dataset path 

## test
To get accuracy and timing run `test_i3d.py -dataset_path 'path/to/ikea/dataset' -db_file 'path/to/ikea/dataset/database/with/annotations/'`
To export a video with GT and prediction text overlay run `test_demo.py` (same input arguments as above)

## Original pytorch-i3d implementation notes: 

# Fine-tuning and Feature Extraction
They provide code to extract I3D features and fine-tune I3D for charades. Thier fine-tuned models on charades are also available in the models director (in addition to Deepmind's trained models). The deepmind pre-trained models were converted to PyTorch and give identical results (flow_imagenet.pt and rgb_imagenet.pt). These models were pretrained on imagenet and kinetics (see [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d) for details). 

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) contains the code to fine-tune I3D based on the details in the paper, fine tuned on [Charades](allenai.org/plato/charades/) dataset based on the author's implementation that won the Charades 2017 challenge. 
Their fine-tuned RGB and Flow I3D models are available in the model directory (rgb_charades.pt and flow_charades.pt).

This relied on having the optical flow and RGB frames extracted and saved as images on dist. [charades_dataset.py](charades_dataset.py) contains our code to load video segments for training.

 
