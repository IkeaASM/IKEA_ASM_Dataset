## Tracking IKEA Parts
We consider that we have access to the results of instance segmentation for a video sample. For example `0007_oak_floor_01_01_2019_08_14_17_17.json` is available.

1. ``` python extract_dets_from_json.py ``` converts the json format to the proper format for tracking algorithm which is: 
`<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>`

2. ``` python sort_bbox.py ``` runs the SORT algorithm on the video sample.

3. ``` python3 visualize_tracker.py ``` visualizes the result of tracking algorithm.

## Evaluating the tracking results
The dataset has the tracking ground-truth for 1% of the frames of each video sample for the test set. In each sample directory under /dev3 directory, we have a file named `tracking_gt.txt`

For evaluating the tracking algorithm, first we run ```python extract_tracking_res_1_percent.py``` on the output of step (2) to extract the results for the same 1% frames that we have the ground-truth for.

Then, run ```python eval.py``` to get the quantitative results in terms of standard MOT metrics.


SORT
=====

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4).
have
By Alex Bewley  

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in an [arXiv tech report](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

### License

SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements. If you require a permissive license contact Alex (alex@bewley.ai).

### Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }


### Dependencies:

This code makes use of the following packages:
1. [`scikit-learn`](http://scikit-learn.org/stable/)
0. [`scikit-image`](http://scikit-image.org/download)
0. [`FilterPy`](https://github.com/rlabbe/filterpy)

---
*If you have any question regarding this code, please contact [fatemehsadat.saleh@anu.edu.au](mailto:fatemehsadat.saleh@anu.edu.au).*
