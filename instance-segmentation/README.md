# Furniture Part Instance Segmentation
### Installation
This code has been developed based on detectron2 framework. See the details for installation in [detectron2 repo](https://github.com/facebookresearch/detectron2).

### Dataset
You can download IKEA Dataset from [here]().
### Training
First, update `detectron_train.py` with the settings and parameters including dataset path and model architecture.
Then, to train a model, run 
```
python3 detectron_train.py
```
If you want to use PointRend head, run
```
python3 detectron_train_PointRend.py
```

### Evaluation
First, update `detectron_val.py` with the settings and parameters including dataset path, model architecture, and the checkpoint.
Then, to evaluate the model, run 
```
python3 detectron_val.py
```

### Inference on new images
To infer instance segmentation of new video/frames, run
```
python3 detectron_batch_inference.py --s <sample_name> \
                                     --f <furniture_type> \
                                     --root <path/to/data> \
                                     --model <checkpoint.pth> \
                                     --batch <batch_szie>
```
The output will be in a json file format.
To visualize the output, run
 ```
 python3 visualize_rgb_from_json.py --s <sample_name> \
                                    --root <path/to/data>
```
### Results 

This table is the results of the pre-trained models on the test set.
If you want to reproduce these results, use the checkpoints for [ResNet-50-FPN](https://drive.google.com/file/d/1uAHhJumAY0hJFINcu6_KujUOQ-aoaSNd/view?usp=sharing),
 [ResNet-101-FPN](https://drive.google.com/file/d/1UKVdRicQLpu15vojTwZ37qeOR8pEC0SA/view?usp=sharing), or [ResNeXt-101-FPN](https://drive.google.com/file/d/1QOzGqAWG-cedaQ1hmRL386I93GmJzxTR/view?usp=sharing).
Note that you need to place these checkpoints under `output` directory specified in the `detectron2/config/defaults.py`.

| Backbone        |Annotation type| AP  | AP50 | AP75 |table-t| leg | shelf | side-p | front-p | bottom-p | rear-p |
|-----------------|---------------|-----|------|------|-------|-----|-------|--------|---------|----------|--------|
| ResNet-50-FPN   |     mask      |56.2 | 76.9 | 62.3 | 80.4  |56.9 |  67.8 |  26.5  |   48.4  |   65.4   |  48.0  |
| ResNet-101-FPN |     mask      |60.2 | 81.2 | 66.1 | 82.1  |64.8 |  73.1 |  28.7  |   51.8  |   69.0   |  52.6  |
| ResNeXt-101-FPN |     mask      |64.1 | 85.3 | 70.6 | 88.2  |67.2 |  75.3 |  37.6  |   50.4  |   73.4   |  56.7  |
|-----------------|---------------|-----|------|------|-------|-----|-------|--------|---------|----------|--------|
| ResNet-50-FPN   |     bbox      |58.4 | 78.1 | 67.5 | 77.5  |61.3 |  64.0 |  36.2  |   58.5  |   62.4   |  49.2  |
| ResNet-101-FPN |     bbox      |63.0 | 82.1 | 71.6 | 80.2  |70.5 |  71.6 |  37.3  |   62.5  |   65.3   |  53.6  |
| ResNeXt-101-FPN |     bbox      |67.5 | 86.7 | 77.2 | 89.1  |72.9 |  73.3 |  47.8  |   64.5  |   68.1   |  56.5  |

---
*If you have any question regarding this code, please contact [fatemehsadat.saleh@anu.edu.au](mailto:fatemehsadat.saleh@anu.edu.au).*
