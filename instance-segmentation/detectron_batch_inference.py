# Run inference on batches of data
# uses default configuration from detectron2
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import numpy as np
import cv2
import os
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch
import detectron2.data.transforms as T
from detectron2.checkpoint import  DetectionCheckpointer
import glob
import json
from skimage import measure
from pycocotools import mask
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', help='name of sample such as, [0002_white_floor_05_02_2019_08_19_17_47]', required=True)
parser.add_argument('-f', help='name of furniture from [Kallax_Shelf_Drawer, Lack_Side_Table, Lack_Coffee_Table, Lack_TV_Bench]', required=True)
parser.add_argument('-root', default='/path/to/dataset/', required=True)
parser.add_argument('-batch', default=10, required=True, type=int)
parser.add_argument('-model', help='name of the model in cfg.OUTPUT_DIR such as [ResNeXt.pth] ', required=True)

args = parser.parse_args()

if __name__ == '__main__':

    test_dir = os.path.join(args.root, args.f, args.s, 'dev3', 'images')
    test_imgs = glob.glob(test_dir + '/*')
    test_imgs.sort()
    test_imgs_transform = []
    test_json = {'images': [], 'annotations': [], 'categories': []}

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.train(False)

    for i in range(len(test_imgs)):
        original_image = cv2.imread(test_imgs[i])
        height, width = original_image.shape[:2]
        image = transform_gen.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        input = {"image": image, "height": height, "width": width}
        test_imgs_transform.append(input)
        test_json['images'].append(
            {"id": i + 1, "file_name": args.f + '/' + args.s + '/dev3/images/' + str.split(test_imgs[i], '/')[-1],
             "height": height, "width": width})

    segment_id = 1
    start = time.time()
    with torch.no_grad():
        for b in range(0, len(test_imgs_transform), args.batch):
            if b + args.batch < len(test_imgs_transform):
                outputs = model(test_imgs_transform[b:b + args.batch])
                for j in range(args.batch):
                    for s in range(len(outputs[j]['instances'])):
                        bimask = outputs[j]['instances'][s].pred_masks.cpu().numpy().astype("uint8")[0]
                        segment = []
                        contours = measure.find_contours(bimask, 0.5)
                        for contour in contours:
                            contour = np.flip(contour, axis=1)
                            segmentation = contour.ravel().tolist()
                            segment.append(segmentation)
                        bimask = np.expand_dims(bimask, axis=-1)
                        Rs = mask.encode(np.asfortranarray(np.uint8(bimask)))
                        area = int(mask.area(Rs)[0]),
                        bbox_seg = mask.toBbox(Rs)[0].tolist()
                        test_json['annotations'].append(
                            {'image_id': int(b + j + 1), 'id': segment_id, 'segmentation': segment,
                             'bbox': outputs[j]['instances'][s].pred_boxes.tensor.cpu().numpy()[0].tolist()
                                , 'category_id': int(outputs[j]['instances'][s].pred_classes.cpu().numpy()[0]) + 1,
                             'score': float(outputs[j]['instances'][s].scores.cpu().numpy()[0]), 'area': int(area[0])})
                        segment_id += 1
            print("Inference Done for ", b, "Frames")
        if b < len(test_imgs_transform):
            outputs = model(test_imgs_transform[b:])
            for j in range(len(test_imgs_transform) - b):
                for s in range(len(outputs[j]['instances'])):
                    bimask = outputs[j]['instances'][s].pred_masks.cpu().numpy().astype("uint8")[0]
                    segment = []
                    contours = measure.find_contours(bimask, 0.5)
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        segmentation = contour.ravel().tolist()
                        segment.append(segmentation)
                    bimask = np.expand_dims(bimask, axis=-1)
                    Rs = mask.encode(np.asfortranarray(np.uint8(bimask)))
                    area = int(mask.area(Rs)[0]),
                    bbox_seg = mask.toBbox(Rs)[0].tolist()
                    test_json['annotations'].append(
                        {'image_id': int(b + j + 1), 'id': segment_id, 'segmentation': segment,
                         'bbox': outputs[j]['instances'][s].pred_boxes.tensor.cpu().numpy()[0].tolist()
                            , 'category_id': int(outputs[j]['instances'][s].pred_classes.cpu().numpy()[0]) + 1,
                         'score': float(outputs[j]['instances'][s].scores.cpu().numpy()[0]), 'area': int(area[0])})
                    segment_id += 1
    print('Total inference time for ', len(test_imgs_transform), ' frames:', time.time()-start)
    test_json['categories'] =[{'id': 1, 'name': 'table_top'}, {'id': 2, 'name': 'leg'}, {'id': 3, 'name': 'shelf'},
                          {'id': 4, 'name': 'side_panel'}, {'id': 5, 'name': 'front_panel'}, {'id': 6, 'name': 'bottom_panel'},
                          {'id': 7, 'name': 'rear_panel'}]
    with open(os.path.join(cfg.OUTPUT_DIR, args.s + '.json'), 'w') as outfile:
        json.dump(test_json, outfile)
        print('Output json file successfully created!')
