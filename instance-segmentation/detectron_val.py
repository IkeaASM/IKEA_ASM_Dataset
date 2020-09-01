# Run evaluation
# uses default configuration from detectron2
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
setup_logger()

if __name__ == '__main__':
    register_coco_instances("ikea_val", {}, "/path/to/annotations/test_manual_coco_format_with_part_id.json", "/path/to/images")

    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "ResNeXt.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("ikea_val",)
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("ikea_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "ikea_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)
