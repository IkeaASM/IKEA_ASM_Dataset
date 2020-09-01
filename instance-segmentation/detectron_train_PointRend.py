# Run training with PointRend head
# uses default configuration from detectron2
# The model is initialized via pre-trained coco models from detectron2 model zoo
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import os
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import sys; sys.path.insert(1, "projects/PointRend")
import point_rend
from detectron2.utils.logger import setup_logger
setup_logger()

if __name__=='__main__':
    register_coco_instances("ikea_train", {}, "path/to/annotation/train_manual_coco_format.json", "/path/to/images/")

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 7

    cfg.DATASETS.TRAIN = ("ikea_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # initialize training
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.STEPS = (20000, 40000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
