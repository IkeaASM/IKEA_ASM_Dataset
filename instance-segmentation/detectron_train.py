# Run training
# uses default configuration from detectron2
# The model is initialized via pre-trained coco models from detectron2 model zoo
#
# Fatemeh Saleh <fatemehsadat.saleh@anu.edu.au>

import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()


if __name__ == '__main__':
    register_coco_instances("ikea_train", {}, "path/to/annotation/train_manual_coco_format.json", "/path/to/images/")

    cfg = get_cfg()
    # Read the config for one of the backbones (Res50, Res101, ResNeXt, ...)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("ikea_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
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
