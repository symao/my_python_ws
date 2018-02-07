from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--video',
        dest='video',
        help='video file(/path/to/video.avi or 0,1,2,3 for webcam)',
        default='0',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    video_file = args.video
    if video_file.rsplit('.')[-1]!='avi':
        video_file = int(video_file)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open camera. exit. check video file:", video_file)
        exit()

    ret,img = cap.read()
    while ret:
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, img, None, timers=timers)
        # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        # for k, v in timers.items():
        #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        img = vis_utils.vis_one_image_opencv(img, cls_boxes, cls_segms, cls_keyps,
                    dataset=dummy_coco_dataset, show_class=True, show_box=True)
        cv2.imshow("detectron_result", img)
        key = cv2.waitKey(3)
        if key == 27:
            break
        ret,img = cap.read()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
