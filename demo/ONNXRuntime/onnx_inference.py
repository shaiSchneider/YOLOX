#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-d",
        "--images_dir",
        type=str,
        default=None,
        help="Path to your input images directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument('--cls_names', nargs='+', type=str, default=None)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    if args.cls_names is None:
        cls_names = COCO_CLASSES
    else:
        cls_names = tuple(args.cls_names)

    input_shape = tuple(map(int, args.input_shape.split(',')))
    if args.images_dir is None:
        args.images_dir = os.path.dirname(args.image_path)
    
    for filename in tqdm(sorted(os.listdir(args.images_dir)), position=0, leave=True):
        img_path = os.path.join(args.images_dir, filename)
        if img_path.endswith('PNG') or img_path.endswith('png') or img_path.endswith('jpg') or img_path.endswith('jpeg'):
            print(img_path)
            origin_img = cv2.imread(img_path)
            img, ratio = preprocess(origin_img, input_shape)

            session = onnxruntime.InferenceSession(args.model)

            ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
            output = session.run(None, ort_inputs)
            predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                conf=args.score_thr, class_names=cls_names)

            mkdir(args.output_dir)
            output_path = os.path.join(args.output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, origin_img)
