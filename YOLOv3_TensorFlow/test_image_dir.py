# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test image test procedure.")
parser.add_argument("--image_dir", type=str, default="./data/demo_image/",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/data.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/new_best_model2_18.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

img_dir = args.image_dir
img_list = os.listdir(img_dir)

os.makedirs("./data/demo_image_results", exist_ok=True)



with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.35, nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    for img_file in img_list:
        print(img_file)
        img_name = img_file.split('.')[0]
        img_type = img_file.split('.')[1]

        img_ori = cv2.imread(img_dir + img_file)
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.


        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        img_height, img_width, _ = img_ori.shape

        # if (img_height > 1000 or img_width > 1000):
        #     mosaic_rate = 20
        # # elif (img_height < 500 or img_width < 500):
        # #     mosaic_rate = 10
        # else:
        #     mosaic_rate = 10

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]

            x0 = int(round(x0))
            x1 = int(round(x1))
            y0 = int(round(y0))
            y1 = int(round(y1))

            # print(x0, y0, x1, y1)

            if ((x1 - x0) >= 100 or (y1 - y0) >= 100):
                mosaic_rate = 20
            elif ((x1 - x0) < 10 or (y1 - y0) < 10):
                mosaic_rate = 2
            else:
                mosaic_rate = 10

            box_w = x1 - x0
            box_h = y1 - y0

            # print(box_w, box_h)

            selected_area = img_ori[y0:y1, x0:x1]

            conv_box_w = box_w // mosaic_rate
            conv_box_h = box_h // mosaic_rate
            if (conv_box_w > 0 and conv_box_h > 0 and selected_area.shape[0] > 0 and selected_area.shape[1] > 0):

                mosaic_img = cv2.resize(selected_area,
                                        (box_w//mosaic_rate, box_h//mosaic_rate))
                mosaic_img = cv2.resize(mosaic_img,
                                        (selected_area.shape[1], selected_area.shape[0]),
                                        interpolation=cv2.INTER_AREA)

                img_ori[y0:y1, x0:x1] = mosaic_img

            # plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
        cv2.imwrite('./data/demo_image_results/{}.{}'.format(img_name, img_type), img_ori)
        cv2.waitKey(0)
