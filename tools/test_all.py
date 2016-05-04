#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib as mpl
mpl.use('Agg')
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
           'object')
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def get_detections(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name#os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    #timer = Timer()
    #timer.tic()
    scores, boxes, pose_a, pose_e = im_detect(net, im)
    #timer.toc()
    #print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #print "a=%s, e=%s"%(5*pose_a, 5*pose_e)
    # Visualize detections for each class
    #CONF_THRESH =0.25#0.75
    #print 'threashold: {}'.format(CONF_THRESH)
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = np.hstack((cls_boxes,
                          5*pose_a[:,np.newaxis], 5*pose_e[:,np.newaxis], cls_scores[:, np.newaxis])).astype(np.float32)
        dets = dets[keep, :]
        #print "a=%s, e=%s"%(5*pose_a[keep], 5*pose_e[keep])
    return dets
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('imlist', help="Input image")
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', dest='prototxt', help='Prototxt of Network')
    parser.add_argument('--weights', dest='caffemodel', help='Weights of trained network')
    parser.add_argument('output_folder', help='Output location of image detections')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = args.prototxt#os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
               #             'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = args.caffemodel#os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                 #             NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _, _= im_detect(net, im)
    with open(args.imlist) as f:
        ims = [os.path.join('/media/dey/debidatd/pascal3d/PASCAL/VOCdevkit/VOC2012/JPEGImages', x.strip().split()[0]+'.jpg') for x in f.readlines()]
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    for i, im in enumerate(ims):
        print im
        dets = get_detections(net, im)
        with open(os.path.join(args.output_folder, str(i)+'.txt'), 'w') as f:
            for det in dets:
                f.write("%s %s %s %s %s %s %s\n"%(tuple(det)))
    #with open(os.path.join(args.output_folder, 'pred_view.txt'), 'w') as f:
    #    for i, im in enumerate(ims):
    #        print im
    #        dets = get_detections(net, im)
    #        f.write("%s %s 0\n"%(dets[0][4], dets[0][5]))
   #im_names = args.im#['000456.jpg', '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg']
    #for im_name in im_names:
    #    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    #    print 'Demo for data/demo/{}'.format(im_name)
    #    demo(net, im_name)

    #plt.show()
    #plt.savefig(args.destination)
