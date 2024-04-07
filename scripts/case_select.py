# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-9-31
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from collections import defaultdict
import numpy as np

object_scores = np.load('experiments/mlic_object_mscoco/exp1/scores.npy')
pixel_scores = np.load('experiments/mlic_pixel_mscoco/exp2/scores.npy')
both_scores = np.load('experiments/mlic5_mscoco/exp178/scores.npy')
labels = [t.strip() for t in open('data/mscoco/label.txt')]
data = [t.strip().split('\t') for t in open('data/mscoco/test.txt')]

for i in range(object_scores.shape[0]):
    object_inds = np.nonzero(object_scores[i] > 0.5)[0]
    pixel_inds = np.nonzero(pixel_scores[i] > 0.5)[0]
    both_inds = np.nonzero(both_scores[i] > 0.5)[0]
    img_path, gt_labels = data[i][0], data[i][1].split(',')
    # if len(both_inds) != 3 or len(gt_labels) != 3 or len(set([labels[t] for t in both_inds]) - set(gt_labels)) != 0:
    #     continue
    # if len(object_inds) != 4 or len(pixel_inds) != 4:
        # continue
    # if len(both_inds) != 4 or len(gt_labels) != 4 or len(set([labels[t] for t in both_inds]) - set(gt_labels)) != 0:
    #     continue
    # if len(object_inds) > 3 or len(pixel_inds) > 3 or len(set(object_inds) - set(pixel_inds)) == 0:
    #     continue
    if len(both_inds) != 3 or len(gt_labels) != 3 or len(set([labels[t] for t in both_inds]) - set(gt_labels)) != 0:
        continue
    # if len(object_inds) > 2 or len(pixel_inds) != 4:
    #     continue
    if len(object_inds) !=4 or len(pixel_inds) > 2:
        continue
    print(img_path)
    print([labels[t] for t in object_inds])
    print([labels[t] for t in pixel_inds])
    print([labels[t] for t in both_inds])
    print('==='*30)
