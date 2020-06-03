from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
from model.config import cfg
import numpy as np
import numpy.random as npr

def iou_select_tf(sample_labels,labels,num_fg,max_overlaps,gt_argmax_overlaps):
	for i  in range(len(labels)):
		if max_overlaps[i] <= 0.1:
			sample_labels[i] = -1
		elif 0.1 < max_overlaps[i] <= 0.2:
			sample_labels[i] = 0
		elif 0.2 < max_overlaps[i] <= 0.3:
			sample_labels[i] = 1
		elif 0.7 <= max_overlaps[i] < 0.8:
			sample_labels[i] = 3
		elif 0.8 <= max_overlaps[i] < 0.9:
			sample_labels[i] = 4
		elif 0.9 <= max_overlaps[i]:
			sample_labels[i] = 5
		else:
			continue
	sample_labels[gt_argmax_overlaps] = 3
	fg_inds = np.where(sample_labels > 2)[0]
	bg_inds = np.where(sample_labels < 2)[0]
	if len(fg_inds) + len(bg_inds) < 256:
			# print("number of pos：",len(fg_inds))
			# print("number of neg：",len(bg_inds))
			return labels
	all_inds = []
	all_inds.extend((fg_inds, bg_inds))
	all_inds_sort = np.argsort((len(fg_inds), len(bg_inds)))
	less_inds = all_inds[all_inds_sort[0]]
	more_inds = all_inds[all_inds_sort[1]]
	if len(less_inds) > math.ceil(num_fg/3) * 3:
		fg_inds_hard = np.where(sample_labels == 3)[0]#hard
		fg_inds_mid = np.where(sample_labels == 4)[0]#mid	
		fg_inds_easy = np.where(sample_labels == 5)[0]#easy
		fg_inds_list = []
		fg_inds_list.extend((fg_inds_hard, fg_inds_mid, fg_inds_easy))
		fg_sort_order = np.argsort((len(fg_inds_hard), len(fg_inds_mid), len(fg_inds_easy)))
		fg_min_num_set = fg_inds_list[fg_sort_order[0]]#short
		fg_mid_num_set = fg_inds_list[fg_sort_order[1]]
		fg_max_num_set = fg_inds_list[fg_sort_order[2]]#long
		if len(fg_min_num_set) > math.ceil(num_fg/3):
			disable_inds_min = npr.choice(fg_min_num_set, size = (len(fg_min_num_set) - math.floor(num_fg/3)), replace = False)
			disable_inds_mid = npr.choice(fg_mid_num_set, size = (len(fg_mid_num_set) - math.ceil(num_fg/3)), replace = False)
			disable_inds_max = npr.choice(fg_max_num_set, size = (len(fg_max_num_set) - (num_fg-math.floor(num_fg/3) - math.ceil(num_fg/3))), replace = False)
			disable_inds = np.concatenate((disable_inds_min, disable_inds_mid, disable_inds_max))
			labels[disable_inds] = -1
		else:
			num_diff = num_fg - len(fg_min_num_set)
			if len(fg_mid_num_set) > math.ceil(num_diff/2):
				disable_inds_mid = npr.choice(fg_mid_num_set, size=(len(fg_mid_num_set) - math.floor(num_diff/2)), replace = False)
				disable_inds_max = npr.choice(fg_max_num_set, size=(len(fg_max_num_set) - (num_diff - math.floor(num_diff/2))), replace = False)
				disable_inds = np.concatenate((disable_inds_mid, disable_inds_max), axis = 0)
				labels[disable_inds] = -1
			else:
				num_diff_last = num_diff - len(fg_mid_num_set)
				disable_inds_max = npr.choice(fg_max_num_set, size=(len(fg_max_num_set) - num_diff_last), replace=False)
				labels[disable_inds_max] = -1

		num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)

		bg_inds_hard = np.where(sample_labels == 1)[0]
		bg_inds_mid = np.where(sample_labels == 0)[0]
		bg_inds_easy = np.where(sample_labels == -1)[0]
		bg_inds_list = []
		bg_inds_list.extend((bg_inds_hard, bg_inds_mid, bg_inds_easy))
		bg_sort_order = np.argsort((len(bg_inds_hard), len(bg_inds_mid), len(bg_inds_easy)))
		bg_min_num_set = bg_inds_list[bg_sort_order[0]]
		bg_mid_num_set = bg_inds_list[bg_sort_order[1]]
		bg_max_num_set = bg_inds_list[bg_sort_order[2]]

		if len(bg_min_num_set) > math.ceil(num_bg/3):
			disable_inds_min = npr.choice(bg_min_num_set, size = (len(bg_min_num_set) - math.floor(num_bg/3)), replace = False)
			disable_inds_mid = npr.choice(bg_mid_num_set, size = (len(bg_mid_num_set) - math.ceil(num_bg/3)), replace = False)
			disable_inds_max = npr.choice(bg_max_num_set, size = (len(bg_max_num_set) - (num_bg - math.floor(num_bg/3) - math.ceil(num_bg/3))), replace = False)
			disable_inds = np.concatenate((disable_inds_min,disable_inds_mid,disable_inds_max))
			labels[disable_inds] = -1
		else:
			num_diff = num_bg - len(bg_min_num_set)
			if len(bg_mid_num_set) > math.ceil(num_diff/2):
				disable_inds_mid = npr.choice(bg_mid_num_set, size = (len(bg_mid_num_set) - math.floor(num_diff/2)), replace = False)
				disable_inds_max = npr.choice(bg_max_num_set, size = (len(bg_max_num_set) - (num_diff - math.floor(num_diff/2))), replace = False)
				disable_inds = np.concatenate((disable_inds_mid, disable_inds_max), axis = 0)
				labels[disable_inds] = -1
			else:
				num_diff_last = num_diff - len(bg_mid_num_set)
				disable_inds_max = npr.choice(bg_max_num_set, size=(len(bg_max_num_set) - num_diff_last), replace=False)
				labels[disable_inds_max] = -1
	else:
		if len(less_inds) == len(fg_inds):
			num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
			bg_inds_hard = np.where(sample_labels == 1)[0]
			bg_inds_mid = np.where(sample_labels == 0)[0]
			bg_inds_easy = np.where(sample_labels == -1)[0]
			bg_inds_list = []
			bg_inds_list.extend((bg_inds_hard, bg_inds_mid, bg_inds_easy))
			bg_sort_order = np.argsort((len(bg_inds_hard), len(bg_inds_mid), len(bg_inds_easy)))
			bg_min_num_set = bg_inds_list[bg_sort_order[0]]
			bg_mid_num_set = bg_inds_list[bg_sort_order[1]]
			bg_max_num_set = bg_inds_list[bg_sort_order[2]]

			if len(bg_min_num_set) > math.ceil(num_bg/3):
				disable_inds_min = npr.choice(bg_min_num_set, size = (len(bg_min_num_set) - math.floor(num_bg/3)), replace = False)
				disable_inds_mid = npr.choice(bg_mid_num_set, size = (len(bg_mid_num_set) - math.ceil(num_bg/3)), replace = False)
				disable_inds_max = npr.choice(bg_max_num_set, size = (len(bg_max_num_set) - (num_bg - math.floor(num_bg/3) - math.ceil(num_bg/3))), replace = False)
				disable_inds = np.concatenate((disable_inds_min, disable_inds_mid, disable_inds_max))
				labels[disable_inds] = -1

			else:
				num_diff = num_bg - len(bg_min_num_set)
				if len(bg_mid_num_set) > math.ceil(num_diff/2):
					disable_inds_mid = npr.choice(bg_mid_num_set, size = (len(bg_mid_num_set) - math.floor(num_diff/2)), replace = False)
					disable_inds_max = npr.choice(bg_max_num_set, size = (len(bg_max_num_set) - (num_diff - math.floor(num_diff/2))), replace = False)
					disable_inds = np.concatenate((disable_inds_mid, disable_inds_max), axis = 0)
					labels[disable_inds] = -1
				else:
					num_diff_last = num_diff - len(bg_mid_num_set)
					disable_inds_max = npr.choice(bg_max_num_set, size = (len(bg_max_num_set) - num_diff_last), replace = False)
					labels[disable_inds_max] = -1
		else:
			num_fg = num_fg - len(less_inds)
			fg_inds_hard = np.where(sample_labels == 3)[0]
			fg_inds_mid = np.where(sample_labels == 4)[0]
			fg_inds_easy = np.where(sample_labels == 5)[0]
			fg_inds_list = []
			fg_inds_list.extend((fg_inds_hard, fg_inds_mid, fg_inds_easy))
			fg_sort_order = np.argsort((len(fg_inds_hard),len(fg_inds_mid), len(fg_inds_easy)))
			fg_min_num_set = fg_inds_list[fg_sort_order[0]]
			fg_mid_num_set = fg_inds_list[fg_sort_order[1]]
			fg_max_num_set = fg_inds_list[fg_sort_order[2]]
			if len(fg_min_num_set) > math.ceil(num_fg/3):
				disable_inds_min = npr.choice(fg_min_num_set, size = (len(fg_min_num_set) - math.floor(num_fg/3)), replace = False)
				disable_inds_mid = npr.choice(fg_mid_num_set, size = (len(fg_mid_num_set) - math.ceil(num_fg/3)), replace = False)
				disable_inds_max = npr.choice(fg_max_num_set, size = (len(fg_max_num_set) - (num_fg - math.floor(num_fg/3) - math.ceil(num_fg/3))), replace = False)
				disable_inds = np.concatenate((disable_inds_min, disable_inds_mid, disable_inds_max), axis = 0)
				labels[disable_inds] = -1
			else:
				num_diff = num_fg - len(fg_min_num_set)
				if len(fg_mid_num_set) > math.ceil(num_diff/2):
					disable_inds_mid = npr.choice(fg_mid_num_set, size = (len(fg_mid_num_set) - math.floor(num_diff/2)), replace = False)
					disable_inds_max = npr.choice(fg_max_num_set, size = (len(fg_max_num_set) - (num_diff - math.floor(num_diff/2))), replace = False)
					disable_inds = np.concatenate((disable_inds_mid,disable_inds_max))
					labels[disable_inds] = -1
				else:
					num_diff_last = num_diff - len(fg_mid_num_set)
					disable_inds_max = npr.choice(fg_max_num_set, size = (len(fg_max_num_set) - num_diff_last), replace = False)
					labels[disable_inds_max] = -1
	return labels

if __name__ == '__main__':
	pass