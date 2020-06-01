import numpy as np
import tensorflow as tf

def iou_select(crowd_overlaps,overlaps):
	overlaps = overlaps_graph(proposals, gt_boxes)
	crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
	crowd_iou_max = tf.reduce_max(crowd_overlaps, axis = 1)
	roi_iou_max = tf.reduce_max(overlaps, axis = 1)
	no_crowd_bool = (crowd_iou_max < 0.001)
	#分段求各位置的bool
	positive_roi_bool_hard = tf.logical_and(roi_iou_max >= 0.5,roi_iou_max <= 0.66)
	positive_roi_bool_mid = tf.logical_and(roi_iou_max > 0.66,roi_iou_max <= 0.83)
	positive_roi_bool_easy = (roi_iou_max > 0.83)
	#根据分段的bool，获取索引
	positive_indices_hard = tf.where(positive_roi_bool_hard)[:,0]
	positive_indices_mid = tf.where(positive_roi_bool_mid)[:,0]
	positive_indices_easy = tf.where(positive_roi_bool_easy)[:,0]
	#分局获取的索引，求得每段的长度
	positive_hard_num = positive_indices_hard.shape[0]
	positive_mid_num = positive_indices_mid.shape[0]
	positive_easy_num = positive_indices_easy.shape[0]

	positive_count = int(config.TRAIN_ROIS_PER_IMAGE *#TRAIN_ROIS_PER_IMAGE = 200
	                         config.ROI_POSITIVE_RATIO)#ROI_POSITIVE_RATIO=0.33

	#如果正样本的总数大于positive_count的话，则进行抽样
	if (positive_hard_num + positive_mid_num + positive_easy_num) > positive_count:
		# positive_count = 66
		#把索引放入一个列表中，方便排序和取出
		positive_inds_list = []
		positive_inds_list.extend((positive_indices_hard,positive_indices_mid,positive_indices_easy))
		#根据每段的长度，进行排序，短、中、长的顺序
		positive_sort_order = np.argsort((positive_hard_num,positive_mid_num,positive_easy_num))
		positive_min_num_set = positive_inds_list[positive_sort_order[0]]#最短数组
		positive_mid_num_set = positive_inds_list[positive_sort_order[1]]#中间数组
		positive_max_num_set = positive_inds_list[positive_sort_order[2]]#最长数组
		
		if len(positive_min_num_set) > positive_count/3:#若最短的数组都大于22，则每个数组抽三分之一即可
			positive_inds_min = np.random.choice(positive_min_num_set, size=positive_count/3, replace=False)
			positive_inds_mid = np.random.choice(positive_mid_num_set, size=positive_count/3, replace=False)
			positive_inds_max = np.random.choice(positive_max_num_set, size=positive_count - positive_count/3-\
				positive_count/3, replace=False)
			positive_indices = np.concatenaet((positive_inds_min,positive_inds_mid,positive_inds_max))
		else:#若最短数组小于22，则需要补66-x个
			positive_diff = positive_count - len(positive_min_num_set)
			if len(positive_mid_num_set) > positive_diff/2:#中等数组和最长数组，平均抽剩余的一半num_diff/2，如果中间数组
			#长度大于这个数，则继续
				positive_inds_mid = np.random.choice(positive_mid_num_set, size = (len(positive_mid_num_set) -\
				 positive_diff/2), replace = False)
				positive_inds_max = np.random.choice(positive_max_num_set, size= (len(positive_max_num_set) - \
					(positive_diff - positive_diff/2)), replace = False)
				positive_indices = np.concatenaet((positive_min_num_set,positive_inds_mid,positive_inds_max))
			else:
				positive_diff_last = positive_diff - len(positive_mid_num_set)
				positive_inds_max = np.random.choice(positive_max_num_set, size= (len(positive_max_num_set) - \
					positive_diff_last), replace=False)
				positive_indices = np.concatenaet((positive_min_num_set,positive_mid_num_set,positive_inds_max))
	else:
		positive_indices = np.concatenaet((positive_indices_hard,positive_indices_mid,positive_indices_easy))

	positive_count = tf.shape(positive_indices)[0]
	r = 1.0/0.33
	# r = 1.0 / config.ROI_POSITIVE_RATIO
	negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count

	negative_roi_bool_hard = tf.logical_and(tf.logical_and(roi_iou_max < 0.5,roi_iou_max > 0.33),no_crowd_bool)
	negative_roi_bool_mid = tf.logical_and(tf.logical_and(roi_iou_max <= 0.33,roi_iou_max > 0.16),no_crowd_bool)
	negative_roi_bool_easy = tf.logical_and(roi_iou_max <= 0.16,no_crowd_bool)

	negative_indices_hard = tf.where(negative_roi_bool_hard)[:,0]
	negative_indices_mid = tf.where(negative_roi_bool_mid)[:,0]
	negative_indices_easy = tf.where(negative_roi_bool_easy)[:,0]

	negative_hard_num = negative_indices_hard.shape[0]
	negative_mid_num = negative_indices_mid.shape[0]
	negative_easy_num = negative_indices_easy.shape[0]

	if (negative_hard_num + negative_mid_num + negative_easy_num) > negative_count:
		negative_inds_list = []
		negative_inds_list.extend((negative_indices_hard,negative_indices_mid,negative_indices_easy))
		#根据每段的长度，进行排序，短、中、长的顺序
		negative_sort_order = np.argsort((negative_hard_num,negative_mid_num,negative_easy_num))
		negative_min_num_set = negative_inds_list[nagetive_sort_order[0]]#最短数组
		negative_mid_num_set = negative_inds_list[negative_sort_order[1]]#中间数组
		negative_max_num_set = negative_inds_list[negative_sort_order[2]]#最长数组

		if len(negative_min_num_set) > negative_count/3:#若最短的数组都大于negative需要补的三分之一，则每个数组抽三分之一即可
			negative_inds_min = np.random.choice(negative_min_num_set, size = negative_count/3, replace = False)
			negative_inds_mid = np.random.choice(negative_mid_num_set, size = negative_count/3, replace = False)
			negative_inds_max = np.random.choice(negative_max_num_set, size = negative_count - negative_count\
				/3 - negative_count/3, replace = False)
			negative_indices = np.concatenaet((negative_inds_min,negative_inds_mid,negative_inds_max))
		else:#若最短数组小于三分之一，则需要补总数减去最少的长度得到的个数
			negative_diff = negative_count - len(negative_min_num_set)
			if len(negative_mid_num_set) > negative_diff/2:#中等数组和最长数组，平均抽剩余的一半num_diff/2，如果中间数组
			#长度大于这个数，则继续
				negative_inds_mid = np.random.choice(negative_mid_num_set, size = (len(negative_mid_num_set) - \
					negative_diff/2), replace=False)
				negative_inds_max = np.random.choice(negative_max_num_set, size = (len(negative_max_num_set) - \
					(negative_diff - negative_diff/2)), replace = False)
				negative_indices = np.concatenaet((negative_min_num_set,negative_inds_mid,negative_inds_max))
			else:
				negative_diff_last = negative_diff - len(negative_mid_num_set)
				negative_inds_max = np.random.choice(negative_max_num_set, size = (len(negative_max_num_set) - \
					negative_diff_last), replace = False)
				negative_indices = np.concatenaet((negative_min_num_set,negative_mid_num_set,negative_inds_max))
	else:
		negative_indices = np.concatenaet((negative_indices_hard,negative_indices_mid,negative_indices_easy))

	return positive_indices,negative_indices

if __name__ == '__main__':
	pass