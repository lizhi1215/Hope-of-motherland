# -*- coding: utf-8 -*-
# @Author: QianWang
# @Date:   2019-06-03 00:12:39
# @Last Modified time: 2019-06-03 11:16:29
import tensorflow as tf

# def unpool(input):#双线性插值
#     return tf.image.resize_bilinear(input, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

# def maxpool2d(input, k):#池化
#     return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def reduce_mean_layers(C2,C3,C4,C5):#卷积层求平均
	return 0.25*tf.add(C2,tf.add(C3,tf.add(C4,C5)))

def weight_variable(shape):#权重
    initial = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initial(shape = shape))

def bias_variable(shape):#偏差
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(input, in_features, out_features, kernel_size, with_bias = False):#卷积层
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding = 'SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv

def non_local_block(input_tensor, computation_compression = 2):
	input_shape = input_tensor.get_shape().as_list()
	batchsize, dim1, dim2, channels = input_shape
	 # Embedded Gaussian instantiation
	input_branch_left = conv2d(input_tensor, channels, channels // 2, 1)
	input_branch_left = tf.reshape(input_branch_left, shape = [-1, dim1 * dim2, channels // 2])

	input_branch_mid = conv2d(input_tensor, channels, channels // 2, 1)
	input_branch_mid = tf.reshape(input_branch_mid, shape = [-1, dim1 * dim2, channels // 2])

	if computation_compression > 1:
		input_branch_mid = tf.layers.max_pooling1d(input_branch_mid, pool_size=2, strides = computation_compression, padding = 'SAME')

	transfer_left = tf.matmul(input_branch_left, input_branch_mid, transpose_b = True)

	input_branch_mid_shape = input_branch_mid.get_shape().as_list()

	transfer_left = tf.reshape(transfer_left, shape = [-1, dim1 * dim2 * input_branch_mid_shape[1]])
	transfer_left = tf.nn.softmax(transfer_left, axis = -1)
	transfer_left = tf.reshape(transfer_left, shape = [-1, dim1 * dim2, phi_shape[1]])

	input_branch_right = conv2d(input_tensor, channels, channels // 2, 1)
	input_branch_right = tf.reshape(input_branch_right, shape = [-1, dim1 * dim2, channels // 2])
	if computation_compression > 1:
		input_branch_right = tf.layers.max_pooling1d(input_branch_right, pool_size = 2, strides = computation_compression, padding = 'SAME')

	temp_out = tf.matmul(transfer_left, input_branch_right)
	temp_out = tf.reshape(temp_out, shape = [-1, dim1, dim2, channels // 2])
	end_out = conv2d(temp_out, channels // 2, channels, kernel_size = 1)
	residual = input_tensor + end_out

	return residual
if __name__ == '__main__':
	pass