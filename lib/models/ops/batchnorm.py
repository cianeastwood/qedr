'''
Based on https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/batchnorm.py
'''

import numpy as np
import tensorflow as tf
import lib.models as lib

def Batchnorm(name, axes, inputs, is_training, stats_iter=None, update_moving_stats=True, fused=False, decay = 0.9,  cpu=False):
    eps = 1e-5
    if ((axes == [0,2,3]) or (axes == [0,2])) and fused==True:
        if axes==[0,2]:
            inputs = tf.expand_dims(inputs, 3)

        offset = lib.param(name+'.offset', tf.zeros(inputs.get_shape()[1]))
        scale = lib.param(name+'.scale', tf.ones(inputs.get_shape()[1]))
        moving_mean = lib.param(name+'.moving_mean', tf.zeros(inputs.get_shape()[1]), trainable=False)
        moving_variance = lib.param(name+'.moving_variance', tf.ones(inputs.get_shape()[1]), trainable=False)
        
        def train_bn():
            outputs, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=eps, data_format='NCHW')
            update_moving_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1. - decay))
            update_moving_variance = tf.assign(moving_variance, moving_variance * decay + batch_var * (1. - decay))
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(outputs)
            
        def infer_bn():
            outputs, _, _ = tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=eps,
                                                   mean=moving_mean, variance=moving_variance,
                                                   data_format='NCHW', is_training=False)
            return outputs
            
        
        outputs = tf.cond(is_training, train_bn, infer_bn)            
        
        if axes == [0,2]:
                return outputs[:,:,:,0] # collapse last dim
        return outputs
        
    else:        
        offset = lib.param(name+'.offset', tf.zeros([inputs.get_shape()[-1]]))
        scale = lib.param(name+'.scale', tf.ones([inputs.get_shape()[-1]]))
        moving_mean = lib.param(name+'.moving_mean', tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        moving_variance = lib.param(name+'.moving_variance', tf.ones([inputs.get_shape()[-1]]), trainable=False)
        
        def train_bn():
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            update_moving_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1. - decay))
            update_moving_variance = tf.assign(moving_variance, moving_variance * decay + batch_var * (1. - decay))
            
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, eps)
        
        def infer_bn():
            return tf.nn.batch_normalization(inputs, moving_mean, moving_variance, offset, scale, eps)
                
        return tf.cond(is_training, train_bn, infer_bn)
        
            