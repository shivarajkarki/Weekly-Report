# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:02:34 2019

@author: Shivaraj J Karki
@Emage Vision India
"""

""" Let's start with inception network on this sleepy afternoon """

import tensorflow as tf


def inception_module(layer_in,
                     f1,
                     f3_in,
                     f3_out,
                     f5_in,
                     f5_out,
                     fpool_out):
    
    # 1x1 layer
    conv1 = tf.keras.layers.Conv2D(filters = f1,
                                   kernel_size = (1,1),
                                   padding = 'same',
                                   activation = 'relu',
                                   kernel_initializer = tf.keras.initializers.glorot_normal,
                                   bias_initializer = tf.initializers.Constant(value = 0.2) )(layer_in)
    
    conv3 = tf.keras.layers.Conv2D(filters = f3_in,
                                  kernel_size = (1,1),
                                  padding = 'same',
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.glorot_normal,
                                  bias_initializer = tf.initializers.Constant(value = 0.2) )(layer_in)
    
    conv3 = tf.keras.layers.Conv2D(filters = f3_out,
                                  kernel_size = (3,3),
                                  padding = 'same',
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.glorot_normal,
                                  bias_initializer = tf.initializers.Constant(value = 0.2) )(conv3)
    
    conv5 = tf.keras.layers.Conv2D(filters = f5_in,
                                  kernel_size = (1,1),
                                  padding = 'same',
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.glorot_normal,
                                  bias_initializer = tf.initializers.Constant(value = 0.2) )(layer_in)
    
    conv5 = tf.keras.layers.Conv2D(filters = f5_out,
                                  kernel_size = (5,5),
                                  padding = 'same',
                                  activation = 'relu',
                                  kernel_initializer = tf.keras.initializers.glorot_normal,
                                  bias_initializer = tf.initializers.Constant(value = 0.2) )(conv5)
    
    max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3),
                                         strides = (1,1),
                                         padding = 'same')(layer_in)
    
    pool_1x1 = tf.keras.layers.Conv2D(filters = fpool_out,
                                      kernel_size = (1,1),
                                      padding = 'same',
                                      activation = 'relu',
                                      kernel_initializer = tf.keras.initializers.glorot_normal,
                                      bias_initializer = tf.initializers.Constant(value = 0.2) )(max_pool)
    
    return tf.keras.layers.Concatenate([conv1, conv3, conv5, pool_1x1], axis = -1)

