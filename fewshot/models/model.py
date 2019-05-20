# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""Model abstract class.
Author: Mengye Ren (mren@cs.toronto.edu)
"""

#TODO: complete this section.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf
from fewshot.models.nnlib import cnn, concat
from fewshot.models.measure import batch_apk, apk
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity


flags = tf.flags
flags.DEFINE_bool("allstep", False,
                  "Whether or not to average loss for all steps")
FLAGS = tf.flags.FLAGS
log = logger.get()


class Model(object):
  """Sub-classes need to implement the following
  two methods:
    1) get_inference: Given reference and candidate images, compute the logits
      of whether the candidate images are relevant.
    2) get_train_op: Given the computed logits and groundtruth mask, compute
      the loss and op to optimize the network.

  B:num classes, N:Examples per class, D:Embedding dimension, M:Number of test images, P:Num unlabelled
  """

  def __init__(self,
               config,
               nway=2,
               nshot=1,
               num_test=30,
               is_training=True,
               dtype=tf.float32):
    """Builds model."""
    self._config = config
    self._dtype = dtype
    self._nshot = FLAGS.classification_nshot
    self._nway = nway
    self._num_test = num_test
    self._is_training = is_training
    self.summaries = []         #Basic scalar summaries like losses
    self.adv_summaries = []     #More expensive/informative summaries about weights, gradients and activations
    self.inference_summaries = []
    height = config.height
    width = config.width
    channels = config.num_channel
    with tf.name_scope('Model-vars'):
      # Train images.
      self._x_train = tf.placeholder(
          dtype, [None, None, height, width, channels], name="x_train")

      # Test images.
      self._x_test = tf.placeholder(
          dtype, [None, None, height, width, channels], name="x_test")

      self._y_train = tf.placeholder(tf.int64, [None, None], name="y_train")

      # Whether the candidate is relevant.
      self._y_test = tf.placeholder(tf.int64, [None, None], name="y_test")

      self.selected_classes = tf.placeholder(tf.int64, [self._nway])

      if self._nway > 1:
        self._y_train_one_hot = tf.one_hot(self._y_train, self._nway)
        self._y_test_one_hot = tf.one_hot(self._y_test, self._nway)

    # Learning rate.
    with tf.name_scope('Model-LR'):
      self._learn_rate = tf.get_variable(
          "learn_rate", shape=[], initializer=tf.constant_initializer(0.0))
      self._new_lr = tf.placeholder(dtype, [], name="new_lr")
      self._assign_lr = tf.assign(self._learn_rate, self._new_lr)

    self._embedding_weights = None
    self.predict()
    # Output.
    with tf.name_scope('Compute-Output'):
      self.compute_output()

    if is_training:
      with tf.name_scope('Training-Op'):
        self._loss, self._train_op = self.get_train_op(self.logits, self.y_test)
        self.merged_summary = tf.summary.merge(self.summaries, 'train summaries')
        self.merged_adv_summary = tf.summary.merge(self.adv_summaries, 'Advanced summaries')
    elif len(self.inference_summaries) > 0 :
        self.inference_summary = tf.summary.merge(self.inference_summaries, 'Advanced summaries')

  def init_episode_classifier(self):
    """"
    Runs the reference and candidate images through the feature model phi.
    Computes prototypes
    h_train: [B, N, D]
    h_unlabel: [B, P, D]
    h_test: [B, M, D]
    """
    with tf.name_scope('Init-Episode_classifier'):
      with tf.name_scope('encode'):
        self._h_train, self._h_test = self.encode(self.x_train, self.x_test)
      y_train = self.y_train
      nclasses = self.nway
      jump = int(FLAGS.nshot / self._nshot)
      self._protos = self._compute_protos(nclasses, self.h_train, y_train)
      self._ssl_protos = self._protos


  def encode(self, *x_list, **kwargs):
    """
    """
    if 'ext_wts' in kwargs:
      ext_wts = kwargs['ext_wts']
    else:
      ext_wts = None
    update_batch_stats = True
    if 'update_batch_stats' in kwargs:
      update_batch_stats = kwargs['update_batch_stats']
    config = self.config
    bsize = tf.shape(self.x_train)[0]
    bsize = tf.shape(x_list[0])[0]
    num = [tf.shape(xx)[1] for xx in x_list]
    x_all = concat(x_list, 1)
    x_all = tf.reshape(x_all,
                       [-1, config.height, config.width, config.num_channel])
    h_all = self.phi(x_all, ext_wts=ext_wts, update_batch_stats=update_batch_stats)
    # tf.assert_greater(tf.reduce_mean(tf.abs(h_all)), 0.0)
    # h_all_p = self.phi(tf.random_normal(tf.shape(x_all)), ext_wts=ext_wts)
    # h_all = tf.Print(h_all, [tf.shape(h_all)], '\n-----------')
    h_all = tf.reshape(h_all, [bsize, sum(num), -1])
    h_list = tf.split(h_all, num, axis=1)
    return h_list

  def predict(self):
    """Build inference graph. To be implemented by sub models.
    Returns:
      logits: [B, M]. Logits on whether each candidate image belongs to the
        reference class.
    """
    raise NotImplemented()

  def _compute_protos(self, nclasses, h_train, y_train):
    """Computes the prototypes, cluster centers.
    Args:
      nclasses: Int. Number of classes.
      h_train: [B, N, D], Train features.
      y_train: [B, N], Train class labels.
    Returns:
      protos: [B, K, D], Test prediction.
    """
    with tf.name_scope('Compute-protos'):
      protos = [None] * nclasses
      for kk in range(nclasses):
        # [B, N, 1]
        ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
        # ksel = tf.Print(ksel, [h_train * ksel, tf.shape(ksel[:,:,0]), tf.shape(tf.boolean_mask(h_train, ksel[:,:,0]))])
        # [B, N, D]
        protos[kk] = tf.reduce_sum(h_train * ksel, [1], keep_dims=True)
        protos[kk] /= tf.reduce_sum(ksel, [1, 2], keep_dims=True)
        protos[kk] = debug_identity(protos[kk], "proto")
      protos = concat(protos, 1)  # [B, K, D]
      self.adv_summaries.append(tf.summary.histogram
                                ('Proto norms', tf.norm(tf.squeeze(protos), axis=1), family='Protos')) #TODO
      self.adv_summaries.append(tf.summary.tensor_summary('Proto norms', protos, family='Protos'))
    return protos

  def compute_cluster_compactness(self):
    raise NotImplemented()

  def compute_output(self):
    # Evaluation.
    logits = self.logits[-1]

    if self.nway > 1:
      self._prediction = tf.nn.softmax(logits)
      self._correct = tf.equal(tf.argmax(self.prediction, axis=2), self.y_test)
    else:
      self._prediction = tf.sigmoid(logits)
      self._correct = tf.equal(
          tf.cast(self.prediction > 0.5, self.dtype), self.y_test)
    self._acc = tf.reduce_mean(tf.cast(self._correct, self.dtype))

  def get_train_op(self, logits, y_test):
    """Builds optimization operation. To be implemented by sub models.
    Args:
      logits: [B, M]. Logits on whether each candidate image belongs to the
        reference class.
      y_test: [B, M, K]. Test image labels.
    Returns:
      loss: Scalar. Loss function to be optimized.
      train_op: TensorFlow operation.
    """
    raise NotImplemented()

  def phi(self, x, ext_wts=None, reuse=tf.AUTO_REUSE, update_batch_stats=False): #AYAD
    """Feature extraction function.
    Args:
      x: [N, H, W, C]. Input.
      reuse: Whether to reuse variables here.

    """
    config = self.config
    is_training = self.is_training
    dtype = self.dtype
    with tf.variable_scope("phi", reuse=reuse):
      h, wts = cnn(
          x,
          config.filter_size,
          strides=config.strides,
          pool_fn=[tf.nn.max_pool] * len(config.pool_fn),
          pool_size=config.pool_size,
          pool_strides=config.pool_strides,
          act_fn=[tf.nn.relu for aa in config.conv_act_fn],
          add_bias=True,
          init_std=config.conv_init_std,
          init_method=config.conv_init_method,
          wd=config.wd,
          dtype=dtype,
          batch_norm=True,
          is_training=is_training,
          ext_wts=ext_wts,
          update_batch_stats=update_batch_stats)
      if self._embedding_weights is None:
        self._embedding_weights = wts
      h_shape = h.get_shape()
      h_size = 1
      for ss in h_shape[1:]:
        h_size *= int(ss)
      h = tf.reshape(h, [-1, h_size])
      self.adv_summaries.append(tf.summary.histogram('Encoded', h , family="phi_out"))
    return h

  def assign_lr(self, sess, value):
    """Assign new learning rate value."""
    sess.run(self._assign_lr, feed_dict={self._new_lr: value})

  def assign_pretrained_weights(self, sess, ext_wts):
    """Load pretrained weights.
    Args:
      sess: TensorFlow session object.
      ext_wts: External weights dictionary.
    """
    assign_ops = []
    with tf.variable_scope("Model/phi/cnn", reuse=True):
      for layer in range(len(self.config.filter_size)):
        with tf.variable_scope("layer_{}".format(layer)):
          for wname1, wname2 in zip(
              ["w", "b", "ema_mean", "ema_var", "beta", "gamma"],
              ["w", "b", "emean", "evar", "beta", "gamma"]):
            assign_ops.append(
                tf.assign(
                    tf.get_variable(wname1), ext_wts["{}_{}".format(
                        wname2, layer)]))
    sess.run(assign_ops)

  @property
  def y_test(self):
    return self._y_test

  @property
  def logits(self):
    return self._logits

  @property
  def prediction(self):
    return self._prediction

  @property
  def correct(self):
    return self._correct

  @property
  def x_train(self):
    return self._x_train

  @property
  def y_train(self):
    return self._y_train

  @property
  def y_train_one_hot(self):
    return self._y_train_one_hot

  @property
  def x_test(self):
    return self._x_test

  @property
  def y_test_one_hot(self):
    return self._y_test_one_hot

  @property
  def h_train(self):
    return self._h_train

  @property
  def h_test(self):
    return self._h_test

  @property
  def protos(self):
    return self._protos

  @property
  def learn_rate(self):
    return self._learn_rate

  @property
  def config(self):
    return self._config

  @property
  def dtype(self):
    return self._dtype

  @property
  def loss(self):
    return self._loss

  @property
  def train_op(self):
    return self._train_op

  @property
  def is_training(self):
    return self._is_training

  @property
  def nshot(self):
    return self._nshot

  @property
  def nway(self):
    return self._nway

  @property
  def candidate_size(self):
    return self._candidate_size

  @property
  def embedding_weights(self):
    return self._embedding_weights

