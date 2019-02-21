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
"""
A prototypical network for few-shot classification task.

Author: Mengye Ren (mren@cs.toronto.edu)

In a single episode, the model computes the mean representation of the positive
reference images as prototypes, and then calculates pairwise similarity in the
retrieval set. The similarity score runs through a sigmoid to give [0, 1]
prediction on whether a candidate belongs to the same class or not. The
candidates are used to backpropagate into the feature extraction CNN model phi.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
FLAGS = tf.flags.FLAGS
log = logger.get()


@RegisterModel("basic")
class BasicModel(Model):
  """A basic retrieval model that runs the images through a CNN and compute
  basic similarity scores."""

  def predict(self):
    """See `model.py` for documentation."""
    with tf.name_scope('Predict'):
      self.init_episode_classifier()
      logits = compute_logits(self.protos, self.h_test)
      self._logits = [logits]

  def compute_cluster_compactness(self, data, labels):
    pass

  def get_train_op(self, logits, y_test):
    """See `model.py` for documentation."""
    with tf.name_scope('Classification-Loss'):
      if FLAGS.allstep:
        log.info("Compute average loss for all timestep.")
        if self.nway > 1:
          loss = tf.add_n([
              tf.nn.sparse_softmax_cross_entropy_with_logits (
                  logits=ll, labels=y_test) for ll in logits
          ]) / float(len(logits))
        else:
          loss = tf.add_n([
              tf.nn.sigmoid_cross_entropy_with_logits(logits=ll, labels=y_test)
              for ll in logits
          ]) / float(len(logits))
      else:
        log.info("Compute loss for the final timestep.")
        if self.nway > 1:
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits[-1], labels=y_test)
        else:
          loss = tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits[-1], labels=y_test)
      loss = tf.reduce_mean(loss)
      self.summaries.append(tf.summary.scalar('Vanilla-loss', loss))

    with tf.name_scope('Weight-Decay'):
      wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      log.info("Weight decay variables: {}".format(wd_losses))
      if len(wd_losses) > 0:
        loss += tf.add_n(wd_losses)
      self.summaries.append(tf.summary.scalar('Regularized-loss', loss))
      #proto_norm loss
      # proto_norms = l2_norm(self.protos)
      # loss+= 0.01 * proto_norms
      # self.summaries.append(tf.summary.scalar('ProtoNorm-Loss', proto_norms))

    # loss = tf.Print(loss, [tf.shape(self.protos)])
    # cluster_distances = -l2_norm( tf.expand_dims(self.protos,1) - tf.expand_dims(self.protos, 2) )
    # cluster_variance = tf.nn.moments(self._logits)
    # loss = tf.Print(loss, [loss, cluster_distances])
    # loss += 0.05 * cluster_distances

    opt = tf.train.AdamOptimizer(self.learn_rate, name='Basic-Optimizer')
    grads_and_vars = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads_and_vars)

    for gradient, variable in grads_and_vars:
      if gradient is None:
        gradient=tf.constant(0.0)
      self.adv_summaries.append(tf.summary.scalar("gradients/" + variable.name, l2_norm(gradient), family="Grads"))
      self.adv_summaries.append(tf.summary.scalar("variables/" + variable.name, l2_norm(variable), family="VARS"))
      self.adv_summaries.append(tf.summary.histogram("gradients/" + variable.name, gradient, family="Grads"))

    return loss, train_op
