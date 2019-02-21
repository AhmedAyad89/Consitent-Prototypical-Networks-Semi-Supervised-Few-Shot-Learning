from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.basic_model_VAT_ENT import BasicModelVAT_ENT
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
FLAGS = tf.flags.FLAGS
log = logger.get()

def construct_proto(points, num_points):
	t1 = tf.expand_dims(points, 0)
	t2 = tf.expand_dims(points, 1)
	pairwise_min =  tf.minimum(t1, t2)
	# pairwise_min = tf.Print(pairwise_min, [pairwise_min, tf.shape(pairwise_min)], summarize=10)
	mean=0
	for i in range(num_points):
		for j in range(i+1, num_points):
			mean += pairwise_min[i,j,:]

	proto = mean / ((num_points * num_points-1)/ 2)
	proto = tf.expand_dims(proto, 0)
	proto = tf.expand_dims(proto, 0)
	# proto = tf.reduce_mean(pairwise_min, [0,1], keep_dims=True)

	return proto

@RegisterModel("basic-pairwise")
class PairwiseModel(RefineModel):
	def _compute_protos(self, nclasses, h_train, y_train):
		num_points = self.nshot
		with tf.name_scope('Compute-protos'):
			protos = [None] * nclasses
			for kk in range(nclasses):
				# [B, N, 1]
				ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
				protos[kk] = construct_proto(tf.boolean_mask(h_train, ksel[:,:,0]), num_points)
			protos = concat(protos, 1)  # [B, K, D]

			return protos

@RegisterModel("pairwise-VAT-ENT")
class PairwiseModelVAT_ENT(BasicModelVAT_ENT, PairwiseModel):
	def get_train_op(self, logits, y_test):
		BasicModelVAT_ENT.get_train_op(self, logits, y_test)

	def _compute_protos(self, nclasses, h_train, y_train):
		PairwiseModel._compute_protos(self, nclasses, h_train, y_train)