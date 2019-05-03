from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model

from fewshot.models.refine_model import RefineModel
from fewshot.models.basic_model_VAT import BasicModelVAT
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *


l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()


@RegisterModel("basic-VAT-ENT")
class BasicModelVAT_ENT(BasicModelVAT):
	def get_train_op(self, logits, y_test):
		loss, train_op = BasicModelVAT.get_train_op(self, logits, y_test)
		config = self.config
		ENT_weight = config.ENT_weight
		VAT_ENT_step_size = config.VAT_ENT_step_size

		logits = self._unlabel_logits

		s = tf.shape(logits)
		s = s[0]
		p = tf.stop_gradient(self.h_unlabel)
		affinity_matrix = compute_logits(p, p) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		# logits = tf.Print(logits, [tf.shape(point_logits)])

		ENT_loss = walking_penalty(logits, affinity_matrix)
		loss += ENT_weight * ENT_loss

		ENT_opt = tf.train.AdamOptimizer(VAT_ENT_step_size * self.learn_rate, name="Entropy-optimizer")
		ENT_grads_and_vars = ENT_opt.compute_gradients(loss)
		train_op = ENT_opt.apply_gradients(ENT_grads_and_vars)

		for gradient, variable in ENT_grads_and_vars:
			if gradient is None:
				gradient = tf.constant(0.0)
			self.adv_summaries.append(tf.summary.scalar("ENT/gradients/" + variable.name, l2_norm(gradient), family="Grads"))
			self.adv_summaries.append(tf.summary.histogram("ENT/gradients/" + variable.name, gradient, family="Grads"))

		self.summaries.append(tf.summary.scalar('entropy loss', ENT_loss))

		return loss, train_op


#################################################################################

@RegisterModel("basic-ENT")
class BasicModelENT(RefineModel):
	def predict(self):
		super(RefineModel, self).predict()
		with tf.name_scope('Predict/VAT'):
			self._unlabel_logits = compute_logits(self._ssl_protos, self.h_unlabel)


	def get_train_op(self, logits, y_test):
		loss, train_op = RefineModel.get_train_op(self, logits, y_test)
		config = self.config
		ENT_weight = config.ENT_weight
		ENT_step_size = config.ENT_step_size

		logits = self._unlabel_logits

		s = tf.shape(logits)
		s = s[0]

		if (config.stop_grad_unlbl):
			p = tf.stop_gradient(self.h_unlabel)
		else:
			p = self.h_unlabel
		affinity_matrix = compute_logits(p, p) - (tf.eye(s, dtype=tf.float32) * 1000.0)

		s = tf.shape(self._logits[0][0])
		s = s[0]
		if (config.stop_grad_lbl):
			p = tf.stop_gradient(self.h_test[0])
		else:
			p = self.h_test[0]
		labeled_affinity_matrix = compute_logits(p, p) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		labeled_logits = self._logits[0][0]

		ENT_loss = walking_penalty_matching(logits, affinity_matrix, labeled_logits, labeled_affinity_matrix)
		loss += ENT_weight * ENT_loss

		ENT_opt = tf.train.AdamOptimizer(ENT_step_size * self.learn_rate, name="Entropy-optimizer")
		ENT_grads_and_vars = ENT_opt.compute_gradients(loss)
		train_op = ENT_opt.apply_gradients(ENT_grads_and_vars)

		for gradient, variable in ENT_grads_and_vars:
			if gradient is None:
				gradient = tf.constant(0.0)
			self.adv_summaries.append(tf.summary.scalar("ENT/gradients/" + variable.name, l2_norm(gradient), family="Grads"))
			self.adv_summaries.append(tf.summary.histogram("ENT/gradients/" + variable.name, gradient, family="Grads"))

		self.summaries.append(tf.summary.scalar('entropy loss', ENT_loss))

		return loss, train_op