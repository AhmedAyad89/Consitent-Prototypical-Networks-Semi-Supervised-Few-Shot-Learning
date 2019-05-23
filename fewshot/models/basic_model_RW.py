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
from fewshot.models.SSL_utils import *


l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()

@RegisterModel("basic-RW")
class BasicModelRW(RefineModel):
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
		p = self.h_unlabel
		affinity_matrix = compute_logits(p, p) - (tf.eye(s, dtype=tf.float32) * 1000.0)

		ENT_loss, rw_loss, landing_probs_list = walking_penalty(logits, affinity_matrix)
		loss += ENT_weight * ENT_loss

		ENT_opt = tf.train.AdamOptimizer(ENT_step_size * self.learn_rate, name="Entropy-optimizer")
		ENT_grads_and_vars = ENT_opt.compute_gradients(loss)
		train_op = ENT_opt.apply_gradients(ENT_grads_and_vars)

		rw_grads = ENT_opt.compute_gradients(rw_loss)
		for gradient, variable in rw_grads:
			if gradient is None:
				gradient = tf.constant(0.0)
			self.adv_summaries.append(tf.summary.scalar("ENT/gradients/" + variable.name, l2_norm(gradient), family="Grads"))
			self.adv_summaries.append(tf.summary.histogram("ENT/gradients/" + variable.name, gradient, family="Grads"))

		self.summaries.append(tf.summary.scalar('RW loss', rw_loss))

		self.adv_summaries.append(tf.summary.histogram('Correct landing hist zero hop', tf.diag_part(landing_probs_list[0]), family='landing probs'))
		self.adv_summaries.append(tf.summary.histogram('Correct landing hist one hop', tf.diag_part(landing_probs_list[1]), family='landing probs'))
		self.adv_summaries.append(tf.summary.histogram('Correct landing hist two hop', tf.diag_part(landing_probs_list[2]), family='landing probs'))

		self.adv_summaries.append(tf.summary.histogram('all landing histo', landing_probs_list, family='landing probs'))
		self.adv_summaries.append(tf.summary.scalar('avg correct prob zero hop', tf.reduce_mean(tf.diag_part(landing_probs_list[0])), family='landing probs'))
		self.adv_summaries.append(tf.summary.scalar('avg correct prob one hop', tf.reduce_mean(tf.diag_part(landing_probs_list[1])), family='landing probs'))
		self.adv_summaries.append(tf.summary.scalar('avg correct prob two hop', tf.reduce_mean(tf.diag_part(landing_probs_list[2])), family='landing probs'))



		return loss, train_op
