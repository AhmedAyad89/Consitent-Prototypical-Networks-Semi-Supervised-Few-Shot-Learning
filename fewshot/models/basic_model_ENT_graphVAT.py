from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model

from fewshot.models.refine_model import RefineModel
from fewshot.models.model_VAT import ModelVAT
from fewshot.models.basic_model_VAT import BasicModelVAT
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *


l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()


@RegisterModel("basic-ENT-graphVAT")
class BasicModelENTGraphVAT(ModelVAT):
	def get_train_op(self, logits, y_test):
		loss, train_op = RefineModel.get_train_op(self, logits, y_test)
		classification_loss = loss

		config = self.config
		ENT_weight = config.ENT_weight
		ENT_step_size = config.ENT_step_size

		logits = self._unlabel_logits

		s = tf.shape(logits)
		s = s[0]
		p = self.h_unlabel
		affinity_matrix = compute_logits(p, p) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		# logits = tf.Print(logits, [tf.shape(point_logits)])

		ENT_loss, rw_loss, landing_probs_list = walking_penalty(logits, affinity_matrix)
		loss += ENT_weight * ENT_loss

		loss += 0.2 * self.virtual_adversarial_loss(self.x_unlabel_flat, landing_probs_list)

		ENT_opt = tf.train.AdamOptimizer(ENT_step_size * self.learn_rate, name="Entropy-optimizer")
		ENT_grads_and_vars = ENT_opt.compute_gradients(loss)
		train_op = ENT_opt.apply_gradients(ENT_grads_and_vars)

		return loss, train_op


	def generate_virtual_adversarial_perturbation(self, x, landing_probs, shape=None, is_training=True):
		with tf.name_scope('Gen-adv-perturb'):
			if shape is None:
				shape = tf.shape(x)
				# shape[0]=1
			d = tf.random_normal(shape=[1, 28, 28, 1] )
			for _ in range(FLAGS.VAT_num_power_iterations):
				d = FLAGS.VAT_xi * get_normalized_vector(d)
				logit_p = landing_probs
				logit_m = self.noisy_forward(x, d)
				dist = KL_matching_loss(logit_p, logit_m)
				self.summaries.append(tf.summary.scalar('perturbation-loss', dist))
				grad = tf.gradients(dist, [d], aggregation_method=2, name='Adversarial-grads')[0]
				d = tf.stop_gradient(grad)
			return FLAGS.VAT_epsilon * get_normalized_vector(d)

	def virtual_adversarial_loss(self, x, landing_probs, is_training=True, name="vat_loss", weights=None):
		with tf.name_scope('VAT'):
			shape = self.get_VAT_shape()
			r_vadv = self.generate_virtual_adversarial_perturbation(x, landing_probs, shape=shape, is_training=is_training)
			logit = tf.stop_gradient(landing_probs)
			logit_p = logit
			logit_m = self.noisy_forward(x, r_vadv)
			loss = KL_matching_loss(logit_p, logit_m)


			self.summaries.append(tf.summary.scalar('kl-loss',loss))
		return tf.identity(loss, name=name)

	def noisy_forward(self, data, noise, update_batch_stats=False):
		#Passes the data + noise through the episode model to return logit
		# noise = tf.tile(noise, tf.shape(data)[0])
		encoded = self.phi(data + noise, update_batch_stats=update_batch_stats)
		logits = compute_logits(self._ssl_protos, encoded)
		s = tf.shape(logits)[0]
		affinity_matrix = compute_logits(encoded, encoded) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		ENT_loss, rw_loss, landing_probs_list = walking_penalty(logits, affinity_matrix)
		return landing_probs_list

	def get_VAT_shape(self):
		#Function to be used by children to set the shape of the VAT noise depending on how it is applied
		return None

