
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()

@RegisterModel("model-VAT")
class ModelVAT(RefineModel):

	def __init__(self,
               config,
               nway=1,
               nshot=1,
               num_unlabel=10,
               candidate_size=10,
               is_training=True,
               dtype=tf.float32):

		super(ModelVAT, self).__init__(config, nway, nshot, num_unlabel, candidate_size,
																	 is_training, dtype)



	def get_train_op(self, logits, y_test):
		loss, train_op = super().get_train_op(logits, y_test)
		config = self.config
		VAT_step_size = config.VAT_step_size
		labeled_weight = config.labeled_weight
		with tf.control_dependencies([self.protos]):
			labeled_flat = tf.reshape(self.x_test, [-1, config.height, config.width, config.num_channel])
			labeled_logits = tf.squeeze(self.logits)

			weights = tf.nn.softmax(logits, 0)
			weights = tf.reduce_sum(weights, 1)
			weights = weights / tf.reduce_sum(weights)
			self.adv_summaries.append(tf.summary.histogram('unlabel weights', weights))

			vat_loss = self.virtual_adversarial_loss(self.x_unlabel_flat, self._unlabel_logits)\
							 + labeled_weight * self.virtual_adversarial_loss(labeled_flat, labeled_logits)


		vat_opt = tf.train.AdamOptimizer(VAT_step_size * self.learn_rate, name="VAT_optimizer")
		vat_grads_and_vars = vat_opt.compute_gradients(vat_loss)
		vat_train_op = vat_opt.apply_gradients(vat_grads_and_vars)

		for gradient, variable in vat_grads_and_vars:
			if gradient is None:
				gradient = tf.constant(0.0)
			self.adv_summaries.append(tf.summary.scalar("VAT/gradients/" + variable.name, l2_norm(gradient), family="Grads"))
			self.adv_summaries.append(tf.summary.histogram("VAT/gradients/" + variable.name, gradient, family="Grads"))


		loss += config.VAT_weight * vat_loss
		train_op = tf.group(train_op, vat_train_op)
		return loss, train_op

	def predict(self):
		super(RefineModel, self).predict()
		with tf.name_scope('Predict/VAT'):
			self._unlabel_logits = compute_logits(self._ssl_protos, self.h_unlabel)

	def generate_virtual_adversarial_perturbation(self, x, logit, shape=None, is_training=True):
		with tf.name_scope('Gen-adv-perturb'):
			if shape is None:
				shape = tf.shape(x)
			d = tf.random_normal(shape=shape)
			for _ in range(FLAGS.VAT_num_power_iterations):
				d = FLAGS.VAT_xi * get_normalized_vector(d)
				logit_p = logit
				logit_m = self.noisy_forward(x, d)
				dist = kl_divergence_with_logit(logit_p, logit_m)
				self.summaries.append(tf.summary.scalar('perturbation-loss', dist))
				grad = tf.gradients(dist, [d], aggregation_method=2, name='Adversarial-grads')[0]
				d = tf.stop_gradient(grad)
			return FLAGS.VAT_epsilon * get_normalized_vector(d)

	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss", weights=None):
		with tf.name_scope('VAT'):
			shape = self.get_VAT_shape()
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, shape=shape, is_training=is_training)
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.noisy_forward(x, r_vadv)
			loss = kl_divergence_with_logit(logit_p, logit_m, weights)


			self.summaries.append(tf.summary.scalar('kl-loss',loss))
		return tf.identity(loss, name=name)

	def noisy_forward(self, data, noise, update_batch_stats=False):
		#Passes the data + noise through the episode model to return logit
		raise NotImplemented()

	def get_VAT_shape(self):
		#Function to be used by children to set the shape of the VAT noise depending on how it is applied
		return None

