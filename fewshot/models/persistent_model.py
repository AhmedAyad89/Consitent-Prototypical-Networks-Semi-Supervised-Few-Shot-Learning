from __future__ import (absolute_import, division, print_function,
												unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits, assign_cluster
from fewshot.models.model import Model
from fewshot.models.basic_model import BasicModel
from fewshot.models.refine_model import RefineModel
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import concat, weight_variable
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *

log = logger.get()
l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))

def distance(x, y):
	return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), [-1]))

@RegisterModel("persistent")
class PersistentModel(BasicModel):

	def __init__(self, config, nway = 2, nshot = 1, num_test = 30,
							 is_training = True, dtype = tf.float32):

		self.training_data = tf.placeholder(
			dtype, [None, config.height, config.width, config.num_channel], name="training_data")
		self.training_labels = tf.placeholder(tf.int64, [None], name="training_labels")

		self.labels_one_hot = tf.one_hot(self.training_labels, config.n_train_classes)

		reg = config.persistent_reg
		trainable = config.trainable
		d = config.proto_dim
		shape = [1, config.n_train_classes, config.proto_dim]
		self._persistent_protos = tf.get_variable(
				name='persistent_protos',
				shape= shape,
				initializer=tf.random_normal_initializer(),
				regularizer=reg,
				dtype=tf.float32,
				trainable=trainable)

		super().__init__(config,  nway, nshot, num_test, is_training, dtype)

	def classification_train_op(self):
		with tf.name_scope('Classification_train_op'):
			config = self.config
			classification_weight = config.classification_weight
			features = self.phi(self.training_data, update_batch_stats=False)

			logits = compute_logits(self._persistent_protos, features)
			# probs = tf.nn.softmax(logits/1000)
			# logits = tf.Print(logits, [self.training_labels],summarize=50)
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits (logits=logits/200, labels=self.training_labels)
			loss = tf.reduce_mean(loss)
			opt = tf.train.AdamOptimizer(classification_weight * self.learn_rate, name='Classification-Optimizer')
			grads_and_vars = opt.compute_gradients(loss)
			train_op = opt.apply_gradients(grads_and_vars)
			self.adv_summaries.append(tf.summary.scalar("loss", loss, family='classification'))

			for gradient, variable in grads_and_vars:
				if gradient is None:
					gradient = tf.constant(0.0)
				self.adv_summaries.append(tf.summary.scalar("gradients/" + variable.name, l2_norm(gradient), family="classification"))
				# self.adv_summaries.append(tf.summary.scalar("variables/" + variable.name, l2_norm(variable), family="VARS"))
				# self.adv_summaries.append(tf.summary.histogram("gradients/" + variable.name, gradient, family="Grads"))


		return loss, train_op

	def get_train_op(self, logits, y_test):
			loss, train_op = super().get_train_op(logits, y_test)
			classification_loss, classification_op = self.classification_train_op()

			# loss = tf.Print(loss, [classification_loss])
			# train_op = tf.group(train_op, classification_op)
			loss += classification_loss
			opt = tf.train.AdamOptimizer(self.learn_rate, name='Classification-Optimizer')
			grads_and_vars = opt.compute_gradients(loss)
			train_op = opt.apply_gradients(grads_and_vars)
			return loss, train_op



@RegisterModel("persistent-SSL")
class PersistentSSLModel(PersistentModel):
	def __init__(self, config, nway = 2, nshot = 1, num_test = 30,
							 is_training = True, dtype = tf.float32):

		self.unlabeled_training_data = tf.placeholder(
			dtype, [None, config.height, config.width, config.num_channel], name="training_data")
		super().__init__(config, nway, nshot, num_test, is_training, dtype)

	def get_SSL_train_op(self):
		config = self.config
		VAT_weight = config.VAT_weight
		ENT_weight = config.ENT_weight
		features = self.phi(self.unlabeled_training_data, update_batch_stats=False)
		logits = compute_logits(self.protos, features)

		VAT_loss = self.virtual_adversarial_loss(self.unlabeled_training_data, logits)
		ENT_loss = entropy_y_x(tf.expand_dims(logits, 0))
		# probs = tf.nn.softmax(logits)
		# ENT_loss = tf.Print(ENT_loss, [])
		VAT_opt = tf.train.AdamOptimizer(VAT_weight * self.learn_rate, name='Classification-VAT-Optimizer')
		VAT_grads_and_vars = VAT_opt.compute_gradients(VAT_loss)
		VAT_train_op = VAT_opt.apply_gradients(VAT_grads_and_vars)

		self.adv_summaries.append(tf.summary.scalar("VAT-loss", VAT_loss, family='classification'))
		self.adv_summaries.append(tf.summary.scalar("ENT-loss", ENT_loss, family='classification'))

		return VAT_loss, VAT_train_op

	def get_train_op(self, logits, y_test):
		loss, train_op = BasicModel.get_train_op(self, logits, y_test)

		SSL_loss, SSL_train_op = self.get_SSL_train_op()

		loss += SSL_loss
		train_op = tf.group(train_op, SSL_train_op)

		return loss, train_op

	def entropy_train_op(self):
		pass

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

	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss"):
		with tf.name_scope('VAT'):
			shape = self.get_VAT_shape()
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, shape=shape, is_training=is_training)
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.noisy_forward(x, r_vadv)
			loss = kl_divergence_with_logit(logit_p, logit_m)
			self.summaries.append(tf.summary.scalar('kl-loss',loss))
		return tf.identity(loss, name=name)


	def noisy_forward(self, data, noise=tf.constant(0.0), update_batch_stats=False):
		with tf.name_scope("forward"):
			encoded = self.phi(data + noise, update_batch_stats=update_batch_stats)
			logits = compute_logits(self.protos , encoded)
		return logits

	def get_VAT_shape(self):
		return None

