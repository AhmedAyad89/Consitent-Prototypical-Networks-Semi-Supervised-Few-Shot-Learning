import tensorflow as tf
import numpy as np
import sys, os
from itertools import combinations
from fewshot.models.kmeans_utils import compute_logits

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_float('VAT_epsilon', 8.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('VAT_num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('VAT_xi', 1e-2, "small constant for finite difference")
tf.app.flags.DEFINE_float('graph_smoothing', 0.1, 'constant for smoothing the random walk graph')
tf.app.flags.DEFINE_float('visit_loss_weight', 1.0, 'weight for the visit loss of the random walker')
tf.app.flags.DEFINE_float('one_hop_weight', 0.7, "weight for the one hop walk")
tf.app.flags.DEFINE_float('two_hop_weight', 0.49, "weight for the two hop walk")
tf.app.flags.DEFINE_float('three_hop_weight', 0.35, "weight for the three hop walk")
tf.app.flags.DEFINE_integer('nhops', 10, "Max number of hops in a random walk")

def entropy_y_x(logit):
				with tf.name_scope('entropy_x_y'):
								p = tf.nn.softmax(logit)
								return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def logsoftmax(x, axis=1):
				with tf.name_scope('Log-of-Softmax'):
								xdev = x - tf.reduce_max(x, axis, keep_dims=True)
								lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), axis, keep_dims=True))
								# return tf.log(tf.nn.softmax(x))
				return lsm

def kl_divergence_with_logit(q_logit, p_logit, weights=None):
				with tf.name_scope('KL-with-logits'):
								# tf.assert_equal(tf.shape(q_logit), tf.shape(p_logit))
								p_logit=tf.squeeze(p_logit)
								q_logit=tf.squeeze(q_logit)
								# p_logit = tf.expand_dims(p_logit, 0)
								# q_logit = tf.expand_dims(q_logit, 0)
								q = tf.nn.softmax(q_logit)
								if weights is None:
												qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
												qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
								else:
												qlogq = tf.reduce_sum(weights * tf.reduce_sum(q * logsoftmax(q_logit), 1))
												qlogp = tf.reduce_sum(weights * tf.reduce_sum(q * logsoftmax(p_logit), 1))
				return qlogq - qlogp

def landing_probs(logit, affinity_matrix):
				shape = tf.shape(logit)
				npoints = tf.to_float(shape[0])
				nclasses = tf.to_float(shape[1])
				affinity_dim = shape[0] / 2
				c = FLAGS.graph_smoothing
				logit = tf.reshape(logit, [-1, shape[1]])
				# affinity_matrix = tf.reshape(affinity_matrix, [shape[0], shape[0]])
				point_prob = (1-c) * tf.nn.softmax(logit, 1) + c * tf.ones(shape)/nclasses
				class_prob = (1-c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape)/npoints
				T0 = tf.matmul(tf.transpose(class_prob), point_prob)
				unlabelled_transition = tf.to_float(tf.nn.softmax(affinity_matrix, -1))
				landing_probs = []
				landing_probs.append(T0)
				T = tf.transpose(class_prob)
				for i in range(FLAGS.nhops - 1):
						T = tf.matmul(T, unlabelled_transition)
						landing_probs.append(tf.matmul(T, point_prob))

				return landing_probs

def walking_penalty(logit, affinity_matrix):
				shape = tf.shape(logit)
				npoints = tf.to_float(shape[0])
				nclasses = tf.to_float(shape[1])
				c = FLAGS.graph_smoothing

				landing_probs_list = landing_probs(logit, affinity_matrix)
				loss = identity_matching_loss(landing_probs_list)

				class_prob = (1-c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape)/npoints
				class_ent = tf.reduce_mean(class_prob, 1)
				u_c = tf.ones(shape=tf.shape(class_ent)) / npoints
				class_ent = -tf.reduce_sum(u_c * tf.log(class_ent))

				penalty = loss + FLAGS.visit_loss_weight * class_ent
				return penalty, loss, landing_probs_list

def get_landing_diag(logit, affinity_matrix):
		shape = tf.shape(logit)
		npoints = tf.to_float(shape[0])
		nclasses = tf.to_float(shape[1])
		c = FLAGS.graph_smoothing

		point_prob = (1-c) * tf.nn.softmax(logit, 1) + c * tf.ones(shape)/nclasses
		class_prob = (1-c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape)/npoints
		T = tf.diag_part(tf.matmul(tf.transpose(class_prob), point_prob))
		T_0 = tf.log(T)
		unlabelled_transition = tf.to_float(tf.nn.softmax(affinity_matrix, -1))

		landing_probs = []
		landing_probs.append(T)
		T = tf.transpose(class_prob)
		for i in range(FLAGS.nhops-1):

						T = tf.matmul(T, unlabelled_transition)
						landing_probs.append(tf.diag_part(tf.matmul(T, point_prob)))

		return  landing_probs
def get_normalized_vector(d):
				with tf.name_scope('Normalize-vector'):
								d /= (1e-12 + tf.reduce_max(tf.abs(d), list(range(1, len(d.get_shape()))), keep_dims=True))
								d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), list(range(1, len(d.get_shape()))) , keep_dims=True))
				return d

def identity_matching_loss(landing):
	loss = []
	for i in range(FLAGS.nhops):
		l =  -tf.reduce_mean(tf.log(tf.diag_part(landing[i])))
		loss.append(l)

	loss_total = loss[0] + FLAGS.one_hop_weight * loss[1] + FLAGS.two_hop_weight * loss[2] + FLAGS.three_hop_weight * \
							 loss[3]
	return loss_total



def get_visit_prob(logit):
	shape = tf.shape(logit)
	npoints = tf.to_float(shape[0])
	c = FLAGS.graph_smoothing

	class_prob = (1 - c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape) / npoints

	return tf.reduce_mean(class_prob,1) *  npoints


def get_dist_filter(logit):
	shape = tf.shape(logit)
	npoints = tf.to_float(shape[0])
	nclasses = tf.to_float(shape[1])
	c = FLAGS.graph_smoothing

	point_prob = (1 - c) * tf.nn.softmax(logit, 1) + c * tf.ones(shape) / nclasses
	class_prob = (1 - c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape) / npoints
	weights = class_prob * point_prob
	weights = tf.reduce_sum(weights, 1)
	median = tf.contrib.distributions.percentile(weights, 50.0)
	weights = tf.where(tf.greater(weights, median), tf.ones_like(weights), tf.zeros_like(weights))
	return weights