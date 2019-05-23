import numpy as np
import tensorflow as tf
from fewshot.data.episode import Episode
from fewshot.models.VAT_utils import *
from fewshot.models.kmeans_utils import compute_logits
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from fewshot.data.mini_imagenet import MiniImageNetDataset, MiniImageNetDatasetAll
import matplotlib.pyplot as plt


l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))


def preprocess_batch(batch):
	if len(batch.x_train.shape) == 4:
		x_train = np.expand_dims(batch.x_train, 0)
		y_train = np.expand_dims(batch.y_train, 0)
		x_test = np.expand_dims(batch.x_test, 0)
		y_test = np.expand_dims(batch.y_test, 0)
		if batch.x_unlabel is not None:
			x_unlabel = np.expand_dims(batch.x_unlabel, 0)
		else:
			x_unlabel = None
		if hasattr(batch, 'y_unlabel') and batch.y_unlabel is not None:
			y_unlabel = np.expand_dims(batch.y_unlabel, 0)
		else:
			y_unlabel = None

		return Episode(
				x_train,
				y_train,
				x_test,
				y_test,
				x_unlabel=x_unlabel,
				y_unlabel=y_unlabel,
				y_train_str=batch.y_train_str,
				y_test_str=batch.y_test_str,
        selected_classes = batch.selected_classes)
	else:
		return batch


def basic_stats(sess, model, dataset):
	num_batches = 10
	s = 25
	entropy_avg = np.zeros([num_batches])
	protos_norm = np.zeros([num_batches])
	pairwise_distance = np.zeros([num_batches])
	pairwise_proto_distance = np.zeros([num_batches])
	within_class_distance = np.zeros([num_batches])
	mean_embedding_norm = np.zeros([num_batches])
	VAT_loss_mean=0
	for i in range(num_batches):
		images = dataset.next()
		batch = preprocess_batch(images)
		feed_dict = {
			model.x_train: batch.x_train,
			model.y_train: batch.y_train,
			model.x_test: batch.x_test,
			model.x_unlabel: batch.x_unlabel
		}

		h_unlabel_ = model.h_unlabel
		unlabel_affinity_matrix_ = compute_logits(h_unlabel_, h_unlabel_) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		outputs = [
			model._unlabel_logits,
			tf.nn.softmax(model._unlabel_logits),
			model.protos,
			h_unlabel_,
			model.h_test
			# unlabel_affinity_matrix_,
			]
		# VAT_loss_ = model.vat_loss
		# VAT_loss_mean += sess.run(VAT_loss_, feed_dict=feed_dict)
		unlabelled_logits, unlabelled_pred, protos, h_unlabel, h_test = sess.run(outputs, feed_dict=feed_dict)
		protos = np.asarray(protos[0])
		h_test = h_test[0]
		class_points = h_test[batch.y_test[0] == 1]
		# print(pairwise_distances(class_points, metric='sqeuclidean'))
		# mean_embedding_norm[i] = np.mean()
		pairwise_proto_distance[i] = np.mean(pairwise_distances(protos, metric='sqeuclidean'))
		within_class_distance[i] = np.mean(pairwise_distances(class_points, metric='sqeuclidean'))
		pairwise_distance[i] = np.mean(pairwise_distances(h_test, metric='sqeuclidean'))
		protos_norm[i] = l2_norm(protos).eval() / protos.shape[0]
		entropy_avg[i] = entropy_y_x(unlabelled_logits).eval()
		# bounding_box_max = np.max(class_points, axis=0)
		# bounding_box_min = np.min(class_points, axis=0)
		# sides = bounding_box_max - bounding_box_min
		# zeros = np.sum(sides < 0.01)
		# # sides[sides<0.5] = 1
		# volume = np.prod(sides)
		# print(np.shape(bounding_box_max), volume, zeros, np.sum(np.abs(sides)))
		# print(sorted(sides, reverse=True)[:100])
		print(batch.selected_classes, 'dfokhgiludfhg')


	print("ENT: ", entropy_avg.mean(), entropy_avg.std())
	print("VAT loss: ", VAT_loss_mean/num_batches)
	print("Proto norms: ", protos_norm.mean())
	print("Proto_Pairwise distance: ", pairwise_proto_distance.mean(), pairwise_proto_distance.std())
	print("Within class distance", within_class_distance.mean(), within_class_distance.std())
	print("pairwise distance: ", pairwise_distance.mean(), pairwise_distance.std())


def graph_stats(sess, model, dataset):
	num_batches = 60
	s = 50
	walk_length = 10
	entropy_avg = np.zeros([num_batches])
	landing_probs_means = np.zeros([num_batches, walk_length])
	eigen_means = np.zeros([25])
	for i in range(num_batches):
		images = dataset.next()
		batch = preprocess_batch(images)
		feed_dict = {
			model.x_train: batch.x_train,
			model.y_train: batch.y_train,
			model.x_test: batch.x_test,
			model.x_unlabel: batch.x_unlabel
		}
		h_unlabel_ = model.h_unlabel
		unlabel_affinity_matrix_ = compute_logits(h_unlabel_, h_unlabel_) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		# outputs = [unlabel_affinity_matrix_, model._unlabel_logits]
		# unlabel_affinity_matrix, unlabel_logits = sess.run(outputs, feed_dict=feed_dict)
		landing_probs = walking_penalty_multi(model._unlabel_logits, unlabel_affinity_matrix_)
		landing_probs = sess.run(landing_probs, feed_dict = feed_dict)
		unlabel_affinity_matrix = sess.run(unlabel_affinity_matrix_, feed_dict=feed_dict)
		# x_to_x =  tf.nn.softmax(unlabel_affinity_matrix, 1)
		# x_to_x = x_to_x.eval()
		# x_to_x2= tf.transpose( tf.nn.softmax(unlabel_affinity_matrix, 1))
		# x_to_x2 = x_to_x2.eval()
		#
		# div = -tf.reduce_mean( tf.reduce_sum(x_to_x * (tf.log(x_to_x2) - tf.log(x_to_x)), -1) )
		# print(div.eval())
		# eigen_vals = np.sort(np.abs(np.linalg.eig(x_to_x)[0]))
		# eigen_means += eigen_vals [:74:-1]
		# print(landing_probs)
		# print(x_to_x)
		landing_probs_means[i] = np.mean(landing_probs, 1)
		# print(np.mean(landing_probs, 1))


	print(np.mean(landing_probs_means, axis=0))
	# plt.plot(landing_probs_means)
	# plt.ylabel('some numbers')
	# plt.show()


def distractor_stats(sess, model, dataset):
	num_batches = 60
	s = 50
	walk_length = 10
	entropy_avg = np.zeros([num_batches])
	landing_probs_means = np.zeros([num_batches, walk_length])
	eigen_means = np.zeros([25])
	non_prob = 0
	dist_prob = 0
	for i in range(num_batches):
		images = dataset.next()
		batch = preprocess_batch(images)
		feed_dict = {
			model.x_train: batch.x_train,
			model.y_train: batch.y_train,
			model.x_test: batch.x_test,
			model.x_unlabel: batch.x_unlabel
		}
		h_unlabel_ = model.h_unlabel
		unlabel_affinity_matrix_ = compute_logits(h_unlabel_, h_unlabel_) - (tf.eye(s, dtype=tf.float32) * 1000.0)
		outputs = [unlabel_affinity_matrix_, model._unlabel_logits]
		unlabel_affinity_matrix, unlabel_logits = sess.run(outputs, feed_dict=feed_dict)
		landing_probs, class_prob = walking_penalty_multi(model._unlabel_logits, unlabel_affinity_matrix_)
		landing_probs, class_prob = sess.run([landing_probs, class_prob], feed_dict = feed_dict)
		unlabel_affinity_matrix = sess.run(unlabel_affinity_matrix_, feed_dict=feed_dict)
		# print(batch.y_unlabel)
		# print(class_prob, np.sum(class_prob[:25]), np.sum(class_prob[25:]), "\n---------------------------------\n")
		non_prob += np.sum(class_prob[:25])
		dist_prob += np.sum(class_prob[25:])

	print(dist_prob/num_batches, non_prob/num_batches)
def gaussian_fit(sess, model, dataset):
	num_batches = 5
	s = 25
	walk_length = 10
	entropy_avg = np.zeros([num_batches])
	landing_probs_means = np.zeros([num_batches, walk_length])
	for i in range(num_batches):
		images = dataset.next()
		batch = preprocess_batch(images)
		feed_dict = {
			model.x_train: batch.x_train,
			model.y_train: batch.y_train,
			model.x_test: batch.x_test,
			model.x_unlabel: batch.x_unlabel
		}

		output = [model.h_test, model.protos]
		h_test , protos = sess.run(output, feed_dict=feed_dict)
		protos = np.asarray(protos[0][0])
		protos = np.expand_dims(protos, axis=0)
		print(protos.shape, h_test.shape)
		h_test =h_test[0][:5]

		weights = [1.0]
		model_full = GaussianMixture(n_components=1, covariance_type='diag', means_init=protos, weights_init=weights)
		model_diag = GaussianMixture(n_components=1, covariance_type='spherical', means_init=protos, weights_init=weights)


		model_diag.fit(h_test)
		model_full.fit(h_test)

		print("Diag: ", model_diag.lower_bound_, model_diag.weights_, model_diag.aic(h_test), model_diag.predict(h_test))
		print("Full: ", model_full.lower_bound_, model_full.weights_, model_full.aic(h_test), model_full.predict(h_test))
		print(l2_norm(protos - model_full.means_).eval(), l2_norm(protos - model_diag.means_).eval())
		print('\n----------------------***********************-------------------------------\n')




