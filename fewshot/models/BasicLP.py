import tensorflow as tf
import numpy as np
from fewshot.models.basic_model import BasicModel
from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model_factory import RegisterModel

def get_cov(c):
	npoints = tf.to_float(tf.shape(c)[0])

	mean = tf.reduce_mean(c, axis=0, keep_dims=True)
	c = c - mean
	cov = tf.matmul(tf.transpose(c), c) / npoints

	return tf.diag_part(cov)

@RegisterModel("basic-LP")
class BasicLP(BasicModel):
	def __init__(self,
               config,
               nway=2,
               nshot=1,
               num_test=30,
               is_training=True,
               dtype=tf.float32):
		super().__init(
               config,
               nway=2,
               nshot=1,
               num_test=30,
               is_training=True,
               dtype=tf.float32)
		height = config.height
		width = config.width
		channels = config.num_channel
		self.additional = tf.placeholder(
          dtype, [None, height, width, channels], name="x_train")
		self.h_additional = self.encode(self.additional)

	def compute_output(self):
		ul_logits = compute_logits(self.h_train, self.h_test)
		uu_logits = compute_logits(self.h_additional, self.h_test)
		print(ul_logits)
		# logits = tf.Print(logits, [tf.shape(logits)])
		ul_logits = ul_logits[0]
		# logits = tf.expand_dims(logits, -1)
		probs = tf.nn.softmax(ul_logits, 1)
		labels = tf.to_float(self.y_train_one_hot[0])
		pred = tf.matmul(probs, labels)
		# pred = tf.Print(pred, [tf.shape(pred), tf.shape(logits)], "prediction dim")
		self._prediction = pred
		self._correct = 0 #tf.equal(tf.argmax(self.prediction, axis=-1), self.y_test)
