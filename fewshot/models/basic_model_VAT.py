
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.refine_model import RefineModel
from fewshot.models.model_VAT import ModelVAT
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity
from fewshot.models.VAT_utils import *

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
log = logger.get()

#Vanilla VAT. Noise added to the inputs
@RegisterModel("basic-VAT")
class BasicModelVAT(ModelVAT):

	def noisy_forward(self, data, noise=tf.constant(0.0), update_batch_stats=False):
		with tf.name_scope("forward"):
			encoded = self.phi(data+noise, update_batch_stats=update_batch_stats)
			logits = compute_logits(self._ssl_protos, encoded)
		return logits


@RegisterModel("basic-VAT-prototypes")
class BasicModelVAT_Prototypes(ModelVAT):
	def __init__(self,
							 config,
							 nway=1,
							 nshot=1,
							 num_unlabel=20,
							 candidate_size=10,
							 is_training=True,
							 dtype=tf.float32):

		super(BasicModelVAT_Prototypes, self).__init__(config, nway, nshot, num_unlabel,
																									 candidate_size, is_training, dtype)


	def noisy_forward(self, data, noise, update_batch_stats=False):
		with tf.name_scope("forward"):
			encoded = self.h_unlabel
			logits = compute_logits(self.protos+noise, encoded)
		return logits

	def get_VAT_shape(self):
		return tf.shape(self.protos)

