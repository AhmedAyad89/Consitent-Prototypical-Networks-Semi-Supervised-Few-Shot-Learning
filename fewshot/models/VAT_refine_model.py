from fewshot.models.kmeans_utils import assign_cluster, update_cluster, compute_logits
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import concat
from fewshot.models.refine_model import RefineModel
from fewshot.utils import logger
from fewshot.models.VAT_utils import *
from fewshot.models.basic_model_VAT import BasicModelVAT, BasicModelVAT_Prototypes

l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
l1_l2_norm = lambda t: tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(t, 2), 0 )))

@RegisterModel("VAT-refine")
class RefineModelVAT(BasicModelVAT):

	def compute_output(self):
		if not self.is_training:
			config = self.config
			VAT_weight = 0.00005
			num_steps = 20

			weights =  self.embedding_weights
			self.fast_weights = self.embedding_weights

			x = tf.reshape(self.x_train, [-1, config.height, config.width, config.num_channel])
			y = tf.squeeze(self.y_train_one_hot)

			for i in range(num_steps):
				loss = self.virtual_adversarial_loss(self.x_unlabel_flat, self._unlabel_logits, name="VAT-Inference")
				loss +=  self.virtual_adversarial_loss(x, y, name="VAT-Inference")
				ent_loss = entropy_y_x(compute_logits(self.protos, self.h_test))
				loss = ent_loss
				grads = tf.gradients(loss, list(weights.values()))
				grads = [tf.stop_gradient(grad) for grad in grads]
				gradients = dict(zip(weights.keys(), grads))

				self.fast_weights = dict(zip(weights.keys(),
																[self.fast_weights[key] - VAT_weight * gradients[key] for key in weights.keys()]))
				# self.vat_grads_and_vars =[]
				# for gradient, variable in vat_grads_and_vars:
				# 	if gradient is None:
				# 		gradient = tf.constant(0.0)
				# 	self.vat_grads_and_vars.append((gradient, variable))

				# with tf.control_dependencies([vat_train_op]):
				encoded_train, encoded_test = self.encode(self.x_train, self.x_test, update_batch_stats=False, ext_wts=self.fast_weights)
				protos = self._compute_protos(self.nway, encoded_train, self.y_train)

				self._logits = [compute_logits(protos, encoded_test)]

		# self._logits = tf.Print(self._logits, [tf.shape(self.x_unlabel_flat), tf.shape(self._unlabel_logits)])
		# self._logits = tf.Print(self._logits, [tf.shape(self.x_train), tf.shape(self.y_train_one_hot)])
		super().compute_output()

	def noisy_forward(self, data, noise=tf.constant(0.0), update_batch_stats=False, wts=None):
		if wts is None:
			wts = self.embedding_weights
		with tf.name_scope("forward"):
			encoded = self.phi(data+noise, update_batch_stats=update_batch_stats, ext_wts=wts)
			logits = compute_logits(self.protos, encoded)
		return logits

	def generate_virtual_adversarial_perturbation(self, x, logit, shape=None, is_training=True, wts=None):
		with tf.name_scope('Gen-adv-perturb'):
			if shape is None:
				shape = tf.shape(x)
			d = tf.random_normal(shape=shape)
			for _ in range(FLAGS.VAT_num_power_iterations):
				d = FLAGS.VAT_xi * get_normalized_vector(d)
				logit_p = logit
				logit_m = self.noisy_forward(x, d, wts)
				dist = kl_divergence_with_logit(logit_p, logit_m)
				self.summaries.append(tf.summary.scalar('perturbation-loss', dist))
				grad = tf.gradients(dist, [d], aggregation_method=2, name='Adversarial-grads')[0]
				d = tf.stop_gradient(grad)
			return FLAGS.VAT_epsilon * get_normalized_vector(d)

	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss", wts=None):
		with tf.name_scope('VAT'):
			shape = self.get_VAT_shape()
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, shape=shape, is_training=is_training)
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.noisy_forward(x, r_vadv, wts)
			loss = kl_divergence_with_logit(logit_p, logit_m)# + (0.2 * entropy_y_x(tf.expand_dims(logit_m, 0)))
			self.summaries.append(tf.summary.scalar('kl-loss',loss))
		return tf.identity(loss, name=name)


###########################################################################################################

@RegisterModel("VAT-refine-prototypes")
class RefineModelVAT_Prototypes(BasicModelVAT_Prototypes):

	def compute_output(self):
		if not self.is_training:
			config = self.config
			num_steps = config.num_steps
			step_size = config.inference_step_size



			# x = tf.reshape(self._h_test, [-1, config.height, config.width, config.num_channel])
			y = tf.squeeze(self.y_train_one_hot)
			# self._protos = tf.Print(self.protos, [tf.shape(self._unlabel_logits),
			# 																			tf.shape(compute_logits(self.protos, self.h_test)),
			# 																			entropy_y_x(tf.expand_dims(self._unlabel_logits, 0))])
			protos = self.protos
			h_train = self.h_train
			h_test = self.h_test
			h_unlabel = self.h_unlabel
			with tf.name_scope('Adaptation'):
				for i in range(num_steps):
					# vat_loss_lbl = self.virtual_adversarial_loss\
					# 	(tf.squeeze(h_train), tf.squeeze(compute_logits(self.protos, h_train)), name="VAT-Inference")
					# vat_loss_ulbl = self.virtual_adversarial_loss\
					# 	(h_unlabel, self._unlabel_logits, name="VAT-Inference")
					# vat_loss_test = self.virtual_adversarial_loss\
					# 	(tf.squeeze(h_test), tf.squeeze(self._logits), name="VAT-Inference")
					ent_loss_lbl_r = relative_entropy_y_x(compute_logits(self.protos, h_train)[0])
					ent_loss_ulbl_r = relative_entropy_y_x(self._unlabel_logits)
					ent_loss_test_r = relative_entropy_y_x(compute_logits(self.protos, h_test)[0])
					ent_loss_lbl = entropy_y_x(compute_logits(self.protos, h_train))
					ent_loss_ulbl = entropy_y_x(tf.expand_dims(self._unlabel_logits, 0))
					ent_loss_test = entropy_y_x(compute_logits(self.protos, h_test))
					# ent_c = entropy_y_x(
					# 	tf.concat([compute_logits(self.protos, tf.concat( [self.h_train, self.h_test], 1)) , tf.expand_dims(self._unlabel_logits,0)],1))

					# grads_vat_ulbl = tf.gradients(vat_loss_ulbl, self.protos)[0]
					# grads_vat_test = tf.gradients(vat_loss_test, self.protos)[0]
					# grads_ent_ulbl = tf.gradients(ent_loss_ulbl, self.protos)[0]
					# grads_ent_test = tf.gradients(ent_loss_test, self.protos)[0]
					# grads_ent_lbl = tf.gradients(ent_loss_lbl, self.protos)[0]

					loss =  ent_loss_ulbl
					grads = tf.gradients(loss, self.protos, name='GRADS')[0]
					# grads = tf.Print(grads, [ent_loss_lbl, ent_loss_ulbl, ent_loss_test], name="PRINT")
					self._protos= self.protos -  step_size * grads
					self._unlabel_logits = compute_logits(self.protos, h_unlabel)
					self._logits = [compute_logits(self.protos, h_test)]

				# self.inference_summaries.append(tf.summary.scalar('ent-ulbl-loss', l2_norm(ent_loss_ulbl), family='loss_inference' ))
				# self.inference_summaries.append(tf.summary.scalar('vat-ulbl-grad', l2_norm(grads_vat_ulbl), family='inference'))
				# self.inference_summaries.append(tf.summary.scalar('vat-test-grad', l2_norm(grads_vat_test), family='inference'))
				# self.inference_summaries.append(tf.summary.scalar('ent-ulbl-grad', l2_norm(grads_ent_ulbl), family='inference'))
				# self.inference_summaries.append(tf.summary.scalar('ent-test-grad', l2_norm(grads_ent_test), family='inference'))
				# self.inference_summaries.append(tf.summary.scalar('ent-lbl-grad', l2_norm(grads_ent_lbl), family='inference'))

			# self._logits = tf.Print(self._logits,[tf.shape(self._logits)], "\n-------------------------------------\n")
			# self._logits = tf.Print(self._logits, [tf.shape(self.x_untf.shape(self.y_train_one_hot)])
		super().compute_output()

	def generate_virtual_adversarial_perturbation(self, x, logit, shape=None, is_training=True, wts=None):
		with tf.name_scope('Gen-adv-perturb'):
			if shape is None:
				shape = tf.shape(x)
			d = tf.random_normal(shape=shape)
			for _ in range(FLAGS.VAT_num_power_iterations):
				d = FLAGS.VAT_xi * get_normalized_vector(d)
				logit_p = logit
				logit_m = self.noisy_forward(x, d)
				dist = kl_divergence_with_logit2(logit_p, logit_m)
				self.summaries.append(tf.summary.scalar('perturbation-loss', dist))
				grad = tf.gradients(dist, [d], aggregation_method=2, name='Adversarial-grads')[0]
				d = tf.stop_gradient(grad)
			return FLAGS.VAT_epsilon * get_normalized_vector(d)

	def virtual_adversarial_loss(self, x, logit, is_training=True, name="vat_loss", wts=None):
		with tf.name_scope('VAT'):
			shape = self.get_VAT_shape()
			r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, shape=shape, is_training=is_training)
			logit = tf.stop_gradient(logit)
			logit_p = logit
			logit_m = self.noisy_forward(x, r_vadv)
			loss = kl_divergence_with_logit2(logit_p, logit_m)
			self.summaries.append(tf.summary.scalar('kl-loss', loss))
		return tf.identity(loss, name=name)

	def noisy_forward(self, encoded, noise, update_batch_stats=False):
		with tf.name_scope("forward"):
			if self.is_training:
				encoded = self.h_unlabel
			logits = compute_logits(self.protos+noise, encoded)
		return logits

	def get_VAT_shape(self):
		return tf.shape(self.protos)
