import tensorflow as tf
import numpy as np
import sys, os
from itertools import combinations
from fewshot.models.kmeans_utils import compute_logits

FLAGS = tf.flags.FLAGS
tf.app.flags.DEFINE_float('VAT_epsilon', 100.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('VAT_num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('VAT_xi', 1e-2, "small constant for finite difference")
tf.app.flags.DEFINE_float('graph_smoothing', 0.1, 'constant for smoothing the random walk graph')
tf.app.flags.DEFINE_float('visit_loss_weight', 1.0, 'weight for the visit loss of the random walker')
tf.app.flags.DEFINE_float('one_hop_weight', 1.0, "weight for the one hop walk")
tf.app.flags.DEFINE_float('two_hop_weight', 1.0, "weight for the two hop walk")
tf.app.flags.DEFINE_float('three_hop_weight', 1.0, "weight for the three hop walk")
tf.app.flags.DEFINE_float('matching_graph_sample_ratio', 0.5, "ratio of unlabelled points to sample for a matching graph")
tf.app.flags.DEFINE_integer('ngraphs', 2, "Number of matching graphs to construct")
tf.app.flags.DEFINE_integer('nhops', 4, "Max number of hops in a random walk")

def entropy_y_x(logit):
        with tf.name_scope('entropy_x_y'):
                p = tf.nn.softmax(logit)
                return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def relative_entropy_y_x(logit):
        p = tf.nn.softmax(logit,1)
        return -tf.reduce_mean(tf.reduce_sum(p * tf.log(tf.nn.softmax(logit, 0)  ), 1))

def reverse_relative_entropy_y_x(logit):
        p = tf.nn.softmax(logit,0)
        # w = tf.reduce_sum(p, 0) / tf.reduce_sum(p)
        # w = tf.Print(w, [w], '\n-----\n', summarize=5)
        return -tf.reduce_mean(tf.reduce_sum(p * tf.log(tf.nn.softmax(logit, 1)), 1))

def entropy_y_x_weighted(logit):
        p = tf.nn.softmax(logit)
        # class_weights = tf.reduce_sum(p, )
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

def kl_divergence_with_logit2(q_logit, p_logit):
        with tf.name_scope('KL-with-logits'):
                # tf.assert_equal(tf.shape(q_logit), tf.shape(p_logit))
                p_logit=tf.squeeze(p_logit)
                q_logit=tf.squeeze(q_logit)
                p_logit = tf.transpose(p_logit)
                q_logit = tf.transpose(q_logit)
                q = tf.nn.softmax(q_logit)
                qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
                qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
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

def walking_penalty_matching(logit, affinity_matrix, labeled_logit, labeled_affinity):
        shape = tf.shape(logit)
        npoints = tf.to_float(shape[0])
        nclasses = tf.to_float(shape[1])
        c = FLAGS.graph_smoothing

        class_prob = (1-c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape)/npoints

        T0, T1, T2, T3  = landing_probs(logit, affinity_matrix)
        T0_l, T1_l, T2_l, T3_l = landing_probs(labeled_logit, labeled_affinity)

        loss_0 = -tf.reduce_mean( tf.reduce_sum(T0_l * (tf.log(T0) - tf.log(T0_l)), -1) )
        loss_1 = -tf.reduce_mean( tf.reduce_sum(T1_l * (tf.log(T1) - tf.log(T1_l)), -1) )
        loss_2 = -tf.reduce_mean( tf.reduce_sum(T2_l * (tf.log(T2) - tf.log(T2_l)), -1) )
        loss_3 = -tf.reduce_mean( tf.reduce_sum(T3_l * (tf.log(T3) - tf.log(T3_l)), -1) )


        T = loss_0 + FLAGS.one_hop_weight * loss_1 + FLAGS.two_hop_weight * loss_2 #+ FLAGS.three_hop_weight * loss_3

        class_ent = tf.reduce_mean(class_prob, 1)
        u_c = tf.ones(shape=tf.shape(class_ent)) / npoints
        class_ent = -tf.reduce_sum(u_c * tf.log(class_ent))


        penalty = T  + FLAGS.visit_loss_weight * class_ent
        return penalty

def construct_partitioned_tournament(logit, affinity_matrix, h):
    shape = tf.shape(logit)
    npoints = tf.to_float(shape[0])
    nclasses = tf.to_float(shape[1])
    sample_size = npoints #tf.ceil(npoints / nclasses)
    sample_size = tf.to_int32(sample_size)
    logits = [None] * 4
    affinity_matrices = [None] * 4

    logits[0] = logit[::2]
    logits[1] = logit[1::2]
    affinity_matrices[0] = affinity_matrix[::2, ::2]
    affinity_matrices[1] = affinity_matrix[1::2, 1::2]
    shuffle = tf.random_shuffle(tf.range(sample_size))
    # idx = tf.tile(shuffle, [10])
    idx = tf.to_int32(shuffle)
    idx = tf.squeeze(idx)
    logit = tf.gather(logit, idx)
    h = tf.gather(h, idx)
    s = tf.shape(logit)[0]
    affinity_matrix = compute_logits(h, h) - (tf.eye(s, dtype=tf.float32) * 1000.0)
    logits[2] = logit[::2]
    logits[3] = logit[1::2]
    affinity_matrices[2] = affinity_matrix[::2, ::2]
    affinity_matrices[3] = affinity_matrix[1::2, 1::2]
    return logits, affinity_matrices

def walking_penalty_matching_tournament(logit, affinity_matrix, h ):
    shape = tf.shape(logit)
    npoints = tf.to_float(shape[0])
    nclasses = tf.to_float(shape[1])
    c = FLAGS.graph_smoothing
    sample_size = tf.ceil(npoints * FLAGS.matching_graph_sample_ratio / nclasses)
    sample_size = tf.to_int32(sample_size)
    class_prob = (1 - c) * tf.nn.softmax(logit, 0) + c * tf.ones(shape) / npoints


    landing_probs_list = []
    logits, affinity_matrices = construct_partitioned_tournament(logit, affinity_matrix, h)
    for i in range(FLAGS.ngraphs):
        # shuffle = tf.random_shuffle(tf.range(sample_size))
        # idx = tf.tile(shuffle, [10])
        # idx = tf.to_int32(idx)
        # idx = tf.squeeze(idx)
        # ########################
        # # logits_i = logit[i::2]
        # # s = tf.shape(logits_i)[0]
        # # affinity_matrix_i = affinity_matrix[i::2, i::2]
        # ########################
        # logits_i = tf.gather(logit, idx)
        # affinity_matrix_i = tf.gather(tf.gather(affinity_matrix, idx, axis=0), idx, axis=1)
        # # logit = tf.Print(logit, [logits_i])
        x = landing_probs(logits[i], affinity_matrices[i])
        landing_probs_list.append(x)

    loss = 0
    count = 0
    for i in range(len(landing_probs_list)):
        flip = np.random.randint(0, 2)
        # arg1 = i[flip]
        # arg2 = i[(flip+1)%2]
        count += 1
        loss+= cross_entropy_matching_loss(landing_probs_list[i], landing_probs_list[(i+1) % len(landing_probs_list)])
        #loss+= KL_matching_loss(i[1], i[0])
    print(count)
    # loss = KL_matching_loss(landing_probs_list[0], landing_probs_list[2]) + KL_matching_loss(landing_probs_list[1], landing_probs_list[3])
    num_pairs = FLAGS.ngraphs * (FLAGS.ngraphs-1)
    loss = loss / count

    class_ent = tf.reduce_mean(class_prob, 1)
    u_c = tf.ones(shape=tf.shape(class_ent)) / npoints
    class_ent = -tf.reduce_sum(u_c * tf.log(class_ent))


    penalty = loss  + FLAGS.visit_loss_weight * class_ent

    return penalty, loss, landing_probs_list

def walking_penalty_multi(logit, affinity_matrix):
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

def KL_matching_loss(landing_1, landing_2):

    loss = []
    for i in range(FLAGS.nhops):
        l =  -tf.reduce_mean( tf.reduce_sum(landing_1[i] * (tf.log(landing_2[i]) - tf.log(landing_1[i])), -1) )
        # l += 0.5 * -tf.reduce_mean(tf.log(tf.diag_part(landing_1[i])))
        loss.append(l)

    loss_total = loss[0] + FLAGS.one_hop_weight * loss[1] + FLAGS.two_hop_weight * loss[2] + FLAGS.three_hop_weight * loss[3]
    return loss_total

def cross_entropy_matching_loss(landing_1, landing_2):

    loss = []
    for i in range(FLAGS.nhops):
        loss.append( -tf.reduce_mean( tf.reduce_sum(landing_1[i] * (tf.log(landing_2[i])), -1) ))

    loss_total = loss[0] + FLAGS.one_hop_weight * loss[1] + FLAGS.two_hop_weight * loss[2] + FLAGS.three_hop_weight * loss[3]
    return loss_total

def identity_matching_loss(landing):
  loss = []
  for i in range(FLAGS.nhops):
    l =  -tf.reduce_mean(tf.log(tf.diag_part(landing[i])))
    loss.append(l)

  loss_total = loss[0] + FLAGS.one_hop_weight * loss[1] + FLAGS.two_hop_weight * loss[2] + FLAGS.three_hop_weight * \
               loss[3]
  return loss_total


if __name__ == '__main__':

        x = np.zeros([55, 2])
        with tf.Session() as sess:
                print(entropy_y_x(x).eval())