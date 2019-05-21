import numpy as np
import tensorflow as tf

class GCN(object):

	def __init__(self, kernel, features, k, n_layer, dropout_rate):
		# kernel:(N*N), feature:(N*F), k-hop neighbor
		self.inp = features
		dim = int(features.shape[-1])
		
		inputs = features
		outputs = inputs
		for _ in range(n_layer):
			with tf.variable_scope('gcn',reuse=tf.AUTO_REUSE):
				#alpha = tf.get_variable('alpha_'+str(_), shape=[k], dtype=tf.float32, initializer=tf.random_normal_initializer(), trainable=True)
				w = tf.get_variable('w_'+str(_), shape=[dim,dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

				result = tf.matmul(kernel, inputs)
				outputs = tf.nn.tanh(tf.einsum('ijk,kl->ijl',result,w))
				inputs = tf.nn.dropout(outputs, dropout_rate)
		
		self.GCN_out = outputs #(N*F)



