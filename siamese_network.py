import tensorflow as tf
import numpy as np
from GCN import GCN

class SiameseLSTM(object):
	"""
	A LSTM based deep Siamese network for text similarity.
	Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
	"""
	
	def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
		n_input=embedding_size
		n_steps=sequence_length
		n_hidden=hidden_units
		n_layers=1
		# Prepare data shape to match `bidirectional_rnn` function requirements
		# Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
		# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
		# Permuting batch_size and n_steps
		x = tf.transpose(x, [1, 0, 2])
		# Reshape to (n_steps*batch_size, n_input)
		x = tf.reshape(x, [-1, n_input])
		#print(x)
		# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
		x = tf.split(x, n_steps, 0)
		#print(x)
		# Define lstm cells with tensorflow
		# Forward direction cell
		with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope, initializer=tf.orthogonal_initializer()):
			stacked_rnn_fw = []
			for _ in range(n_layers):
				fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
				lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
				stacked_rnn_fw.append(lstm_fw_cell)
			lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

		with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope, initializer=tf.orthogonal_initializer()):
			stacked_rnn_bw = []
			for _ in range(n_layers):
				bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
				lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
				stacked_rnn_bw.append(lstm_bw_cell)
			lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
		# Get lstm cell output

		with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope, initializer=tf.orthogonal_initializer()):
			outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
		#return outputs[-1]

		# output transformation to the original tensor type
		outputs = tf.stack(outputs)
		outputs = tf.transpose(outputs, [1, 0, 2])
		return outputs

	def highway(self, m_scope, x, activation=tf.nn.tanh, carry_bias=-1.0):
		size = int(x.get_shape()[-1])

		with tf.variable_scope('highway_'+m_scope, reuse=tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
			W_T = tf.get_variable(name="weight_transform", shape=[size,size])
			b_T = tf.get_variable(initializer=tf.constant(carry_bias, shape=[size,]), name="bias_transform")

			W = tf.get_variable(name='w', shape=[size, size])
			b = tf.get_variable(initializer=tf.constant(0.1, shape=[size,]), name="b")

			T = tf.nn.sigmoid(tf.einsum('ijk,kl->ijl',x, W_T)+b_T, name="transform_gate")
			H = activation(tf.einsum('ijk,kl->ijl',x, W) + b, name="activation")
			C = 1.0 - T

			y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")

		return y

# to od
	def contrastive_loss(self, y,d,batch_size):
		tmp= y *tf.square(d)
		#tmp= tf.mul(y,tf.square(d))
		tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
		return tf.reduce_sum(tmp +tmp2)/batch_size/2
	
	# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
	def max_pooling(self, lstm_out):
		height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])		 # (step, length of input for one step)

		# do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
		lstm_out = tf.expand_dims(lstm_out, -1)
		output = tf.nn.max_pool(
			lstm_out,
			ksize=[1, height, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')

		output = tf.reshape(output, [-1, width])

		return output

	def avg_pooling(self, lstm_out):
		height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])		 # (step, length of input for one step)
		
		# do avg-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
		lstm_out = tf.expand_dims(lstm_out, -1)
		output = tf.nn.avg_pool(
			lstm_out,
			ksize=[1, height, 1, 1],
			strides=[1, 1, 1, 1],
			padding='VALID')
		
		output = tf.reshape(output, [-1, width])
		
		return output

	def kb_module(self, H, ent_emb, ent_W):
		h_h, w = int(H.get_shape()[1]), int(H.get_shape()[2]) #sequence_len, word_emb_dims
		h_e, w_e = int(ent_emb.get_shape()[1]), int(ent_emb.get_shape()[2]) #Q_ent_length/A_ent_length, ent_emb_dims

		
		out1 = tf.reduce_mean(H, axis=1) #(?,word_emb_dims)

		reshape_h1 = tf.expand_dims(out1, 1) #(?,1,word_emb_dims)
		reshape_h1 = tf.tile(reshape_h1, [1, h_e, 1]) #(?, Q_ent_length/A_ent_length, word_emb_dim)
		reshape_h1 = tf.reshape(reshape_h1, [-1, w]) 
		reshape_h2 = tf.reshape(ent_emb, [-1, w_e]) 
		print(reshape_h1.get_shape(),reshape_h2.get_shape())
		M = tf.tanh(tf.add(tf.matmul(reshape_h1, ent_W['Wqm']), tf.matmul(reshape_h2, ent_W['Wam']))) #(?,att_dims)
		M = tf.matmul(M, ent_W['Wms']) #(?,1)

		S = tf.reshape(M, [-1, h_e]) #(?,Q/A_ent_length)
		S = tf.nn.softmax(S)

		attention_a = tf.einsum('ijk,ij->ik',ent_emb,S)

		out2 = attention_a

		#return tf.concat([H, out2],2),out2
		#return tf.concat([H, out2],2)
		return out1,out2

	def attentive(self, input_q, input_a, att_W):
		h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
		h_a, w_a = int(input_a.get_shape()[1]), int(input_a.get_shape()[2])

		#output_q = tf.reduce_mean(input_q, axis=1)
		output_q = self.avg_pooling(input_q)
		#output_q = self.max_pooling(input_q)

		reshape_q = tf.expand_dims(output_q, 1)
		reshape_q = tf.tile(reshape_q, [1, h_a, 1])
		reshape_q = tf.reshape(reshape_q, [-1, w])
		reshape_a = tf.reshape(input_a, [-1, w_a])

		M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
		M = tf.matmul(M, att_W['Wms'])

		S = tf.reshape(M, [-1, h_a])
		S = tf.nn.softmax(S)

		S_diag = tf.matrix_diag(S)
		attention_a = tf.matmul(S_diag, input_a)
		attention_a = tf.reshape(attention_a, [-1, h_a, w_a])

		#output_a = tf.reduce_mean(attention_a, axis=1)
		output_a = self.avg_pooling(attention_a)
		#output_a = self.max_pooling(attention_a)

		return attention_a

	def overlap(self, embed1, embed2):
		overlap1 = tf.matmul(embed1,tf.transpose(embed2,[0,2,1]))
		overlap1 = tf.expand_dims(tf.reduce_max(overlap1,axis=2),-1)
		
		overlap2 = tf.matmul(embed2,tf.transpose(embed1,[0,2,1]))
		overlap2 = tf.expand_dims(tf.reduce_max(overlap2,axis=2),-1)
		embed1 = tf.concat([embed1,overlap1],2)
		embed2 = tf.concat([embed2,overlap2],2)
		return embed1,embed2

	def attentive_pooling(self, h1, h2, U):
		dim = int(h1.get_shape()[2])
		transform_left = tf.einsum('ijk,kl->ijl',h1, U)
		att_mat = tf.tanh(tf.matmul(transform_left, tf.transpose(h2,[0,2,1])))
		row_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=1)),-1, name='answer_attention')
		column_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(att_mat, axis=2)),-1, name='question_attention1')
		out2 = tf.reshape(tf.matmul(tf.transpose(h2,[0,2,1]),row_max),[-1,dim])
		out1 = tf.reshape(tf.matmul(tf.transpose(h1,[0,2,1]),column_max),[-1,dim])
		return out1,out2

	def sent_know_atten(self, m_scope, sent_rep, know_rep, hidden_dim):
		sent_len = int(sent_rep.get_shape()[1])
		know_len = int(know_rep.get_shape()[1])
		sent_dim = int(sent_rep.get_shape()[-1])
		know_dim = int(know_rep.get_shape()[-1])
		with tf.variable_scope('sentence_knowledge_attention_'+m_scope, initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE) as scope:
			W_sent_trans = tf.get_variable('W_sent_trans', shape=[sent_dim, hidden_dim])
			W_know_trans = tf.get_variable('W_know_trans', shape=[know_dim, hidden_dim])
			W_trans = tf.get_variable('W_trans',shape=[hidden_dim, hidden_dim])
			_sent = tf.einsum('ijk,kl->ijl',sent_rep, W_sent_trans)
			_know = tf.einsum('ijk,kl->ijl',know_rep, W_know_trans)
			rep_left = tf.einsum('ijk,kl->ijl',_sent,W_trans)
			rep = tf.tanh(tf.matmul(rep_left, tf.transpose(_know,[0,2,1]))) # batch*sent_len*know_len
			print(rep)
			
			row_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(rep,axis=1)),-1)
			know = tf.reshape(tf.matmul(tf.transpose(know_rep,[0,2,1]),row_max),[-1,know_dim])
			#know = tf.einsum('ijk,ij->ik',know_rep,row_max)
			column_max = tf.expand_dims(tf.nn.softmax(tf.reduce_max(rep,axis=2)),-1)
			sent = tf.reshape(tf.matmul(tf.transpose(sent_rep,[0,2,1]),column_max),[-1,sent_dim])
			#sent = tf.einsum('ijk,ij->ik',sent_rep,column_max)
			out = tf.concat([sent,know],axis=1)

		return sent, know, out
	
	def max_concat_attention(self, m_scope, sent_rep, know_rep, hidden_dim):
		sent_dim = int(sent_rep.get_shape()[2])
		know_dim = int(know_rep.get_shape()[2])
		sent_len = int(sent_rep.get_shape()[1])
		know_len = int(know_rep.get_shape()[1])
		with tf.variable_scope('sentence_knowledge_attention_'+m_scope, initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE) as scope:
			w1_sent = tf.get_variable('w1_sent',shape=[sent_dim+know_dim,hidden_dim])
			w1_know = tf.get_variable('w1_know',shape=[know_dim+sent_dim,hidden_dim])
			w2_sent = tf.get_variable('w2_sent',shape=[hidden_dim,1])
			w2_know = tf.get_variable('w2_know',shape=[hidden_dim,1])

			max_know = tf.transpose(tf.expand_dims(tf.reduce_max(know_rep,axis=1),[-1]),[0,2,1])
			tile_know = tf.tile(max_know,[1,sent_len,1])
			con_sent_rep = tf.concat([sent_rep,tile_know],axis=-1) #batch×sent_len*(dim1+dim2)
			_sent = tf.nn.tanh(tf.einsum('ijk,kl->ijl',con_sent_rep,w1_sent))
			sent_att = tf.nn.softmax(tf.reshape(tf.einsum('ijk,kl->ijl',_sent, w2_sent),[-1,sent_len]))
			sent = tf.einsum('ijk,ij->ik', sent_rep, sent_att)

			max_sent = tf.transpose(tf.expand_dims(tf.reduce_max(sent_rep,axis=1),[-1]),[0,2,1])
			tile_sent = tf.tile(max_sent,[1,know_len,1])
			con_know_rep = tf.concat([know_rep,tile_sent],axis=-1) #batch×know_len*(dim1+dim2)
			_know = tf.nn.tanh(tf.einsum('ijk,kl->ijl',con_know_rep,w1_know))
			know_att = tf.nn.softmax(tf.reshape(tf.einsum('ijk,kl->ijl',_know, w2_know),[-1,know_len]))
			know = tf.einsum('ijk,ij->ik', know_rep, know_att)

		out = tf.concat([sent,know],axis=1)

		return sent_att, know_att
		return sent, know, out


	def attentive_combine(self, h1, h2, U1, h3, h4, U2):
		dim1 = int(h1.get_shape()[2])
		dim2 = int(h3.get_shape()[2])
		transform_left = tf.einsum('ijk,kl->ijl',h1, U1)
		att_mat1 = tf.tanh(tf.matmul(transform_left, tf.transpose(h2,[0,2,1])))
		row_max1 = tf.reduce_max(att_mat1, axis=1)
		column_max1 = tf.reduce_max(att_mat1, axis=2)

		transform_right = tf.einsum('ijk,kl->ijl',h3, U2)
		att_mat2 = tf.tanh(tf.matmul(transform_right, tf.transpose(h4,[0,2,1])))
		row_max2 = tf.reduce_max(att_mat2, axis=1)
		column_max2 = tf.reduce_max(att_mat2, axis=2)

		#row_max1 = tf.expand_dims(tf.nn.softmax(row_max1),-1) #a sent
		#row_max2 = tf.expand_dims(tf.nn.softmax(row_max2),-1) #a know
		#column_max1 = tf.expand_dims(tf.nn.softmax(column_max1),-1) #q sent
		#column_max2 = tf.expand_dims(tf.nn.softmax(column_max2),-1) #q know

		sent_att_q, know_att_q = self.max_concat_attention('shared', h1, h3, 200)
		sent_att_a, know_att_a = self.max_concat_attention('shared', h2, h4, 200)
		row_max1 = tf.expand_dims(tf.nn.softmax(tf.add(row_max1,sent_att_a)),-1) #a sent
		row_max2 = tf.expand_dims(tf.nn.softmax(tf.add(row_max2,know_att_a)),-1) #a know
		column_max1 = tf.expand_dims(tf.nn.softmax(tf.add(column_max1, sent_att_q)),-1) #q sent
		column_max2 = tf.expand_dims(tf.nn.softmax(tf.add(column_max2, know_att_q)),-1) #q know

		#with tf.variable_scope('transfrom', initializer=tf.contrib.layers.xavier_initializer()):
			#trans1 = tf.get_variable('trans1', shape=[self.length['sequence_length'], self.length['Q_ent_length']])
			#trans2 = tf.get_variable('trans2', shape=[self.length['Q_ent_length'], self.length['sequence_length']])

			#row_max1 = tf.add(row_max1, tf.einsum('ij,jk->ik',row_max2, trans2))
			#column_max1 = tf.add(column_max1, tf.einsum('ij,jk->ik',column_max2, trans2))
			#row_max2 = tf.add(row_max2, tf.einsum('ij,jk->ik',row_max1, trans1))
			#column_max2 = tf.add(column_max2, tf.einsum('ij,jk->ik',column_max1, trans1))

			#row_max1 = tf.expand_dims(tf.nn.softmax(row_max1),-1,name='answer_attention1')
			#column_max1 = tf.expand_dims(tf.nn.softmax(column_max1),-1,name='question_attention1')
			#row_max2 = tf.expand_dims(tf.nn.softmax(row_max2),-1,name='answer_attention2')
			#column_max2 = tf.expand_dims(tf.nn.softmax(column_max2),-1,name='question_attention2')
			

		#row_max = tf.tanh(tf.add(row_max1, row_max2, name='answer_attention2'))
		#column_max = tf.tanh(tf.add(column_max1, column_max2, name='question_attention2'))
		#row_max = tf.add(row_max1, row_max2, name='answer_attention2')
		#column_max = tf.add(column_max1, column_max2, name='question_attention2')
		out_a1 = tf.reshape(tf.matmul(tf.transpose(h2,[0,2,1]),row_max1),[-1,dim1])
		out_q1 = tf.reshape(tf.matmul(tf.transpose(h1,[0,2,1]),column_max1),[-1,dim1])
		out_a2 = tf.reshape(tf.matmul(tf.transpose(h4,[0,2,1]),row_max2),[-1,dim2])
		out_q2 = tf.reshape(tf.matmul(tf.transpose(h3,[0,2,1]),column_max2),[-1,dim2])
		if self.mode == 'nokg':
			out1 = out_q1
			out2 = out_a1
		else:
			out1 = tf.concat([out_q1,out_q2],1)
			out2 = tf.concat([out_a1,out_a2],1)
			print(out1.get_shape(),out2.get_shape())
		return out1,out2


	def __init__(
		self, length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, embedding_matrix, entity_embedding_matrix,entity_embedding_dim, entity_vocab_size,mode, args):
		self.length = length
		self.mode = mode
		sequence_length = length['sequence_length']
		Q_ent_length = length['Q_ent_length']
		A_ent_length = length['A_ent_length']
		k = args.k
		n_layer = args.n_layer

		# Placeholders for input, output and dropout
		self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
		self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
		self.ent_x1 = tf.placeholder(tf.int32, [None, Q_ent_length], name="ent_x1")
		self.ent_x2 = tf.placeholder(tf.int32, [None, A_ent_length], name="ent_x2")
		self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
		self.add_fea = tf.placeholder(tf.float32, [None, 4], name="add_fea")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		#
		self.kernel_x1 = tf.placeholder(tf.float32,[None,Q_ent_length,Q_ent_length], name='ques_sys_laplacian')
		self.kernel_x2 = tf.placeholder(tf.float32,[None,A_ent_length,A_ent_length], name='ans_sys_laplacian')
		self.entity_mask_q = tf.placeholder(tf.float32, [None, Q_ent_length], name='entity_mask_ques')
		self.entity_mask_a = tf.placeholder(tf.float32, [None, A_ent_length], name='entity_mask_ans')

		# Keeping track of l2 regularization loss (optional)
		#l2_loss = tf.constant(0.0, name="l2_loss")
		print(self.ent_x1.get_shape())
		# Embedding layer
		with tf.name_scope("embedding"):
			'''
			self.W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				trainable=True,name="W")
			'''
			# char embedding
			if embedding_matrix.all() != None:
				print('pre-trained word embedding')
				self.W = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
			else:
				print('random word embedding')
				self.W = tf.get_variable("emb", [vocab_size, embedding_size])

			# char embedding
			if entity_embedding_matrix.all() == None:
				print('random graph embedding')
				self.ent_emb = tf.Variable(tf.random_uniform([entity_vocab_size, entity_embedding_dim], -1.0, 1.0),trainable=False,name="ent_emb")
			else:
				print('pre-trained graph embedding')
				self.ent_emb = tf.Variable(entity_embedding_matrix, trainable=False, name="ent_emb", dtype=tf.float32)
			
			self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
			self.embedded_ent1 = tf.nn.embedding_lookup(self.ent_emb, self.ent_x1)
			#self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
			self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
			self.embedded_ent2 = tf.nn.embedding_lookup(self.ent_emb, self.ent_x2)
			#self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)\
			self.embedded_chars1,self.embedded_chars2 = self.overlap(self.embedded_chars1,self.embedded_chars2)
			#self.embedded_ent1,self.embedded_ent2 = self.overlap(self.embedded_ent1,self.embedded_ent2)
			#------------------highway------------------------------
			self.embedded_chars1 = self.highway('sentence', self.embedded_chars1)
			self.embedded_chars2 = self.highway('sentence', self.embedded_chars2)
			self.embedded_ent1 = self.highway('knowledge', self.embedded_ent1)
			self.embedded_ent2 = self.highway('knowledge', self.embedded_ent2)

			print(self.embedded_chars1.get_shape())
			print(self.embedded_chars2.get_shape())
			print(self.embedded_ent1.get_shape())
			print(self.embedded_ent2.get_shape())

		attention_size = 200
		with tf.name_scope("ent_weight"), tf.variable_scope("ent_weight",initializer=tf.contrib.layers.xavier_initializer()):
			self.ent_W = {
				'Wam' : tf.get_variable("ent_Wam",[entity_embedding_dim, attention_size]),
				'Wqm' : tf.get_variable("ent_Wqm",[2*hidden_units, attention_size]),
				'Wms' : tf.get_variable("ent_Wms",[attention_size,1])
			}

		# Create a convolution + maxpool layer for each filter size
		with tf.name_scope("RNN"):
			self.h1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size+1, sequence_length, hidden_units)
			self.h2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size+1, sequence_length, hidden_units)
			#self.out1=self.kb_module(self.h1, self.embedded_ent1, self.ent_W)
			#self.out2=self.kb_module(self.h2, self.embedded_ent2, self.ent_W)
			#self.out1,self.ent_att1=self.kb_module(self.h1, self.embedded_ent1, self.ent_W)
			#self.out2,self.ent_att2=self.kb_module(self.h2, self.embedded_ent2, self.ent_W)

		with tf.name_scope('GCN'):
			if mode == 'nogcn':
				print('no gcn!')
				self.gcn_ent1 = tf.einsum('ijk,ij->ijk', self.embedded_ent1, self.entity_mask_q)
				self.gcn_ent2 = tf.einsum('ijk,ij->ijk', self.embedded_ent2, self.entity_mask_a)
			else:
				with tf.name_scope('shared'):
					GCN_Q = GCN(self.kernel_x1, self.embedded_ent1, k, n_layer, self.dropout_keep_prob)
					self.gcn_ent1 = GCN_Q.GCN_out
				with tf.name_scope('shared'):
					GCN_A = GCN(self.kernel_x2, self.embedded_ent2, k, n_layer, self.dropout_keep_prob)
					self.gcn_ent2 = GCN_A.GCN_out
					self.gcn_out = GCN_A.GCN_out
					self.gcn_in = GCN_A.inp
				print(self.gcn_ent1.shape, self.gcn_ent2.shape)
				#self.gcn_ent1,self.gcn_ent2 = self.overlap(self.gcn_ent1,self.gcn_ent2)
				self.gcn_ent1 = tf.einsum('ijk,ij->ijk', self.gcn_ent1, self.entity_mask_q)
				self.gcn_ent2 = tf.einsum('ijk,ij->ijk', self.gcn_ent2, self.entity_mask_a)

		# Create a convolution + maxpool layer for each filter size
		filter_sizes = [2,3,4,5]
		height = int(self.gcn_ent1.get_shape()[2]) #ent_embed_dim
		num_filters = 100
		
		h_left = []
		h_right = []

		for i, filter_size in enumerate(filter_sizes):
			filter_shape = [filter_size, height, num_filters]
			with tf.name_scope("conv-maxpool-left-%s" % filter_size):
				# Convolution Layer
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv1d(
					self.gcn_ent1,
					W,
					stride=1,
					padding="SAME",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				h = tf.nn.dropout(h, self.dropout_keep_prob)
				print(h.get_shape())
				h_left.append(h)
			with tf.name_scope("conv-maxpool-right-%s" % filter_size):
				# Convolution Layer
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv1d(
					self.gcn_ent2,
					W,
					stride=1,
					padding="SAME",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				h = tf.nn.dropout(h, self.dropout_keep_prob)
				h_right.append(h)
		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		num_filters_total = num_filters * len(filter_sizes)
		self.h_q = tf.concat(h_left,2)
		self.h_a = tf.concat(h_right,2)

		with tf.name_scope("attentive_pooling"):
			U1 = tf.get_variable(
					"U1",
					shape=[2*hidden_units, 2*hidden_units],
					initializer=tf.contrib.layers.xavier_initializer())
			U2 = tf.get_variable(
					"U2",
					shape=[num_filters_total, num_filters_total],
					initializer=tf.contrib.layers.xavier_initializer())
			#self.attentive_h_q, self.attentive_h_a = self.attentive_pooling(self.h_q,self.h_a,U2)
			#self.out1 = tf.concat([self.attentive_h1,self.attentive_h_q],1)
			#self.out2 = tf.concat([self.attentive_h2,self.attentive_h_a],1)
			if mode == 'noatt':
				print('noatt')
				self.h1, self.h2 = tf.reduce_max(self.h1,axis=1), tf.reduce_max(self.h2, axis=1)
				self.h_q, self.h_a = tf.reduce_max(self.h_q,axis=1), tf.reduce_max(self.h_a, axis=1)
				self.out1 = tf.concat([self.h1, self.h_q], axis=-1)
				self.out2 = tf.concat([self.h2, self.h_a], axis=-1)
				#self.out1, self.out2 = self.attentive_pooling(self.h1,self.h2,U1)
			else:
				self.out1,self.out2 = self.attentive_combine(self.h1,self.h2,U1,self.h_q,self.h_a,U2)

		#with tf.name_scope('max_concat_attention'):
		#	_,_, self.out_q = self.max_concat_attention('shared', self.h1, self.h_q, 200)
		#	_,_, self.out_a = self.max_concat_attention('shared', self.h2, self.h_a, 200)
		#	self.out1 = tf.concat([self.out1,self.out_q],axis=1)
		#	self.out2 = tf.concat([self.out2,self.out_a],axis=1)
		
		#with tf.name_scope('sent_know_attention'):
		#	_, _, self.out_q = self.sent_know_atten('shared', self.h1, self.h_q, 200)
		#	_, _, self.out_a = self.sent_know_atten('shared', self.h2, self.h_a, 200)
		#	self.out1 = tf.concat([self.out1,self.out_q],axis=1)
		#	self.out2 = tf.concat([self.out2,self.out_a],axis=1)

		#self.out1 = tf.concat([self.h1,self.out1], 1)
		#self.out2 = tf.concat([self.h2,self.out2], 1)

		 # Compute similarity
		with tf.name_scope("similarity"):
			#self.q = self.avg_pooling(self.h1)
			#self.a = self.avg_pooling(self.h2)
			sim_size = int(self.out1.get_shape()[1])
			W = tf.get_variable(
				"W",
				shape=[sim_size, sim_size],
				initializer=tf.contrib.layers.xavier_initializer())
			self.transform_left = tf.matmul(self.out1, W)
			self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.out2), 1, keep_dims=True)
			
			#print self.sims

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Make input for classification
		if mode == 'raw':
			self.new_input = tf.concat([self.out1, self.sims, self.out2], 1, name='new_input')
		else:
			self.new_input = tf.concat([self.out1, self.sims, self.out2, self.add_fea], 1, name='new_input')

		num_feature = int(self.new_input.get_shape()[1])
		
		# hidden layer
		hidden_size = 200
		with tf.name_scope("hidden"):
			W = tf.get_variable(
				"W_hidden",
				shape=[num_feature, hidden_size],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
			print(self.h_drop)

		# Final (unnormalized) scores and predictions
		with tf.name_scope("output"):
			W = tf.get_variable(
				"W_output",
				shape=[hidden_size, 2],
				initializer=tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.prob = tf.nn.xw_plus_b(self.h_drop, W, b)
			#self.gcn_out = self.prob
			self.soft_prob = tf.nn.softmax(self.prob, name='distance')
			self.predictions = tf.argmax(self.soft_prob, 1, name="predictions")

		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=tf.one_hot(self.input_y,2))
			#self.y_2 = tf.one_hot(self.input_y,2)
			#losses = - (self.y_2*tf.log(self.soft_prob)*1.5 + (1-self.y_2)*tf.log(1-self.soft_prob))
			self.loss = tf.reduce_mean(losses) 
			self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())

			self.total_loss = self.loss + self.l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, self.input_y)
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		'''
		with tf.name_scope("output"):
			#self.sim_mat = tf.get_variable("sim_mat", [4*sequence_length, 4*sequence_length])
			#a = tf.matmul(tf.reshape(self.out1,[-1,4*sequence_length]),self.sim_mat)
			#a = tf.reshape(a,[-1,1,4*sequence_length])
			#self.mix = tf.reshape(tf.matmul(a,tf.reshape(self.out2,[-1,4*sequence_length,1])),[-1,1])
			self.mix = tf.reshape(tf.reduce_sum(tf.multiply(self.out1,self.out2),axis=1),[-1,1])
			if mode == 'raw':
				self.outputs = tf.concat([self.out1, self.mix, self.out2],1)
				#self.softmax_w = tf.get_variable("softmax_w", [8*sequence_length+1, 2])
				self.softmax_w = tf.get_variable("softmax_w", [2*num_filters_total+1, 2])
			else:
				self.outputs = tf.concat([self.out1, self.mix, self.out2, self.add_fea],1)
				#self.softmax_w = tf.get_variable("softmax_w", [8*sequence_length+5, 2])
				self.softmax_w = tf.get_variable("softmax_w", [2*num_filters_total+5, 2])
			self.softmax_b = tf.get_variable("softmax_b", [2])
			self.prob = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
			self.soft_prob = tf.nn.softmax(self.prob, name='distance')
			self.predictions = tf.argmax(tf.nn.softmax(self.prob), 1, name="predictions")
		with tf.name_scope("loss"):
			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=tf.one_hot(self.input_y,2))
			self.loss = tf.reduce_sum(self.cross_entropy)

			self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())

			self.total_loss = self.loss + self.l2_loss
		#### Accuracy computation is outside of this class.
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, self.input_y)
			self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
		'''
