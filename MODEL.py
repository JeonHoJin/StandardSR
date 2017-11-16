import tensorflow as tf

def model(input_tensor, f_size, keep_prob):
	with tf.device("/gpu:0"):
		weights = []
		tensor = None

		conv_00_w = tf.get_variable("conv_00_w", [3,3,1,f_size], initializer=tf.contrib.layers.xavier_initializer())
		conv_00_b = tf.get_variable("conv_00_b", [f_size], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

		for i in range(18):
			conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,f_size,f_size], initializer=tf.contrib.layers.xavier_initializer())
			conv_b = tf.get_variable("conv_%02d_b" % (i+1), [f_size], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

		tensor = tf.nn.dropout(tensor, keep_prob)

		conv_w = tf.get_variable("conv_20_w", [3,3,f_size,1], initializer=tf.contrib.layers.xavier_initializer())
		conv_b = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
		weights.append(conv_w)
		weights.append(conv_b)
		tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)
		tensor = tf.add(tensor, input_tensor)
		return tensor, weights
