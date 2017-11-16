import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import tensorflow as tf
import numpy as np
from MODEL import model
import h5py
import scipy.io

DATA_PATH = "./data/train"
IMG_SIZE = (41, 41)
BATCH_SIZE = 64
f_size = 96
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 10
MAX_EPOCH = 40
USE_PYTHON = True

if USE_PYTHON: DATA_PATH = "./data/train.h5"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

TEST_DATA_PATH = "./data/test/"

def get_train_list(data_path):
	l = glob.glob(os.path.join(data_path,"*"))
	print(len(l))
	l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
	print(len(l))
	train_list = []
	for f in l:
		if os.path.exists(f):
			if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
			if os.path.exists(f[:-4]+"_3.mat"): train_list.append([f, f[:-4]+"_3.mat"])
			if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])
	return train_list

def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
	return input_list, gt_list, np.array(cbcr_list)

if __name__ == '__main__':
	if USE_PYTHON: ## Read h5py file (created by Create_Train.Set.py)
		with h5py.File(DATA_PATH,'r') as hf:
			train_list = hf['train'][:]
		print("Train List : " + str(len(train_list)))
	else: ## Read mat file (created by aug_train.m)
		train_list = get_train_list(DATA_PATH)
	keep_prob = tf.placeholder("float")
	train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))	# shape(41,41,1)
	train_gt_single  	= tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))	# shape(41,41,1)
	q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0], IMG_SIZE[1], 1]]) # (41,41,1), (41,41,1)
	enqueue_op = q.enqueue([train_input_single, train_gt_single])

	train_input, train_gt	= q.dequeue_many(BATCH_SIZE)

	shared_model = tf.make_template('shared_model', model)
	train_output, weights 	= shared_model(train_input, f_size, keep_prob)
	loss = tf.reduce_sum(abs(tf.subtract(train_output, train_gt)))
	for w in weights:
		loss += abs(w) * 1e-4

	global_step 	= tf.Variable(0, trainable=False)

	learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)

	optimizer = tf.train.AdamOptimizer(learning_rate)
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)

	shuffle(train_list) 
	config = tf.ConfigProto()

	with tf.Session(config=config) as sess:
		tf.global_variables_initializer().run()
		start_time = time.time()
		if model_path:
			print("restore model...")
			saver.restore(sess, model_path)
			print("Done")

		def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
			count = 0
			length = len(file_list)
			try:
				while not coord.should_stop():
					i = count % length
					if USE_PYTHON: ## Use h5py to input data
						input_img = file_list[i][1].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
						gt_img = file_list[i][0].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					else: ## Use mat file to input data
						input_img = scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
						gt_img = scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
					sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:gt_img, keep_prob: 0.5})
					count+=1
			except Exception as e:
				print("stopping...", idx, e)
		threads = []
		def signal_handler(signum,frame):
			sess.run(q.close(cancel_pending_enqueues=True))
			coord.request_stop()
			coord.join(threads)
			print("Done")
			sys.exit(1)
		original_sigint = signal.getsignal(signal.SIGINT)
		signal.signal(signal.SIGINT, signal_handler)

		num_thread=20
		del threads[:]

		coord = tf.train.Coordinator()
		for i in range(num_thread):
			length = len(train_list)/num_thread
			t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*int(length):(i+1)*int(length)],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
			threads.append(t)
			t.start()

		for epoch in range(0, MAX_EPOCH):
			for step in range(len(train_list)//BATCH_SIZE):
				_,l,output,lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step], feed_dict= {keep_prob: 1.0})
			print("[epoch %d] loss %.4f\t lr %.8f\t %5.3f sec"%(epoch, np.sum(l)/BATCH_SIZE, lr,  time.time() - start_time))
			saver.save(sess, "./checkpoints/adam_epoch_%03d(%5.3f_sec).ckpt" % (epoch,  time.time() - start_time) ,global_step=global_step)