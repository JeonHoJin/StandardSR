import numpy as np
from PIL import Image as im
import tensorflow as tf
import glob, os, re, sys
from PSNR import psnr
from scipy.misc import imread
from scipy.misc import imresize
import os
import argparse
from MODEL import model

f_size = 96
S_EPOCH = 27
DATA_PATH = "./data/test/"
path_dir = "./data/test/Set14/"	# Set5, Set14, B100, Urban100
result_path_dir = "./Result/"
file_list = os.listdir(path_dir)
file_list.sort()
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

# www.equasys.de/colorconversion.html
def rgb2ycbcr(img_raw):
	y = 16 + (+ 65.738 * img_raw[:, :, 0] + 129.057 * img_raw[:, :, 1] + 25.064 * img_raw[:, :, 2])/256
	cb = 128 + (- 37.945 * img_raw[:, :, 0] - 74.494 * img_raw[:, :, 1] + 112.439 * img_raw[:, :, 2])/256
	cr = 128 + (+ 112.439 * img_raw[:, :, 0] - 94.154 * img_raw[:, :, 1] - 18.285 * img_raw[:, :, 2])/256
	img = np.dstack((y,cb,cr))
	return img

def colorize(y, ycbcr, flag):
	if flag == True:
		y = (y * 255).astype('uint8')
		img = im.fromarray(y, "L")
	else:
		y = (y * 255).astype('uint8')
		img = np.dstack((y, ycbcr[:, :, 1], ycbcr[:, :, 2]))
		img = im.fromarray(img, "YCbCr").convert("RGB")
	return img

def cropping(img, h, w):
	return img[:h - divmod(h, 12)[1], :w - divmod(w, 12)[1]]

def test_with_sess(epoch, ckpt_path, data_path,sess):
	psnr_list = []
	for i in range(2):
		psnr_list.append([])
	for i in range(3):
		psnr_list[0].append([])
		psnr_list[1].append([])
	saver.restore(sess, ckpt_path)
	for i, item in enumerate(file_list):
		file_name = os.path.splitext(item)
		img_raw = imread(path_dir+item)
		width = img_raw.shape[1]
		height = img_raw.shape[0]
		if len(img_raw.shape) == 3:
			im_gt_ycbcr = np.round(rgb2ycbcr(img_raw)).astype('uint8')
			im_gt_ycbcr = cropping(im_gt_ycbcr,height,width)
			img_raw = im_gt_ycbcr[:,:,0]/255.0
			is_gray = False
		elif len(img_raw.shape) == 2:
			im_gt_ycbcr = 0
			img_raw = img_raw/255.0
			is_gray = True
		img_raw = cropping(img_raw, height, width).astype(np.float32)
		for x in range(2,5):#About scale X2, X3, X4
			im_gt_y = imresize(imresize(img_raw, 1/x, interp='bicubic', mode='F'), [img_raw.shape[0], img_raw.shape[1]], interp='bicubic', mode='F')
			result_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(im_gt_y, (1, im_gt_y.shape[0], im_gt_y.shape[1], 1)), keep_prob: 1.0})

			result_y = np.resize(result_y, (im_gt_y.shape[0], im_gt_y.shape[1]))
			result_y = result_y.clip(16/255, 235/255)
			image = colorize(result_y, im_gt_ycbcr, is_gray)
			image.save(result_path_dir+file_name[0]+"X"+str(x)+file_name[1])

			psnr_list[0][x - 2].append(psnr(result_y, img_raw, x))
			psnr_list[1][x - 2].append(psnr(im_gt_y, img_raw, x))
	print("OURS\nX2 : %f db \nX3 : %f db \nX4 : %f db \n" % (np.mean(psnr_list[0][0]), np.mean(psnr_list[0][1]), np.mean(psnr_list[0][2])))
	print("BICUBIC\nX2 : %f db \nX3 : %f db \nX4 : %f db \n" % (np.mean(psnr_list[1][0]), np.mean(psnr_list[1][1]), np.mean(psnr_list[1][2])))

if __name__ == '__main__':
	model_list = sorted(glob.glob("./checkpoints/adam_epoch_*"))
	model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
	model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("index")]
	if os.path.exists(result_path_dir):
		files = [f for f in os.listdir(result_path_dir)]
		for f in files:
			os.remove(str(result_path_dir+f))
	else:
		os.makedirs(result_path_dir)

	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
		keep_prob = tf.placeholder("float")
		shared_model = tf.make_template('shared_model', model)
		output_tensor, weights 	= shared_model(input_tensor, f_size, keep_prob)
		saver = tf.train.Saver(weights)
		tf.global_variables_initializer().run()
		for model_ckpt in model_list:#read only .data file (started index::step)
			model_ckpt = model_ckpt.split(".data")[0]
			epoch = int(model_ckpt.split('epoch_')[-1].split('(')[0])
			if epoch == S_EPOCH:
				print("Testing ", model_ckpt)
				test_with_sess(epoch, model_ckpt, DATA_PATH,sess)
