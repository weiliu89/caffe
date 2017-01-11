import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import xml.dom.minidom
# %matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = 'your_caffe_root/'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

model_def = 'your_caffe_root/examples/TextBoxes/deploy.prototxt'
model_weights = 'your_caffe_root/examples/TextBoxes/TextBoxes_icdar13.caffemodel'

scales=((300,300),(700,700),(700,500),(700,300),(1600,1600))


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
print(net.blobs['data'].data.shape)

test_list=open('icdar_2013_dataset_root/test_list.txt')
save_dir='your_caffe_root/data/TextBoxes/test_bb_multi_scale/'
for line in test_list.readlines():
	line=line.strip()
	image_name=line
	image_path='icdar_2013_dataset_root/test_images/'+line
	save_detection_path=save_dir+line[0:len(line)-4]+'.txt'
	print(image_path)
	image=caffe.io.load_image(image_path)
	image_height,image_width,channels=image.shape
	print(max(image_height,image_width))
	detection_result=open(save_detection_path,'wt')
	for scale in scales:
		image_resize_height = scale[0]
		image_resize_width = scale[1]
		transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_mean('data', np.array([104,117,123])) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
		
		net.blobs['data'].reshape(1,3,image_resize_height,image_resize_width)		
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image

		# Forward pass.
		detections = net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		# Get detections with confidence higher than 0.1.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		top_labels = get_labelname(voc_labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			xmin = max(1, xmin)
			ymin = max(1, ymin)
			xmax = min(image.shape[1]-1, xmax)
			ymax = min(image.shape[0]-1, ymax)
			score = top_conf[i]
			result=str(xmin)+' '+str(ymin)+' '+str(xmax)+' '+str(ymin)+' '+str(xmax)+' '+str(ymax)+' '+str(xmin)+' '+str(ymax)+' '+str(score)+'\n'
			detection_result.write(result)
	detection_result.close()
test_list.close()
print('success')

