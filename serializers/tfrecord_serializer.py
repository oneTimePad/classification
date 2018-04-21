import sys
import tarfile
from six.moves import urllib
import os
import numpy as np
from PIL import Image
import random
import json
import tensorflow as tf
import argparse
from shutil import copyfile
"""
	TFRecord serializer for muilti-task classification
	Please see the following format below:

	Construct TFRecords format from input dataset
	Expected format:
		  inputs_dir/images:
					{image_name}.{image_file_extension}
		  inputs_dir/labels/{image_name}.json:
				  {
					  {annotation_1} : {category_name},
					  {annotation_2} : {category_name},
					  ...
				  }
		   inputs_dir/annotations.json (mapping between string label and number) :
				  {
					  {annotation_1} : {category_name: category_byte, ...},
					  {annotation_2} : {category_name: category_byte, ...},
					  ...
				  }
"""



parser = argparse.ArgumentParser()
parser.add_argument('--inputs_dir',required=True,help='directory containing "inputs/" and "annotations/"')
parser.add_argument('--records_dir',default='record_dir',help='directory to store records')
parser.add_argument('--records_per_file',type=int,default=10000,help='number of records to put in a file')
parser.add_argument('--times_to_shuffle',type=int,default=60,help='number of times to shuffle inputs dir')
parser.add_argument('--overwrite',type=bool,default=False)

args = parser.parse_args()

IMAGES_DIR = 'images'
LABELS_DIR = 'labels'
ANNOTATIONS_FILE = 'annotations.json'

records_dir = args.records_dir
inputs_dir = args.inputs_dir
records_per_file = args.records_per_file

if not os.path.exists(inputs_dir):
	raise ValueError('%s does not exist' % inputs_dir)

overwrite = args.overwrite


shuffle_times = args.times_to_shuffle

if not os.path.exists(os.path.join(inputs_dir,IMAGES_DIR)):
	raise Exception('%s/%s does not exist. This directory must contain network input examples' % (inputs_dir,IMAGES_DIR))

file_names = []
#shuffle targets in directory

for input_example in os.listdir(os.path.join(inputs_dir,IMAGES_DIR)):
	file_names.append(input_example)
print(len(file_names))

for _ in range(shuffle_times):
	random.shuffle(file_names)

annotation_dir = os.path.join(inputs_dir,LABELS_DIR)

if not os.path.exists(annotation_dir):
	raise Exception('%s/%s does not exist. This directory must contain a directory of annotation json files' %(inputs_dir,LABELS_DIR))

with open(os.path.join(inputs_dir,ANNOTATIONS_FILE),'r') as f:
	annotation_map = json.load(f)
#list of output classes
annotations = annotation_map.keys()
#annotatons for a single instance
examples_dict = {key:None for key in annotations}

record_file_num = 0
record_number = 0


if os.path.exists(records_dir) and not overwrite:
	raise ValueError('%s does exist and overwrite argument not specified' % records_dir)

if not os.path.exists(records_dir):
	os.mkdir(records_dir)


def byte_annotation(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_annotation(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#model will use this to unserialize data
copyfile(os.path.join(inputs_dir,ANNOTATIONS_FILE),os.path.join(records_dir,ANNOTATIONS_FILE))

inputs = os.path.join(inputs_dir,IMAGES_DIR)
record_writer = None
for input_example in file_names:
	#open a new record file if maximum files per record file is exceeded
	if record_number %records_per_file == 0:
		record_filename = os.path.join(records_dir,'record%d.tfrecord'%record_file_num)
		#remove file is exists and overwrite is specified
		if os.path.exists(record_filename):
			os.remove(record_filename)

		#construct writer for record file
		if record_writer is not None:
			record_writer.close()
		record_writer = tf.python_io.TFRecordWriter(record_filename)
		record_file_num+=1


	image = Image.open(os.path.join(inputs,input_example))
	image_np = np.array(image)
	#take the name before the extension and append .json to get the annotations file
	input_name = input_example.split('.')[0]
	input_annotation = os.path.join(annotation_dir,input_name+'.json')
	if not os.path.exists(input_annotation):
		raise Exception('%s does not exist. No annotation found.' % input_annotation)

	with open(input_annotation,'r') as f:
		annotation_dict = json.load(f)

	examples_dict = {}

	examples_dict = {}
	for an in annotation_map.keys():
		if an == 'rotation_angle':
			rot = annotation_dict[an]
			"""
			if (rot >= 0 and rot <22.5) or rot == 360:
				examples_dict[an] = int64_annotation(1)
			elif rot >=22.5 and rot <45:
				examples_dict[an] = int64_annotation(2)
			elif rot >= 45 and rot < 67.5:
				examples_dict[an] = int64_annotation(3)
			elif rot >=67.5 and rot < 90:
				examples_dict[an] = int64_annotation(4)
			elif rot >= 90 and rot < 112.5:
				examples_dict[an] = int64_annotation(5)
			elif rot >= 112.5 and rot < 135:
				examples_dict[an] = int64_annotation(6)
			elif rot >=135 and rot < 157.5:
				examples_dict[an] = int64_annotation(7)
			elif rot >= 157.5 and rot < 180:
				examples_dict[an] = int64_annotation(8)
			elif rot >= 180 and rot < 202.5:
				examples_dict[an] = int64_annotation(9)
			elif rot >= 202.5 and rot < 225:
				examples_dict[an] = int64_annotation(10)
			elif rot >= 225   and rot < 247.5:
				examples_dict[an] = int64_annotation(11)
			elif rot >= 247.5 and rot < 270:
				examples_dict[an] = int64_annotation(12)
			elif rot >= 270   and rot < 292.5:
				examples_dict[an] = int64_annotation(13)
			elif rot >= 292.5 and rot < 315:
				examples_dict[an] = int64_annotation(14)
			elif rot >= 315   and rot < 337.5:
				examples_dict[an] = int64_annotation(15)
			elif rot >= 337.5 and rot < 360:
				examples_dict[an] = int64_annotation(16)
			"""
			examples_dict[an] = int64_annotation(rot)
		else:
			examples_dict[an] = int64_annotation(annotation_map[an][annotation_dict[an]])

	print(input_example)
	print(examples_dict)
	#add in image
	examples_dict['input'] = byte_annotation(image.tobytes())
	#write record
	features = tf.train.Features(feature=examples_dict)

	example = tf.train.Example(features=features)
	record_writer.write(example.SerializeToString())
	record_number+=1
#record_writer.close()
