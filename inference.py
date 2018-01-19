import tensorflow as tf
import numpy as np
from PIL import Image
"""Inference from Frozen Graph from Tensorflow Object Detection API
"""

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_path',None,'')
tf.app.flags.DEFINE_string('frozen_graph',None,'') # use a frozen pb

classification_graph = tf.Graph()
with classification_graph.as_default()
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.model_ckpt,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
