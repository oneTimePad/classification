import tensorflow as tf
import numpy as np
from PIL import Image
from classification import helpers

Helper = helpers.Helper

"""Inference from Frozen Graph from Tensorflow Object Detection API
"""

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')
tf.app.flags.DEFINE_string('image_path',None,'')
tf.app.flags.DEFINE_string('frozen_graph',None,'') # use a frozen pb

if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

configs = Helper.get_configs_from_pipeline_file(FLAGS.pipeline_config)
model_config = configs["model_config"]

classification_graph = tf.Graph()
with classification_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.frozen_graph,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with classification_graph.as_default():
  with tf.Session(graph=classification_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = classification_graph.get_tensor_by_name('image_tensor:0')
    predictions = {name: classification_graph.get_tensor_by_name("Predictions/"+name+":0") }
    predictions  = {n.name.split('/')[1]: classification_graph.get_tensor_by_name(n.name + ":0") for n in classification_graph.as_graph_def().node if n.name.startswith("Predictions/")}
    print(predictions)
