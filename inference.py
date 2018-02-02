import tensorflow as tf
import numpy as np
from PIL import Image
from google.protobuf import text_format
from classification import helpers
from classification.protos import label_map_pb2
Helper = helpers.Helper

"""Inference from Frozen Graph from Tensorflow Object Detection API
"""

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_path',None,'')
tf.app.flags.DEFINE_string('frozen_graph',None,'') # use a frozen pb
tf.app.flags.DEFINE_string('outputs',None,'output labels with starting index  i.e. shape:1,alphanumeric:0')
tf.app.flags.DEFINE_string('label_map',None,'')

if not FLAGS.outputs:
    raise ValueError("Must specify common separated class outputs")
if not FLAGS.label_map:
    raise ValueError("Must specify label map")

label_int_map = {}
with tf.gfile.GFile(FLAGS.label_map, 'r') as fid:
    label_map_string = fid.read()
    label_map = label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
    for label in label_map.label:
         label_int_map[label.name] = {}
         for item in label.item:
             label_int_map[label.name][item.id] = item.name


classification_graph = tf.Graph()
with classification_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.frozen_graph,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with classification_graph.as_default():
  with tf.Session(graph=classification_graph) as sess:
    image = np.array(Image.open(FLAGS.image_path),dtype='float32')
    image = image*2/255.0 -1
    # Definite input and output Tensors for detection_graph
    image_tensor = classification_graph.get_tensor_by_name('image_tensor:0')
    #predictions = {name: classification_graph.get_tensor_by_name("Predictions/"+name+":0")  for }
    predictions  = {n.split(":")[0]:classification_graph.get_tensor_by_name("Predictions/"+n.split(":")[0]+ ":0") for n in FLAGS.outputs.split(",")}#{n.name.split('/')[1]: classification_graph.get_tensor_by_name(n.name + ":0") for n in classification_graph.as_graph_def().node if n.name.startswith("Predictions/")}
    offsets = {n.split(":")[0]: int(n.split(':')[1]) for n in FLAGS.outputs.split(",")}
    #softmax = dict(map((lambda kv: (kv[0],tf.nn.softmax(kv[1]))), predictions.items()))
    predictions = dict(map((lambda kv: (kv[0],np.argmax(kv[1]))),sess.run(predictions,feed_dict={image_tensor:[image]}).items()))
    predictions = {k: predictions[k]+v for k, v in offsets.items()}
    labels = {}
    for k, v in predictions.items():
        labels[k] = label_int_map[k][v]
    print(labels)
