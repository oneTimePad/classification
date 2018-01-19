import tensorflow as tf
from classification.builders import model_builder
from classification import helpers
import os


"""Exports Classification model to frozen pb graph
Modeled after https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
"""

Helper = helpers.Helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')
tf.app.flags.DEFINE_string('model_ckpt_path',None,'')
tf.app.flags.DEFINE_string('export_dir',None,'')

if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

if not FLAGS.model_ckpt_path:
    raise ValueError("Must specify path to model checkpoint path and prefix")

if not os.path.exists(FLAGS.export_dir):
    raise ValueError("Export directory doesn't exist")

configs = Helper.get_configs_from_pipeline_file(FLAGS.pipeline_config)
model_config = configs["model_config"]

image_tensor = tf.placeholder(shape=(None,150,150,3),name="image_tensor",dtype=tf.float32)
classification_model = model_builder.build(model_config, False)
predictions = classification_model.predict(image_tensor)
logits = {n.name.split('/')[1]: tf.get_default_graph().get_tensor_by_name(n.name + ":0") for n in tf.get_default_graph().as_graph_def().node if n.name.startswith("Logits/") and n.name.endswith('BiasAdd')}
with tf.name_scope('Predictions'):
    predictions = dict(map((lambda kv : (kv[0],tf.nn.softmax(tf.squeeze(kv[1]),name=kv[0]))),logits.items()))

input_checkpoint = FLAGS.model_ckpt_path

output_graph = FLAGS.export_dir+"/frozen_model.pb"#absolute_model_dir + "/frozen_model.pb"

# We clear devices to allow TensorFlow to control on which device it will load operations
clear_devices = True
# We retrieve our checkpoint fullpath
#checkpoint = tf.train.get_checkpoint_state(model_dir)
#input_checkpoint = checkpoint.model_checkpoint_path
# We start a session using a temporary fresh Graph
with tf.Session() as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.Saver()#var_list = slim.get_variables_to_restore())#tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
        [v.name.split(":")[0] for v in predictions.values()] # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))
