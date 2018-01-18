import tensorflow as tf

"""Inference from Frozen Graph from Tensorflow Object Detection API
"""

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_ckpt',None,'')
tf.app.flags.DEFINE_string('image_path',None,'')

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


with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state(FLAGS.model_ckpt)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
		saver.restore(sess,ckpt.model_checkpoint_path)
	#logits_train = {'alphanumeric':tf.get_default_graph().get_tensor_by_name('alphanumeric_logits/alphanumeric_logits/BiasAdd:0')}
	logits = list(map(tf.nn.softmax,[logits_train[l] for l in logits_train.keys()]))
	img = np.array(Image.open(FLAGS.image_path),dtype='float32')
	results = [res for res in sess.run(logits,feed_dict={'input_image:0':[img]})]
	for r in results[0]:
		print('%.2f'%r)
	print('Softmax: ',[np.argmax(res) for res in results])
