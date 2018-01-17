"""
Modeled after the Tensorflow Object Detection API
"""
import tensorflow as tf
from google.protobuf import text_format
from classification.batchers import pre_fetch_batcher
from classification.protos import pipeline_pb2
from classification.builders import input_reader_builder
from classification.builders import model_builder
from classification.coordinators import training_coordinator
import collections

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')

def get_configs_from_pipeline_file(pipeline_config_path):
  """Reads configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

  configs = {}
  configs["model_config"] = pipeline_config.model
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
  configs["eval_config"] = pipeline_config.eval_config
  configs["eval_input_config"] = pipeline_config.eval_input_reader
  return configs

def _xentropy_loss_op(logit,label,name):
	"""
		Constructs Cross Entropy loss
		Args:
			logits -> output units of network rank 0
			label  -> matching labels rank 0
		return:
	 		mean cross entropy
	"""
	return tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logit,
					labels=label),name=name)

def _eval_op(logits,label,name):
	"""
		Construct accuracy operation
		Args:
			logits -> output units of network rank 0
			label  -> matching labels rank 0
		return:
	 		mean accuracy op
	"""
	with tf.name_scope(name):
		return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,label,1),tf.float32))


def get_inputs(input_config,preprocessor = None):
    """Generate input pipeline

       Args:
            input_config: configuration for eval or training
            preprocessor: the preprocessor method of the classification model
       Returns:
            batcher: the output of the total pipeline
    """
    #TODO: add support for data augmentation
    decoded_tensors = input_reader_builder.build(input_config)

    if preprocessor:
        decoded_tensors['input'] = tf.map_fn(preprocessor,
                                             decoded_tensors['input'])

    batcher = pre_fetch_batcher.PreFetchBatcher(input_config. \
                                    prefetch_queue_capacity). \
                batch_examples(decoded_tensors)

    return batcher


def get_acc(predictions,
            batched_tensors):
    """Generates accuracy operations

       Args:
             predictions: last layer output of classification_model (from predict)
             batched_tensors: output of get_inputs
       Returns:
             accs_dict : dict mapping labels to accuracy operations
    """
    accs_dict = {}
    labels = {label:
                tensor for label, tensor in batched_tensors.items()
                            if label != "input"}
    for label, tensor in labels.items():
        acc = _eval_op(predictions[label],
                                 tensor-1,
                                 name=label)
        accs_dict[label] = acc

    return accs_dict
def get_loss(predictions,
             batched_tensors):
    """Generates combined loss and evaluation

       Args:
          predictions: last layer output of classification_model (from predict)
          batched_tensors: output of get_inputs
       Returns:
          loss: the combined loss for all labels
          scalar_updates : any updates for keeping stats
    """
    losses = []
    labels = {label:
                tensor for label, tensor in batched_tensors.items()
                            if label != "input"}
    for label, tensor in labels.items():
        loss = _xentropy_loss_op(predictions[label],
                                 tensor-1,
                                 name=label)
        tf.losses.add_loss(loss,
                            tf.GraphKeys.LOSSES)
        losses.append(loss)
    loss = tf.reduce_sum(losses,
                         name = "total_loss")

    #get moving average
    loss_avg = tf.train.ExponentialMovingAverage(0.9,name='moving_avg')
    #get the moving average ops (create shadow variables)

    loss_avg_op = loss_avg.apply([loss])
    #log loss and shadow variables for avg loss
    #tf.summary.scalar(loss.op.name+' (raw)',loss)
    tf.summary.scalar(loss.op.name,loss_avg.average(loss))
    scalar_updates = [loss_avg_op]

    return loss, scalar_updates



if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

configs = get_configs_from_pipeline_file(FLAGS.pipeline_config)


train_input_config = configs["train_input_config"]
model_config = configs["model_config"]
train_config = configs["train_config"]
eval_config = configs["eval_config"]
eval_input_config = configs["eval_input_config"]


eval_ops_dict = collections.OrderedDict({})
scalar_updates = []
with tf.name_scope("train"):
    is_training = tf.placeholder_with_default(False,shape=(),name="is_training")
    classification_model = model_builder.build(model_config, is_training)
    batcher = get_inputs(train_input_config, classification_model.preprocess)
    batched_tensors = batcher.dequeue()
    predictions = classification_model.predict(batched_tensors["input"])
    train_loss, train_scalar_updates = get_loss(predictions, batched_tensors)
    train_acc_dict = get_acc(predictions, batched_tensors)
    scalar_updates += train_scalar_updates
    eval_ops_dict['loss %.2f '] = train_loss
    for label in train_acc_dict:
        eval_ops_dict['train_'+label+"_acc %.2f "] = train_acc_dict[label]
    optimizer = tf.train.AdamOptimizer(0.01)

if train_config.eval_while_training:
    with tf.name_scope("test"):
        classification_model_test = model_builder.build(model_config,
                                                         is_training = False,
                                                         reuse = True)
        batcher = get_inputs(eval_input_config, classification_model_test.preprocess)
        predictions = classification_model_test.predict(batched_tensors["input"])
        test_loss, test_scalar_updates = get_loss(predictions, batched_tensors)
        test_acc_dict = get_acc(predictions, batched_tensors)
        scalar_updates += test_scalar_updates
        for label in train_acc_dict:
            eval_ops_dict['test_'+label+"_acc %.2f "] = test_acc_dict[label]
import pdb;pdb.set_trace()
train_coord = training_coordinator.TrainingCoordinator().\
                    train(train_config,
                          train_loss,
                          scalar_updates,
                          optimizer,
                          eval_ops_dict = eval_ops_dict)
