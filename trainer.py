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
 # configs["eval_config"] = pipeline_config.eval_config
 # configs["eval_input_config"] = pipeline_config.eval_input_reader
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

def get_loss(logits,
             batcher):
    """Generates combined losss

       Args:
          logits: last layer output of classification_model (from predict)
          batcher: output of get_inputs
       Returns:
          loss: the combined loss for all labels
          scalar_updates : any updates for keeping stats
    """
    batched_tensors = batcher.dequeue()
    predictions = logits(batched_tensors["input"])
    losses = []
    labels = {label:
                tensor for label, tensor in batched_tensors.items()
                            if label != "input"}
    for label, tensor in labels.items():
        loss = _xentropy_loss_op(predictions[label],
                                 tensor,
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

input_config = configs["train_input_config"]
model_config = configs["model_config"]
train_config = configs["train_config"]


with tf.name_scope('train'):
    is_training = tf.placeholder_with_default(False,shape=(),name="is_training")
    classification_model = model_builder.build(model_config, is_training)
    batcher = get_inputs(input_config, classification_model.preprocess)
    loss, scalar_updates = get_loss(classification_model.predict, batcher)
    optimizer = tf.train.AdamOptimizer(0.01)

train_coord = training_coordinator.TrainingCoordinator().\
                    train(train_config,
                          loss,
                          scalar_updates,
                          optimizer)
