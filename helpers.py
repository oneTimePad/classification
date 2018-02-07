from classification.protos import pipeline_pb2
import tensorflow as tf
from google.protobuf import text_format
from classification.batchers import pre_fetch_batcher
from classification.builders import input_reader_builder

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


class Helper:
    """Contains helper functions for creating various training
        and eval utils
    """
    @staticmethod
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

    @staticmethod
    def get_inputs(input_config,preprocessor = None):
        """Generate input pipeline

           Args:
                input_config: configuration for eval or training
                preprocessor: the preprocessor method of the classification model
           Returns:
                batcher: the output of the total pipeline
        """
        #TODO: add support for data augmentation
        with tf.device('/cpu:0'):
            decoded_tensors = input_reader_builder.build(input_config)

            if preprocessor:
                decoded_tensors['input'] = tf.map_fn(preprocessor,
                                                 decoded_tensors['input'])

            #batcher = pre_fetch_batcher.PreFetchBatcher(input_config. \
            #                            prefetch_queue_capacity). \
            #        batch_examples(decoded_tensors)

        return decoded_tensors

    @staticmethod
    def get_acc(predictions,
                batched_tensors,
                starts_from):
        """Generates accuracy operations

           Args:
                 predictions: last layer output of classification_model (from predict)
                 batched_tensors: output of get_inputs
                 starts_from: start label index
           Returns:
                 accs_dict : dict mapping labels to accuracy operations
        """
        accs_dict = {}
        labels = {label:
                    tensor for label, tensor in batched_tensors.items()
                                if label != "input"}
        for label, tensor in labels.items():
            acc = _eval_op(predictions[label],
                                     tensor-starts_from[label],
                                     name=label)
            accs_dict[label] = acc

        return accs_dict

    @staticmethod
def get_loss(predictions,
                 batched_tensors,
                 starts_from_dict, loss_string_dict, num_classes_dict):

        """Generates combined loss and evaluation

           Args:
              predictions: last layer output of classification_model (from predict)
              batched_tensors: output of get_inputs
              starts_from: start label index
              starts_from_dict: start label index
              loss_string_dict: dict mapping label name to loss string name
      loss_config: inputted loss function
           Returns:
              loss: the combined loss for all labels
              scalar_updates : any updates for keeping stats
        """
        losses = []
        labels = {label:
                    tensor for label, tensor in batched_tensors.items()
                                if label != "input"}


        loss_function_dict = {}
        for key, value in loss_string_dict.items():
            loss_function_dict[key] = losses_map.NAME_TO_LOSS_MAP[value]



        for label, tensor in labels.items():
            loss = loss_function_dict[label](predictions[label], tensor-starts_from_dict[label], num_classes_dict[label])
            loss = _loss_op(loss, name = label)
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
