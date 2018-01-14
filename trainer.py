"""
Modeled after the Tensorflow Object Detection API
"""
import tensorflow as tf
from google.protobuf import text_format
from classification.batchers import pre_fetch_batcher
from classification.protos import pipeline_pb2
from classification.builders import input_reader_builder
from classification.builders import model_builder


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
  #configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
 # configs["eval_config"] = pipeline_config.eval_config
 # configs["eval_input_config"] = pipeline_config.eval_input_reader
  return configs

def get_input(input_config,preprocessor = None):
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


configs = get_configs_from_pipeline_file("/home/lie/aiaa/ComputerVision/deeplearning/pipeline_config.config")

input_config = configs["train_input_config"]
model_config = configs["model_config"]
is_training = False


classification_model = model_builder.build(model_config,is_training)
batcher = get_input(input_config,classification_model.preprocess)
