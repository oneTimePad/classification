"""
Modeled after the Tensorflow Object Detection API
"""
import tensorflow as tf
from classification.protos import pipeline_pb2
from google.protobuf import text_format

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
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
 # configs["eval_config"] = pipeline_config.eval_config
 # configs["eval_input_config"] = pipeline_config.eval_input_reader
  return configs




configs = get_configs_from_pipeline_file("/home/lie/aiaa/ComputerVision/deeplearning/pipeline_config.config")

input_config = configs["train_input_config"]
model_config = configs["model_config"]
is_training = False

decoded_tensors = input_reader_builder.build(input_config)
classification_model = model_builder.build(model_config,is_training)

#import pdb;pdb.set_trace()
