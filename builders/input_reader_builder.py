"""
Modeled after the Tensorflow Object Detection API
"""
import os

"""Input reader builder

Creates data sources for ClassificationModels from an InputReader config.
See input_reader.proto for options
"""


import tensorflow as tf

from classification.decoders import multi_task_tf_examples_decoder as tf_exam
from classification.protos import input_reader_pb2


parallel_reader = tf.contrib.slim.parallel_reader

def build(input_reader_config,training=False):
    """Builds a tensor dictionary based on InputReader config

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on input_reader_config.

    Raises:
        ValueError: On invalid input reader proto
        ValueError: If no input paths are specified
    """
    #import pdb;pdb.set_trace()
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not type'
                         'input_reader_pb2.InputReader.')

    if input_reader_config.WhichOneof("input_reader") == 'tf_record_input_reader':
        config = input_reader_config.tf_record_input_reader
        if not config.input_path:
            raise ValueError('At least one input path must be specified in'
                            '`input_reader_config.`')

        records = [os.path.join(config.input_path,f) for f in os.listdir(config.input_path)]
        #doesn't control number of epochs very nicely
        filenames_queue = tf.train.string_input_producer(records,shuffle=training,num_epochs=None)
        reader = tf.TFRecordReader()
        _,string_tensor = reader.read(filenames_queue)
        #TODO add batcher
        decoder = tf_exam.MultiTaskTfExamplesDecoder(
            input_reader_config.multi_task_label_name,
            (input_reader_config.image_height,input_reader_config.image_width))

        return decoder.decode(string_tensor,11)

    raise ValueError('Unsupported input_reader_config.')
