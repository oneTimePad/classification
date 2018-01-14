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

from classification.batchers import train_serialized_batcher
from classification.batchers import eval_serialized_batcher
from classification.batchers import batched_eval_serialized_batcher
#import pdb;pdb.set_trace()


parallel_reader = tf.contrib.slim.parallel_reader

def build(input_reader_config):
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

        shuffling = input_reader_config.shuffle
        if not input_reader_config.num_examples:
            raise ValueError('Must specify number of data examples')
        num_examples = input_reader_config.num_examples
        records = [os.path.join(config.input_path,f) for f in os.listdir(config.input_path)]

        #doesn't control number of epochs very nicely
        filenames_queue = tf.train.string_input_producer(records,shuffle=input_reader_config.shuffle,num_epochs=None)
        reader = tf.TFRecordReader()
        _,string_tensor = reader.read(filenames_queue)

        fraction_of_examples_in_queue = input_reader_config.fraction_of_examples_in_queue
        num_batches_past_min_queue_size = input_reader_config.num_batches_past_min_queue_size

        if shuffling:
            if not input_reader_config.batch_size:
                raise ValueError('batch_size must be specified when training.')
            batch_size = input_reader_config.batch_size
            num_threads = input_reader_config.num_threads

            batcher = train_serialized_batcher.TrainSerializedBatcher(batch_size,
                                             num_examples,
                                             fraction_of_examples_in_queue,
                                             num_batches_past_min_queue_size,
                                             num_threads).\
                        batch_examples(string_tensor)

        elif not input_reader_config.eval_batch_mode:
            batcher = batchersEvalSerializerBatcher(num_examples).\
                        batch_examples(string_tensor)

        elif input_reader_config.eval_batch_mode:
            if not input_reader_config.batch_size:
                raise ValueError('batch_size must be specified when doing batched eval.')
            batcher = batchers.BatchEvalSerializedBatcher(batch_size,
                              num_examples,
                              fraction_of_examples_in_queue,
                              num_batches_past_min_queue_size).\
                        batch_examples(string_tensor)

        else:
            raise ValueError("Please check for InputReader config...")

        decoder = tf_exam.MultiTaskTfExamplesDecoder(
            input_reader_config.multi_task_label_name,
            (input_reader_config.image_height,input_reader_config.image_width))

        return decoder.decode(batcher,batcher["batch_size"])

    raise ValueError('Unsupported input_reader_config.')
