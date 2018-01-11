from classification.batchers import batcher
import tensorflow as tf
import classification.fields as fields

class BatchEvalSerializedBatcher(batcher.Batcher):
    """Used for Evaluating in batches and batch up serialized examples"""
    def __init__(self,
                 batch_size,
                 number_of_eval_examples,
                 fraction_of_examples_in_queue,
                 num_batches_past_min_queue_size):
        """
            Args:
                number_of_eval_examples : how many examples to use for evaluation
        """

        self._number_of_training_examples = number_of_training_examples
        self._fraction_of_examples_in_queue = fraction_of_examples_in_queue

        num_threads = 1

        min_after_dequeue = int(self._number_of_eval_examples*self._fraction_of_examples_in_queue)
        batch_capacity = min_after_dequeue + num_batches_past_min_queue_size * batch_size

        super(BatchEvalSerializedBatcher,self).__init__(batch_size,
                                                    batch_capacity,
                                                    min_after_dequeue,
                                                    num_threads = num_threads,
                                                    shuffle = False)


    def batch_examples(self,tensors_to_batch):
        """See batcher ABC
            Args:
                tensors_to_batch: tensors from TFRecordReader containing serialized examples
            Returns:
                batch: tf.train.batch tensors of serialized examples (dict)
        """

        serialized_examples = tensors_to_batch
        batch_dict = tf.train.batch({fields.InputDataFields.serialized:
                                     serialized_examples},
                                self._batch_size,
			                    capacity = slef._batch_capacity,
			                    num_threads = self._num_threads,
                                name = "batched_eval_serialized_queue")
        self._set_batch_size_to_dict(batch_dict)
        return batch_dict
