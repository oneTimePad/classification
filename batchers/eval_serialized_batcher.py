
from classification.batchers import batcher
import tensorflow as tf
import classification.fields as fields

class EvalSerializerBatcher(batcher.Batcher):
    """Used for Eval step to batch up serialized Examples"""
    def __init__(self,
                number_of_eval_examples,
                batch_size = 50,
                name = 'eval_serialized_batcher'):
        """
            Args:
                number_of_eval_examples : how many examples to use for evaluation
        """

        self._number_of_eval_examples = number_of_eval_examples
        self._name = name

        batch_size = self._number_of_eval_examples
        min_after_dequeue = None # not used
        num_threads = 1 # else it becomes non-deterministic
        batch_capacity = self._number_of_eval_examples


        super(EvalSerializerBatcher,self).__init__(batch_size,
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
			                    capacity = self._batch_capacity,
			                    num_threads = self._num_threads,
                                name = self._name)
        self._set_batch_size_to_dict(batch_dict)
        return batch_dict
