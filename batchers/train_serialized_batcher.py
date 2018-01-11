
from classification import batcher

class TrainSerializedBatcher(batcher.Batcher):
    """Used at the training step to batch up serialized records"""
    def __init__(self,
                 batch_size
                 number_of_training_examples,
                 fraction_of_examples_in_queue,
                 num_batches_past_min_queue_size = 3,
                 batch_threads = 4):
        """
            Args:
                batch_size: batch size for training step
                number_of_training_examples : total amount of training data
                fraction_of_examples_in_queue: amount of data to hold in RAM
                num_batches_past_min_queue_size : number of batches extra to hold in queue past min queue size
                batch_threads : how many threads should be batching
        """
        self._number_of_training_examples = number_of_training_examples
        self._fraction_of_examples_in_queue = fraction_of_examples_in_queue


        min_queue_size = int(self._number_of_training_examples*self._fraction_of_examples_in_queue)
        batch_capacity = min_queue_size + num_batches_past_min_queue_size * batch_size

        super(TrainSerializedBatcher,self).__init__(batch_size,
                                                    batch_capacity,
                                                    min_after_dequeue,
                                                    num_threads=num_threads,
                                                    shuffle=True)
    def batch_examples(self, tensors_to_batch):
        """See batcher ABC
            Args:
                tensors_to_batch: tensors from TFRecordReader containing serialized examples
            Returns:
                batch: tf.train.batch tensors of serialized examples (dict)
        """
        serialized_examples = tensors_to_batch
        batch = tf.train.shuffle_batch(
                {fields.InputDataFields.serialized: [serialized_examples]},
                batch_size = self._batch_size,
                num_threads = self._num_threads,
                capacity = self._batch_capacity,
                min_after_dequeue = self._min_after_dequeue,
                "train_serialized_queue")

        return batch
