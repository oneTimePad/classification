
from abc import ABCMeta
from abc import abstractmethod

class Batcher(object):
    __metaclass__ = ABCMeta
    """
        Constructs a batcher for training or eval steps
    """
    def __init__(self,
                 batch_size,
                 batch_capacity,
                 min_after_dequeue,
                 num_threads=1,
                 shuffle=False):

        self._batch_size = batch_size
        self._batch_capacity = batch_capacity
        self._min_after_dequeue = min_after_dequeue
        self._num_threads = num_threads

    @abstractmethod()
    def batch_examples(self,tensors_to_batch):
        """Batches up examples and returns a tensor representing a batch

           Args:
            tensors_to_batch: data to batch, could be list, dict, or tensor
           Returns:
            tensor_dict: A dict containing batched tensors, see fields.py
        """
        pass
