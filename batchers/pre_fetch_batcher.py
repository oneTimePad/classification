
from classification import batcher

class PreFetchBatcher(batcher.Batcher):
    """Used after examples are deserialzied to batch of batches of preprocessed tensors"""
    def __init__(self,
                 capacity):

            self._batch_capacity = capacity

            #most args are not used
            super(PreFetchBatcher,self).__init__(None,
                                                 self._batch_capacity,
                                                 None)

    def batch_examples(self, tensors_to_batch):
        """See batcher ABC
            Args:
                tensors_to_batch: tensors from TFRecordReader containing serialized examples
            Returns:
                batch: dict containing one batch
        """
        names = list(tensor_dict.keys())
        dtypes = [t.dtype for t in tensor_dict.values()]
        shapes = [t.get_shape() for t in tensor_dict.values()]
        prefetch_queue = tf.FIFIQueue(capacity,
                                    dtypes=dtypes,
                                    shapes=shapes,
                                    names=names,
                                    name='prefetch_queue')
        enqueue_op = prefetch_queue.enqueue(tensor_to_batch)
        # start the queue runners that enqueue tensors from the batch( pre-fetch preprocessed batches)
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            prefetch_queue,[enqueue_op]))
        tf.summary.scalar('queue/%s/fraction_of_%d_full' % (prefetch_queue_name,
                                                        capacity),
                     tf.to_float(prefetch_queue.size()) * (1. / capacity))
        return prefetch_queue
