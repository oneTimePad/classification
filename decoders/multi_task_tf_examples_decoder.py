import tensorflow as tf
from classification import fields

<<<<<<< HEAD
=======
TYPE_MAP = {
    "int64": tf.int64,
    "float32": tf.float32,
    "string" : tf.string
}

>>>>>>> 43c89f6... fix
class MultiTaskTfExamplesDecoder(object):

    def __init__(self,multi_task_labels,image_spatial_size):
        """Decodes Examples from TFRecord files

           Args:
             multi_task_labels: list of strings representing the multi_task label name (from protobuf)
        """
        self._keys_to_features = {
            'input' : tf.FixedLenFeature([],tf.string)
        }

        for label in multi_task_labels:
            self._keys_to_features.update({
                label : tf.FixedLenFeature([],tf.int64)
            })

        self._image_height, self._image_width, self._channels = image_spatial_size
        self._multi_task_labels = multi_task_labels

    def decode(self,batched_serialized_tensors,batch_size):
        """Decodes the input from batch of serialized tensors
           Formats and reshapes image
           Args:
            batched_serialized_tensors: tensor output from Batcher containing read in
                serialized tensors

          Returns:
            batched_decoded_tensors: dict of batches of decoded TFRecords of batch_size
        """

        #faster to decode tensors as a batch
        batched_decoded_tensors = tf.parse_example(batched_serialized_tensors[fields.InputDataFields.serialized],
                                                    self._keys_to_features)
<<<<<<< HEAD
=======
        #Decode and cast tensors if needed
        for label in self._multi_task_labels:
            tensor = batched_decoded_tensors[label.name]
            #only strings need t obe decoded
            if label.dtype == "string":
                if label.decodetype:
                    tensor = tf.decode_raw(tensor, TYPE_MAP[label.decodetype])
                else:
                    raise ValueError("string type must have a type to be decoded to.")
            if label.casttype:
                tensor = tf.cast(tensor, TYPE_MAP[label.casttype])

            if label.shape:
                tensor = tf.reshape(tensor, [batch_size,*label.shape])
                tensor.set_shape([batch_size, *label.shape])

            batched_decoded_tensors[label.name] = tensor

        #input is handlded separately
>>>>>>> 43c89f6... fix
        image_float = tf.cast(
                            tf.decode_raw(batched_decoded_tensors['input'],
                                          tf.uint8),
                            tf.float32)
        image_float = tf.reshape(image_float,[batch_size,
                                              self._image_height,
                                              self._image_width,
                                              self._channels])
        image_float.set_shape([batch_size,
                               self._image_height,
                               self._image_width,
                               3])

        batched_decoded_tensors['input'] = image_float

        return batched_decoded_tensors
