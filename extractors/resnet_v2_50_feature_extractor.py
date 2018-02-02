import tensorflow as tf
from classification.extractors import feature_extractor
import tensorflow.contrib.slim as slim
from classification.nets.resnet_v2 import resnet_arg_scope, resnet_v2_50

class ResNetV250FeatureExtractor(feature_extractor.FeatureExtractor):

    def get_pre_ops(self):
        pass

    def preprocess(self,image):
        """
        Preprocess inputs (per image) before going into model

        Args:
            image: input to model
        Returns:
            processed: processed batch of tensors
        """
        scaled = tf.multiply(image,2.0/255.0)
        scaled = tf.subtract(scaled,1.0)
        processed = scaled
        return processed


    def predict(self,preprocessed_inputs):
        """
        Runs inference on model

        Args:
            preprocessed_inputs: processed batch to go to model
        Returns:
            pre_logits: layer right before the logits
        """
        with slim.arg_scope(resnet_arg_scope()):
              _,end_points = resnet_v2_50(
                  preprocessed_inputs,num_classes=None,is_training=self._is_training,reuse=self._reuse)
        return end_points['PreLogits']
