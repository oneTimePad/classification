import tensorflow as tf
from classification.extractors import feature_extractor
import tensorflow.contrib.slim as slim
from classification.nets.nasnet import nasnet_mobile_arg_scope, build_nasnet_mobile

class NASNetMobileFeatureExtractor(feature_extractor.FeatureExtractor):

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
        with slim.arg_scope(nasnet_mobile_arg_scope()):
              #NASNet doesn't support resue
              _,end_points = build_nasnet_mobile(
                  preprocessed_inputs,num_classes=None,is_training=self._is_training)
        return end_points['PreLogits']
