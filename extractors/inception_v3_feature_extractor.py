

from classification.extractors import feature_extractor

class InceptionV3FeatureExtractor(feature_extractor.FeatureExtractor):

    def get_trainable_variables():
        pass

    def get_update_ops():
        pass

    def get_pre_ops():
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
        with slim.arg_scope(inception_v3_arg_scope()):
              _,end_points = inception_v3(
    image_batch,num_classes=1001,is_training=self._is_training,reuse=self._reuse)
        return end_points('PreLogits')
