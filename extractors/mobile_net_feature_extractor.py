import tensorflow as tf
from classification.extractors import feature_extractor
import tensorflow.contrib.slim as slim
from classification.nets.mobilenet_v1 import mobilenet_v1_arg_scope,mobilenet_v1


class MobileNetFeatureExtractor(feature_extractor.FeatureExtractor):

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
		with slim.arg_scope(mobilenet_v1_arg_scope(is_training=self._is_training)):
			_,end_points = mobilenet_v1(
				preprocessed_inputs,num_classes=1001,is_training=self._is_training, reuse = self._reuse)
		return end_points['PreLogits']
