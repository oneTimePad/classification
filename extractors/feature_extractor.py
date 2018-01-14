


from abc import ABCMeta
from abc import abstractmethod



class FeatureExtractor(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                is_training,
                reuse=False):
        self._is_training = is_training
        self._reuse = reuse

    @abstractmethod
    def get_pre_ops(self):
        """
            Operations to peform before restoring

            Returns:
                pre_ops
        """
        pass

    @abstractmethod
    def preprocess(self,image):
        """
        Preprocess inputs before going into model

        Args:
            image: input to model
        Returns:
            processed: processed batch of tensors
        """
        pass

    @abstractmethod
    def predict(self,preprocessed_inputs):
        """
        Runs inference on model

        Args:
            preprocessed_inputs: processed batch to go to model
        Returns:
            logits: dict of logits keys by strings in fields.InputDataFields.labels
        """
        pass
