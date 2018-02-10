import tensorflow as tf
from classification import fields
from classification.builders.feature_extractor_map import NAME_TO_FEATURE_EXTRACTOR
from classification.protos import model_pb2
def generate_logits(pre_logits,
                    filter_map,
                    apply_logits_map,
                    reuse):
    """Generates logits from pre logits of FeatureExtractor

       Args:
        pre_logits : output of FeatureExtractor
        filter_map : dict mapping label names to number of classes
        reuse: whether to reuse logit variables

      Returns:
        logits: dict mapping label to logits
    """

    logits = {}

    from functools import reduce
    import operator
    #validation on keys in filter_map and pre_logits (if dict)
    if isinstance(pre_logits,dict) and not bool(reduce(operator.mul,[k in pre_logits for k in filter_map.keys()])):
        raise Exception("Keys in PreLogits must match labels in proto file")

    with tf.variable_scope('Logits',reuse=reuse):
        for label, filters  in filter_map.items():
            if apply_logits_map[label]:
                pre_logits_expanded, kernel_size = _kernel_size(pre_logits[label] if isinstance(pre_logits,dict) else pre_logits)
                layer = tf.layers.conv2d(pre_logits_expanded,
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        				                 kernel_size=kernel_size,
                                         strides=1,
                                         filters=filters,
                                         padding='VALID',
                                         name=label)
                logits[label] = tf.squeeze(layer)
            else:
                logits[label] = pre_logits[label]

    return logits

def _kernel_size(pre_logits):
    """Retrieve kernel size and expands input rank if necessary
       Args:
         pre_logits: Tensor
       Returns:
         pre_logits_expanded: pre_logits with increased rank
         kernel_size: size of symmetric kernel
    """
    tensor_rank = len(pre_logits.get_shape())
    #expand dims to rank 4 tensor if needed
    if tensor_rank < 4:
        increase_dims = 4 - tensor_rank
        for _ in range(increase_dims):
            pre_logits = tf.expand_dims(pre_logits,[1])

    #for full-connection table (i.e. mimics a fully-connected layer)
    kernel_size= int(pre_logits.get_shape()[1])

    return pre_logits,kernel_size

def build(model_config,
          is_training,
          reuse = False):
    """Builds FeatureExtraction model based on config

       Args:
        model_build_config: config for model from protobuf
        and builds logits from configuration

       Returns:
        FeatureExtractor object
    """

    if not isinstance(model_config,model_pb2.Model):
        raise ValueError('model_build_config not type'
                        'model_pb2.Model')
    Model = NAME_TO_FEATURE_EXTRACTOR[model_config.extractor]
    built_model = Model(is_training,reuse)

    label_to_classes = {
                    entry.name :
                        entry.num for entry in model_config.multi_task_label
    }

    label_to_apply_logits = {
                    entry.name:
                        entry.apply_logits for entry in model_config.multi_task_label
    }

    #hook predict function from feature_extractor to add logits
    def logit_wrapper(predict_fn):
        nonlocal label_to_classes
        nonlocal label_to_apply_logits
        nonlocal reuse

        def logits(*args,**kwargs):
            #undo hook
            Model.predict = Model._predict
            return generate_logits(
                         predict_fn(*args, **kwargs),
                         label_to_classes,
                         label_to_apply_logits,
                         reuse)
        return logits

    #save original
    Model._predict = Model.predict
    #create hook to generate logits
    Model.predict = logit_wrapper(Model.predict)

    return built_model
