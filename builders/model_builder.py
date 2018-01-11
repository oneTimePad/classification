
from classification import fields
from classification.builders.feature_extractor_map import NAME_TO_FEATURE_EXTRACTOR


def generate_logits(pre_logits,
                    filter_map,
                    reuse):
    """Generates logits from pre logits of FeatureExtractor

       Args:
        pre_logits : output of FeatureExtractor
        filter_map : dict mapping label names to number of classes
        reuse: whether to reuse logit variables

      Returns:
        logits: dict mapping label to logits
    """
    tensor_rank = len(pre_logits.get_shape())
    #expand dims to rank 4 tensor if needed
    if tensor_rank < 4:
        increase_dims = 4 - tensor_rank
        for _ in range(increase_dims):
            pre_logits = tf.expand_dims(pre_logits,[1])

    #for full-connection table (i.e. mimics a fully-connected layer)

    kernel_size= int(pre_logits.get_shape()[1])

    logits = {}

    with tf.variable_scope('logits',reuse=reuse):
        for label, filters  in filter_map:
            layer = tf.layers.conv2d(pre_logits,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
    				                 kernel_size=kernel_size,
                                     strides=1,
                                     filters=filter_map,padding='VALID',name=label)
            logits[label] = tf.squeeze(layer)

    return logits


def build(model_build_config,
          is_training,
          reuse = False):
    """Builds FeatureExtraction model based on config

       Args:
        model_build_config: config for model from protobuf
        and builds logits from configuration

       Returns:
        FeatureExtractor object
    """
    if not isinstance(model_build_config,model_pb2.Model):
        raise ValueError('model_build_config not type'
                        'model_pb2.Model')
    model = NAME_TO_FEATURE_EXTRACTOR[model_build_config.extractor]
    built_model = model(is_training,reuse)

    label_to_classes = {
                    entry.name :
                        entry.num for entry in model_config.multi_task_label
    }

    def logit_wrapper(pre_logits):
        nonlocal label_to_classes
        nonlocal reuse

        return generate_logits(pre_logits,label_to_classes,reuse)

    #create hook to generate logits
    built_model.predict = logit_wrapper

    return built_model
