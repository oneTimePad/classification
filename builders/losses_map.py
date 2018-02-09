import tensorflow as tf

def _xentropy_loss_op(logit, label, num_classes):
    """
    Constructs Cross Entropy loss
    Args:
    logits -> output units of network rank 0
    label  -> matching labels rank 0
    name   -> name of operation
    return:
     		mean cross entropy
    """
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,
                                                          labels=label)

# L1 Loss
def _l1_loss_op(logit, label, num_classes):
    """
    Constructs Cross Entropy loss
    Args:
    logits -> output units of network rank 0
    label  -> matching labels rank 0
    name   -> op name
    return:
     		mean l1
    """

    label = tf.one_hot(label, num_classes)
    return tf.losses.absolute_difference(labels = label,
                                         predictions = logit)

def _l2_loss_op(logit, label, num_classes):
    """
    Constructs Cross Entropy loss
    Args:
    logits -> output units of network rank 0
    label  -> matching labels rank 0
    return:
     		mean l1
    """
    #label = tf.one_hot(label, num_classes)
    return tf.square(logit-label)

NAME_TO_LOSS_MAP = {
    "cross_entropy": _xentropy_loss_op,
    "l1": _l1_loss_op,
    "l2": _l2_loss_op
}
