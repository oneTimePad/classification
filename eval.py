import tensorflow as tf
from classification.builders import input_reader_builder
from classification.builders import model_builder
from classification.coordinators import evaluation_coordinator
from classification import helpers
import collections


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')


if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

configs = helpers.get_configs_from_pipeline_file(FLAGS.pipeline_config)

eval_config = configs["eval_config"]
eval_input_config = configs["eval_input_config"]

eval_ops_dict = collections.OrderedDict({})
scalar_updates = []
with tf.name_scope("eval"):
    classification_model = model_builder.build(model_config, False)
    batcher = helpers.get_inputs(eval_input_config, classification_model.preprocess)
    batched_tensors = batcher.dequeue()
    predictions = classification_model.predict(batched_tensors["input"])
    train_loss, train_scalar_updates = helpers.get_loss(predictions, batched_tensors)
    train_acc_dict = helpers.get_acc(predictions, batched_tensors)
    scalar_updates += train_scalar_updates
    eval_ops_dict['loss %.2f '] = train_loss
    for label in train_acc_dict:
        eval_ops_dict['eval_'+label+"_acc %.2f "] = train_acc_dict[label]

evaluation_coordinator.EvaluationCoordinator(eval_input_config.\
                                                eval_batch_mode).\
        evaluate(eval_config,
                eval_ops_dict)
