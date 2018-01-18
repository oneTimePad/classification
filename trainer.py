"""
Modeled after the Tensorflow Object Detection API
"""
import tensorflow as tf
from classification.builders import model_builder
from classification.coordinators import training_coordinator
from classification import helpers
import collections

Helper = helpers.Helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')


if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

configs = Helper.get_configs_from_pipeline_file(FLAGS.pipeline_config)


train_input_config = configs["train_input_config"]
model_config = configs["model_config"]
train_config = configs["train_config"]
eval_config = configs["eval_config"]
eval_input_config = configs["eval_input_config"]


eval_ops_dict = collections.OrderedDict({})
scalar_updates = []
""" Training Model """
with tf.name_scope("train") as scope:
    is_training = tf.placeholder_with_default(False,shape=(),name="is_training")
classification_model = model_builder.build(model_config, is_training)
batcher = Helper.get_inputs(train_input_config, classification_model.preprocess)
batched_tensors = batcher.dequeue()
predictions = classification_model.predict(batched_tensors["input"])
with tf.name_scope(scope):
    train_loss, train_scalar_updates = Helper.get_loss(predictions, batched_tensors)
    train_acc_dict = Helper.get_acc(predictions, batched_tensors)
scalar_updates += train_scalar_updates
eval_ops_dict['loss %.2f '] = train_loss
for label in train_acc_dict:
    eval_ops_dict['train_'+label+"_acc %.2f "] = train_acc_dict[label]
optimizer = tf.train.AdamOptimizer(0.01)

""" Testing Model """
if train_config.eval_while_training:
    classification_model_test = model_builder.build(model_config,
                                                     is_training = False,
                                                     reuse = True)
    batcher = Helper.get_inputs(eval_input_config, classification_model_test.preprocess)
    predictions = classification_model_test.predict(batched_tensors["input"])
    with tf.name_scope("test"):
        test_loss, test_scalar_updates = Helper.get_loss(predictions, batched_tensors)
        test_acc_dict = Helper.get_acc(predictions, batched_tensors)
    scalar_updates += test_scalar_updates
    for label in train_acc_dict:
        eval_ops_dict['test_'+label+"_acc %.2f "] = test_acc_dict[label]

train_coord = training_coordinator.TrainingCoordinator().\
                    train(train_config,
                          train_loss,
                          scalar_updates,
                          optimizer,
                          eval_ops_dict = eval_ops_dict)
