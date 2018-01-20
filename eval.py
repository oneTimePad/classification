import tensorflow as tf
from classification.builders import input_reader_builder
from classification.builders import model_builder
from classification.coordinators import evaluation_coordinator
from classification import helpers
import collections


Helper = helpers.Helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config',None,'')


if not FLAGS.pipeline_config:
    raise ValueError("Must specify pipeline config file")

configs = Helper.get_configs_from_pipeline_file(FLAGS.pipeline_config)

model_config = configs["model_config"]
eval_config = configs["eval_config"]
eval_input_config = configs["eval_input_config"]

eval_ops_dict = collections.OrderedDict({})
scalar_updates = []
label_names = [label.name
                    for label in model_config.multi_task_label ]
starts_from = {label.name: label.starts_from
                    for label in model_config.multi_task_label}
classification_model = model_builder.build(model_config, False)
batcher = Helper.get_inputs(eval_input_config, classification_model.preprocess)
batched_tensors = batcher#batcher.dequeue()
batched_tensors = {label: tensor for label,tensor in batched_tensors.items() if label in label_names or label == "input"}
predictions = classification_model.predict(batched_tensors["input"])
with tf.name_scope("eval"):
    train_loss, train_scalar_updates = Helper.get_loss(predictions, batched_tensors, starts_from)
    train_acc_dict = Helper.get_acc(predictions, batched_tensors, starts_from)
scalar_updates += train_scalar_updates
eval_ops_dict['loss %.2f '] = train_loss
for label in train_acc_dict:
    eval_ops_dict['eval_'+label+"_acc %.2f "] = train_acc_dict[label]

evaluation_coordinator.EvaluationCoordinator(eval_ops_dict,
                                             eval_input_config.\
                                                eval_batch_mode).\
        evaluate(eval_config)
