import tensorflow as tf
from classification.builders import input_reader_builder
from classification.builders import model_builder
from classification.coordinators import evaluation_coordinator
from classification import helpers
import collections


Helper = helpers.Helper

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('pipeline_config', None, '')


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
losses_dict = {label.name : label.loss_config for label in model_config.multi_task_label}
num_classes_dict = {label.name : label.num for label in model_config.multi_task_label}
has_accuracy = {label.name: label.has_accuracy for label in model_config.multi_task_label}

classification_model = model_builder.build(model_config, False)
batcher = Helper.get_inputs(eval_input_config, classification_model.preprocess)
batched_tensors = batcher#batcher.dequeue()
batched_tensors = {label: tensor for label, tensor in batched_tensors.items() if label in label_names or label == "input"}
tf.summary.image("train",batched_tensors["input"])
predictions = classification_model.predict(batched_tensors["input"])
with tf.name_scope("eval"):
    eval_loss, _ = Helper.get_loss(predictions, batched_tensors, starts_from, losses_dict, num_classes_dict)
    eval_acc_dict = Helper.get_acc(predictions, batched_tensors, starts_from, has_accuracy)
eval_ops_dict['loss %.2f '] = eval_loss
for label in eval_acc_dict:
    eval_ops_dict['eval_' + label + "_acc %.2f "] = eval_acc_dict[label]

confusion_matrices = {}
for label_name in batched_tensors.keys():
    if label_name == "input":
        continue
    label = batched_tensors[label_name]
    if starts_from[label_name] != 0:
        label -= 1
    logit = tf.argmax(predictions[label_name], axis = 1)
    num_classes = num_classes_dict[label_name]

    eval_ops_dict.update({'eval_'+label_name+'_confusion_matrix\n %s\n' : tf.confusion_matrix(label,
                                                               logit,
                                                               num_classes,
                                                               name = "confusion_"+label_name)})


    conf = eval_ops_dict['eval_' + label_name + '_confusion_matrix\n %s\n']

    #precision
    precisions = []
    conf_shape = conf.get_shape()
    for col_i in range(conf_shape[1]):

        actual_num = 0
        sum_num = 0

        for row_i in range(conf_shape[0]):

            if row_i == col_i:
                actual_num = conf[row_i][col_i]

            sum_num += conf[row_i][col_i]

        prec_this_class = actual_num / sum_num
        precisions.append(prec_this_class)

    eval_ops_dict.update({'eval_' + label_name + '_precision %s\n': precisions})


    #recall
    recalls = []
    for row_i in range(conf_shape[1]):

        actual_num = 0
        sum_num = 0

        for col_i in range(conf_shape[0]):

            if row_i == col_i:
                actual_num = conf[row_i][col_i]

            sum_num += conf[row_i][col_i]

        recall_this_class = actual_num / sum_num
        recalls.append(recall_this_class)

    eval_ops_dict.update({'eval_' + label_name + '_recall %s\n': recalls})





evaluation_coordinator.EvaluationCoordinator(eval_ops_dict,
                                             eval_input_config.\
                                                eval_batch_mode).\
        evaluate(eval_config)
