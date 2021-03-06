import os
from functools import reduce
import operator
import tensorflow as tf
import tensorflow.contrib.slim as slim
from classification.coordinators import logger_hook
from classification.protos import train_pb2
class TrainingCoordinator(object):
    def __init__(self):
        self._global_step = tf.train.get_or_create_global_step()

    def train(self,
                train_config,
                loss,
                scalar_updates,
                optimizer,
                eval_ops_dict=None,
                pre_ops = None):
        """Coordinates the Training stage
                Args:
                  train_config: protobuf configuration for training
                  loss: loss defined by FeatureExtractor
                  scalar_updates: any scalar update ops to peform while training
                  optimizer: the optimizer to use for training
                  eval_ops_dict: a dict mapping format strings to evaluation operations
                  pre_ops: any operations to before before training after restoring
        """
        if not isinstance(train_config, train_pb2.TrainConfig):
            raise ValueError('train_config not type'
                                'train_pb2.TrainConfig.')

        if train_config.eval_while_training and not eval_ops_dict:
            raise ValueError("Can't eval and train without eval ops")

        fine_tune = False
        """Check whether classification ckpt dir exists. If not create it"""
        if not os.path.exists(
            os.path.join(train_config.from_classification_checkpoint)):
            print("TENSORFLOW INFO: Classification Checkpoint doesn't exist. Created checkpoint dir. ")
            os.mkdir(train_config.from_classification_checkpoint)
            #we must be fine tuning
            fine_tune = True
        else:
            print("TENSORFLOW INFO: Classification Checkpoint exists...restoring from it.")

        #this however will restore all variables if nothing is specified
        vars_to_restore = None

        if fine_tune and train_config.fine_tune_checkpoint:
                #things we always need to exclude
                exclude_list = ["train/total_loss","test_1/total_loss","Logits",'eval','global_step']
                if train_config.exclude_from_fine_tune:
                    exclude_list+=list(train_config.exclude_from_fine_tune)
                vars_to_restore = slim.get_variables_to_restore(exclude = exclude_list)
                #TODO add restore from map

        saver = tf.train.Saver(var_list = vars_to_restore) if vars_to_restore else None

        def flatten(l):
            flat = []
            for sublist in l:
                if isinstance(sublist,list):
                    for item in sublist:
                        flat.append(item)
                else:
                    flat.append(sublist)
            return flat

        #if variables to train is not specified, just train them all
        if not train_config.scopes_or_variables_to_train:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        #if vars to train is specified, then just train those
        else:
          train_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope = s)
                            for s in train_config.scopes_or_variables_to_train]

        train_vars = flatten(train_vars)
        #user must specify the scope
        if train_config.scopes_or_names_for_update_ops and train_config.scopes_or_names_for_update_ops[0] != "all":
            update_ops = [tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                            scope = s)
                            for s in train_config.scopes_or_names_for_update_ops]
            update_ops = flatten(update_ops)

        elif train_config.scopes_or_names_for_update_ops and train_config.scopes_or_names_for_update_ops[0] == "all":
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if update_ops:
            with tf.control_dependencies(reduce(operator.concat,[scalar_updates,update_ops])):
                train_op = optimizer.minimize(loss,
                                              var_list = train_vars,
                                              global_step = self._global_step)
        else:
            with tf.control_dependencies(scalar_updates):
                train_op = optimizer.minimize(loss,
                                              var_list = train_vars,
                                              global_step = self._global_step)
        print(update_ops)
        checkpoint_dir = train_config.from_classification_checkpoint

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        file_writer = tf.summary.FileWriter(checkpoint_dir,tf.get_default_graph())
        with tf.train.MonitoredTrainingSession(save_checkpoint_secs = train_config.\
                                                                          keep_checkpoint_every_n_minutes*60,
                                               checkpoint_dir = checkpoint_dir,
                                               hooks = [ tf.train.StopAtStepHook(num_steps = train_config.\
                                                                                                 num_steps),
                                               logger_hook.LoggerHook(eval_ops_dict,
                                               train_config.\
                                                   log_frequency,
                                                   self._global_step)],
                                               config = config) as mon_sess:
            if fine_tune and train_config.fine_tune_checkpoint and saver:
                saver.restore(mon_sess,train_config.fine_tune_checkpoint)
            if pre_ops:
                mon_sess.run(pre_ops)
            for v in train_vars:
                print(v.name)
            print("TENSORFLOW INFO: Proceeding to training stage")
            while not mon_sess.should_stop():
                mon_sess.run(train_op,feed_dict = {'train/is_training:0':train_config.is_training})
