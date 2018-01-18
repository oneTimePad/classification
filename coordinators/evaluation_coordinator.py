import tensorflow as tf
from classification.protos import eval_pb2
import tensorflow.contrib.slim as slim

class EvaluationCoordinator(object):

    def __init__(self,
                 eval_ops_dict,
                 is_batch_evaluation):
        """Coordinates an Evaluation Sesssion (not used for Eval while training)

           Args:
            eval_ops_dict: a dict mapping format strings to evaluation operations
            is_batch_evaluation: whether we are running in batch mode
        """
        self._coord = tf.train.Coordinator()
        self._threads = []
        self._is_batch_evaluation = is_batch_evaluation
        self._eval_ops = []
        self._eval_fmt_str = ""

        for fmt,op in eval_ops_dict.items():
            self._eval_fmt_str+=fmt
            self._eval_ops.append(op)

    def evaluate(self,
            eval_config):
            """Runs Evaluation ops on examples

               Args:
                eval_config: protobuf config for evaluation
            """
            if not isinstance(eval_config, eval_pb2.EvalConfig):
                raise ValueError('train_config not type'
                                 'train_pb2.TrainConfig.')

            init_local = tf.local_variables_initializer()

            ckpt = tf.train.get_checkpoint_state(eval_config.\
                                                    checkpoint_to_restore)
            if not ckpt or not ckpt.model_checkpoint_path:
                raise ValueError("checkpoint to restore does not exist \
                                    or is not a valid checkpoint")

            print("TENSORFLOW INFO: Restoring from %s" % ckpt.model_checkpoint_path)
            with tf.Session() as sess:
                init_local.run()
                exclude_list = ["eval/total_loss"]
                vars_to_restore = slim.get_variables_to_restore(exclude = exclude_list)
                saver = tf.train.Saver(var_list = vars_to_restore) #TODO maybe allow var_list option?
                saver.restore(sess,ckpt.model_checkpoint_path)

                #create and start Queue runners
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    self._threads.extend(
                                        qr.create_threads(sess = sess,
                                                          coord = self._coord,
                                                          daemon = True,
                                                          start = True))
                try:

                    while not self._coord.should_stop():
                        results = sess.run(self._eval_ops)
                        print(self._eval_fmt_str % tuple(results))

                        if not self._is_batch_evaluation:
                            break

                except Exception as e:
                    print('TENSORFLOW INFO: Evaluation failed with ',e)

                finally:
                    self._coord.request_stop()
                    self._coord.join(self._threads,
                                     stop_grace_period_secs = 10)
