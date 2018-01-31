import tensorflow as tf
import time
from datetime import datetime
class LoggerHook(tf.train.SessionRunHook):
       def __init__(self,
                    eval_ops_dict,
                    log_frequency,
                    global_step):
            """A hook that gets called at each training step

               Args:
                eval_ops_dict: a dict mapping format strings to evaluation operations
                log_frequency: how frequently to log evaluations to stdout
                global_step: global_step variable
            """
            self._eval_fmt_str = "step %d: "
            self._eval_ops = [global_step]
            if eval_ops_dict:
                for k,v in eval_ops_dict.items():
                    self._eval_fmt_str += k
                    self._eval_ops.append(v)
            self._log_frequency = log_frequency
       def begin(self):
           """Called at the start of the training session"""
           self._step = -1
           self._start_time = time.time()
       def before_run(self,run_context):
           """Called before each training step"""
           self._step+=1
           #change this if you want to reduce the number of times eval ops are ran
           return tf.train.SessionRunArgs(self._eval_ops)

       def after_run(self,run_context,run_values):
           """Called after each training step"""
           if self._step % self._log_frequency ==0 and run_values.results:
               """Log evaluation operations"""
               current_time = time.time()
               duration = current_time - self._start_time
               self._start_time = current_time
               print('TENSORFLOW INFO: %s: ' %(datetime.now()),end='')
               print(self._eval_fmt_str % tuple(run_values.results))
