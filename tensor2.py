"""Distributed MNIST training and validation, with model replicas.
A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.
The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#define parameter
flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1, "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 20000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "sync_replicas", False,
    "Use the sync_replicas (synchronized replicas) mode, "
    "wherein the parameter updates from workers are aggregated "
    "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "", "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28


def main(unused_argv):
  # Parse environment variable TF_CONFIG to get job_name and task_index

  # If not explicitly specified in the constructor and the TF_CONFIG
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  # Construct the cluster and start the server
  #解析叢集參數
  ps_spec = FLAGS.ps_hosts.split(",") #可以有多個ps 用逗號隔開
  worker_spec = FLAGS.worker_hosts.split(",")#可以有多個worker 用逗號隔開

  # Get the number of workers.
  num_workers = len(worker_spec)

  #建立當前任務節點伺服器
  cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    #設定內部server(設定cluster、job、task index等)
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    
    #如果是ps，呼叫server.join()無休止等待
    if FLAGS.job_name == "ps":
      server.join()

  #編號0的worker設定為chief worker
  is_chief = (FLAGS.task_index == 0)

  #設定用哪一顆GPU或者不使用GPU
  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  
  # 構建神經網路
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU
  with tf.device(

      #此device_setter可以回傳被tf.device接受的device名稱
      tf.train.replica_device_setter(
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    # truncated_normal：生成的值follow具有指定平均值和標準偏差的正態分佈
    #                   如果生成的值大於平均值2個標準偏差的值則丟棄重新選擇。
    # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    # 創造一個代表輸入資料的佔位符號(placeholder)，可以將它視作盛裝輸入資料的「容器」，並設定好它的維度資訊。
    # 因為我們要輸入的資料是維度為 784 個 pixels 的圖片，所以這個佔位符號的形狀是 [None, 784]，
    # None 可以是一個任何大小的維度，取決於輸入資料圖片數多寡。
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b) # x*w+b
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    print(y) 
    #exit()
    # clip_by_value(A, min, max)：把y值壓縮在[min,max]之間> max ，則=max ; < min,則 = min
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    #如果使用同步訓練機制
    if FLAGS.sync_replicas:
      #如果平行副本數量沒有指定
      if FLAGS.replicas_to_aggregate is None:
        #平行數量等於worker數量
        replicas_to_aggregate = num_workers
      else:
        #若有指定就使用指定的數量
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      #設定SyncReplicasOptimizer
      opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")

    train_step = opt.minimize(cross_entropy, global_step=global_step)

    #同步訓練機制的初始化
    if FLAGS.sync_replicas:
      #local_step初始化 (chief_worker會改寫此行)
      local_init_op = opt.local_step_init_op
      if is_chief:
        #chief_worker使用的global_step，也需要初始化
        local_init_op = opt.chief_init_op

      #將未初始化的Variable初始化
      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()

    #初始化全域變數
    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()

    if FLAGS.sync_replicas:
      #設定同步訓練的Supervisor
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
      #設定異步訓練的Supervisor
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          recovery_wait_secs=1,
          global_step=global_step)

    #設定session
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,  #沒GPU時可以用CPU
        log_device_placement=False, #不印出device placement的log
        device_filters=["/job:ps",  
                        "/job:worker/task:%d" % FLAGS.task_index]) #過濾沒有綁定的ps或worker

    #如果是chief 初始化所有worker的session
    #不是則等待chief回傳session
    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    #同步更新模式的chief worker
    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    #開始訓練
    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}

      _, step = sess.run([train_step, global_step], feed_dict=train_feed)
      local_step += 1

      now = time.time()
      print("%f: Worker %d: training step %d done (global step: %d)" %
            (now, FLAGS.task_index, local_step, step))

      if step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    #使用測試資料及做測試
    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()

