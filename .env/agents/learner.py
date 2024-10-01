"Deep reinforcement learning with RNN"
from absl import flags
from absl import logging
from gym.envs.classic_control import Embedding
# from agents import networks
import os
import grpc
from agents import TrajRL_Trainer_5
from common import common_flags  
from common import utils, profiling
import tensorflow as tf
import numpy as np

flags.DEFINE_integer('save_checkpoint_secs', 1800, 'Checkpoint save period in seconds.')
flags.DEFINE_integer('inference_batch_size', -1,
                     'Batch size for inference, -1 for auto-tune.')

FLAGS = flags.FLAGS

def compute_query_init(data_set):
    length = len(data_set)
    ground_truth = np.load('./', allow_pickle=True) # './' to denote the direction path of ground-truths
    dist_matrix = ground_truth[:length][:length] 
    dist_matrix_tf = tf.constant(dist_matrix)
    top_k_distances, top_k_indices = tf.math.top_k(-dist_matrix_tf, k=50)
    return top_k_indices

def get_num_training_envs():
  return FLAGS.num_envs - FLAGS.num_eval_envs

def is_training_env(env_id): 
  """Training environment IDs are in range [0, num_training_envs)."""
  return env_id < get_num_training_envs()

def get_envs_epsilon(env_ids, num_training_envs, num_eval_envs, eval_epsilon):
  """Per-environment epsilon as in Apex and R2D2.

  Args:
    env_ids: <int32>[inference_batch_size], the environment task IDs (in range
      [0, num_training_envs+num_eval_envs)).
    num_training_envs: Number of training environments. Training environments
      should have IDs in [0, num_training_envs).
    num_eval_envs: Number of evaluation environments. Eval environments should
      have IDs in [num_training_envs, num_training_envs + num_eval_envs).
    eval_epsilon: Epsilon used for eval environments.

  Returns:
    A 1D float32 tensor with one epsilon for each input environment ID.
  """
  # <float32>[num_training_envs + num_eval_envs]
  epsilons = tf.concat(
      [tf.math.pow(0.4, tf.linspace(1., 8., num=num_training_envs)),
       tf.constant([eval_epsilon] * num_eval_envs)],
      axis=0) 
  return tf.gather(epsilons, env_ids)

def apply_epsilon_greedy(actions, env_ids, num_training_envs,
                         num_eval_envs, eval_epsilon, num_actions):
  """Epsilon-greedy: randomly replace actions with given probability.

  Args:
    actions: <int32>[batch_size] tensor with one action per environment.
    env_ids: <int32>[inference_batch_size], the environment task IDs (in range
      [0, num_envs)).
    num_training_envs: Number of training environments.
    num_eval_envs: Number of eval environments.
    eval_epsilon: Epsilon used for eval environments.
    num_actions: Number of environment actions.

  Returns:
    A new <int32>[batch_size] tensor with one action per environment. With
    probability epsilon, the new action is random, and with probability (1 -
    epsilon), the action is unchanged, where epsilon is chosen for each
    environment.
  """
  batch_size = tf.shape(actions)[0]
  epsilons = get_envs_epsilon(env_ids, num_training_envs, num_eval_envs,
                              eval_epsilon)
  random_actions = tf.random.uniform([batch_size], maxval=num_actions,
                                     dtype=tf.int32)
  probs = tf.random.uniform(shape=[batch_size])
  return tf.where(tf.math.less(probs, epsilons), random_actions, actions)

def inference(env_output, raw_reward): 
  actions = {
    #'batch_size': 100, #spaces.Discrete(5), #32, 64, 128, 256, 512
    'layer_num': 4, #spaces.Discrete(5), # 0-4共5个选择
    'hidden_size': 64, #spaces.Discrete(5),
    'learning_rate': 0.001, #spaces.Box(low=0.001, high=0.5, shape=(1,), dtype=float),
    'training_epoch': 150,
    'batch_size': 128, #spaces.Discrete(5) #32, 64, 128, 256, 512
    'optim_method': 'Adam', #spaces.Discrete(2),
    'active_func': 'sigmoid', #spaces.Discrete(3),
  }
  net = TrajRL_Trainer_5.STsim_Trainer(actions) #networks.LSTMTrainer(env_output, actions)
  return net

def validate_config():
  utils.validate_learner_config(FLAGS)
  #assert FLAGS.n_steps >= 1, '--n_steps < 1 does not make sense.'
  assert FLAGS.num_envs > FLAGS.num_eval_envs, (
      'Total number of environments ({}) should be greater than number of '
      'environments reserved to eval ({})'.format(
          FLAGS.num_envs, FLAGS.num_eval_envs))

def learner_loop(create_env_fn, create_agent_fn): #agent=environment self-train
  logging.info('Starting learner loop')
  validate_config()
  settings = utils.init_learner(0) #(FLAGS.num_training_tpus)
  strategy, inference_devices, training_strategy, encode, decode = settings
  env = create_env_fn(0, FLAGS)
  env_output_specs = utils.EnvOutput(
      0.0,
      env.reset(),
      compute_query_init(env.reset()),
  )
  action_specs = {
    'layer_num': 4,
    'hidden_size': 64,
    'learning_rate': 0.001,
    'training_epoch': 150,
    'batch_size': 128,
    'optim_method': 'Adam', 
    'active_func': 'sigmoid'
  }

  #num_actions = len(env.action_space)
  #gent_input_specs = (action_specs, env_output_specs)

  with strategy.scope():
    #server = grpc.Server(FLAGS.server_address)
    #server.start()
    # Initialize agent and variables.
    agent = create_agent_fn(env_output_specs, action_specs) 

    agent_estimator = tf.keras.estimator.model_to_estimator(keras_model=agent)
    agent_estimator.train(input_fn=train_input_fn, steps=2000)

    #agent.LSTMtrain(env_output_specs, action_specs)

    # 日志
    # summary_writer = tf.summary.create_file_writer(
    #   os.path.join(FLAGS.logdir, 'learner'),
    #   flush_millis=20000, max_queue=1000) 
    # timer_cls = profiling.ExportingTimer 

    #server.shutdown()

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00001, 'Learning rate.')
flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')

flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')


def create_agent(env_output_specs, actions):
  return TrajRL_Trainer_5.STsim_Trainer(env_output_specs, actions) #networks.LSTMTrainer(env_output_specs, actions)

# def create_optimizer(): #unused_final_iteration):
#   learning_rate_fn = lambda iteration: FLAGS.learning_rate
#   optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
#                                        epsilon=FLAGS.adam_epsilon)
#   return optimizer, learning_rate_fn
