from absl import app
from absl import flags
from agents.learner import learner_loop, inference
from gym.envs.classic_control import Embedding
from agents import TrajRL_Trainer_5
from common.actor import actor_loop
from common import common_flags
import tensorflow as tf

FLAGS = flags.FLAGS

# Optimizer settings.
#flags.DEFINE_float('learning_rate', 0.00001, 'Learning rate.')
#flags.DEFINE_float('adam_epsilon', 1e-3, 'Adam epsilon.')

#flags.DEFINE_integer('stack_size', 4, 'Number of frames to stack.')


def create_agent(env_output_specs, actions):
  return TrajRL_Trainer_5.STsim_Trainer(env_output_specs, actions) #networks.LSTMTrainer(env_output_specs, actions)

# def create_optimizer(): #unused_final_iteration):
#   learning_rate_fn = lambda iteration: FLAGS.learning_rate
#   optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate,
#                                        epsilon=FLAGS.adam_epsilon)
#   return optimizer, learning_rate_fn


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if FLAGS.run_mode == 'actor':
    actor_loop(Embedding.create_environment)
  elif FLAGS.run_mode == 'learner':
    learner_loop(Embedding.create_environment, create_agent)
  else:
    raise ValueError('Unsupported run mode {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
  app.run(main)