# Flags common across learners and actors.

from absl import flags

flags.DEFINE_string('logdir', '/agents', 'TensorFlow log directory.')
flags.DEFINE_alias('job-dir', 'logdir')
flags.DEFINE_string('server_address', 'localhost:8686', 'Server address.',
                    allow_hide_cpp=True)


flags.DEFINE_enum(
    'run_mode', 'learner', ['learner', 'actor'],
    'Whether we run the learner or the actor. Each actor runs the environment '
    'and sends to the learner each env observation and receives the action to '
    'play. A learner performs policy inference for batches of observations '
    'coming from multiple actors, and use the generated trajectories to learn.')

flags.DEFINE_integer('num_eval_envs', 0,
                     'Number of environments that will be used for eval '
                     ' (for agents that support eval environments).')
flags.DEFINE_integer('env_batch_size', 1,
                     'How many environments to operate on together in a batch.')
flags.DEFINE_integer('num_envs', 4,
                     'Total number of environments in all actors.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')
