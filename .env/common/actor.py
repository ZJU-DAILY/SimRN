# by dlhu 05/24
import os
import timeit
from absl import flags
from absl import logging
import numpy as np
import grpc
import common.common_flags as common_flags
import common.env_wrappers as env_wrappers
import common.profiling as profiling
import common.utils as utils
import tensorflow as tf
from agents.learner import inference

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_actors', 4, 'Number of actors')
flags.DEFINE_integer('act', 0, 'actor id.')

def compute_query(data_set):
    length = len(data_set)
    ground_truth = np.load('./', allow_pickle=True)
    dist_matrix = ground_truth[:length][:length]
    dist_matrix_tf = tf.constant(dist_matrix)
    top_k_distances, top_k_indices = tf.math.top_k(-dist_matrix_tf, k=50)
    return top_k_indices

def actor_loop(create_env_func, config=None, log_period=1):
    '''
    create_env_func: the function that returns a newly created environment.
    config: configuration of training (flags).
    log_period: How often to log in seconds.
    '''
    if not config: config=FLAGS
    env_batch_size = FLAGS.env_batch_size
    logging.info('Starting actor loop. Environment batch size: %r',env_batch_size)

    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.logdir, 'actor_{}'.format(FLAGS.act)),
        flush_millis=20000, max_queue=1000) 
    timer_cls = profiling.ExportingTimer 

    # communicate with learner
    actor_step = 0
    with summary_writer.as_default():
        while True:
            try:
                # Client to communicate with the learner.
                #client = grpc.Client() #FLAGS.server_address
                #utils.update_config(config, client)

            #     batched_env = env_wrappers.BatchedEnvironment(
            #         create_env_func, env_batch_size, 0, config) 

            #     env_id = batched_env.env_ids
            #     run_id = np.random.randint(
            #         low=0,
            #         high=np.iinfo(np.int64).max,
            #         size=env_batch_size,
            #         dtype=np.int64)
            #     observation = batched_env.reset() 
            #     query_result = compute_query(observation)
            #     reward = np.zeros(env_batch_size, np.float32)
            #     raw_reward = np.zeros(env_batch_size, np.float32)

            #     elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
            #     while True:
            #         tf.summary.experimental.set_step(actor_step)
            #         env_output = utils.EnvOutput(reward, observation, query_result)
            #         with elapsed_inference_s_timer:
            #             net = inference(env_output, raw_reward) #client.inference(env_output, raw_reward)
            #         with timer_cls('actor/elapsed_env_step_s', 1000):
            #             observation, reward, info = batched_env.step(net) 
            #             query_result = compute_query(observation)
            #         for i in range(env_batch_size):
            #             raw_reward[i] = float((info[i] or {}).get('score_reward',
            #                                           reward[i]))

            #             actor_step += 1
            # except (tf.errors.UnavailableError, tf.errors.CancelledError):
            #     logging.info('Inference call failed. This is normal at the end of '
            #                  'training.')
            #     batched_env.close()

                env = create_env_func(0, FLAGS)
                observation = env.reset()
                query_result = compute_query(observation)
                reward = 0
                raw_reward = 0

                elapsed_inference_s_timer = timer_cls('actor/elapsed_inference_s', 1000)
                while True:
                    tf.summary.experimental.set_step(actor_step)
                    env_output = utils.EnvOutput(reward, observation, query_result)
                    with elapsed_inference_s_timer:
                        net = inference(env_output, raw_reward) #client.inference(env_output, raw_reward)
                    with timer_cls('actor/elapsed_env_step_s', 1000):
                        observation, reward, info = env.step(net) 
                        query_result = compute_query(observation)
                    
                    raw_reward = float((info or {}).get('score_reward', reward))

                    #actor_step += 1

            except (tf.errors.UnavailableError, tf.errors.CancelledError):
                logging.info('Inference call failed. This is normal at the end of '
                             'training.')
