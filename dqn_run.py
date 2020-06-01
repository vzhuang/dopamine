import os
from dopamine.colab import utils as colab_utils
import gin
import dopamine.discrete_domains.run_experiment as run_experiment

BASE_PATH = 'dopamine/agents/dqn/configs'


# Modified from dopamine/agents/dqn/config/dqn_cartpole.gin
dqn_config = """
# Hyperparameters for a simple DQN-style Cartpole agent. The hyperparameters
# chosen achieve reasonable performance.
import dopamine.discrete_domains.gym_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables

DQNAgent.observation_shape = (8, 1)
DQNAgent.observation_dtype = %gym_lib.CARTPOLE_OBSERVATION_DTYPE
DQNAgent.stack_size = %gym_lib.CARTPOLE_STACK_SIZE
DQNAgent.network = @gym_lib.CartpoleDQNNetwork
DQNAgent.gamma = 0.99
DQNAgent.update_horizon = 1
DQNAgent.min_replay_history = 500
DQNAgent.update_period = 1
DQNAgent.target_update_period = 20
DQNAgent.epsilon_fn = @dqn_agent.linearly_decaying_epsilon #identity_epsilon
DQNAgent.epsilon_decay_period = 50000
DQNAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
DQNAgent.optimizer = @tf.train.AdamOptimizer()

tf.train.AdamOptimizer.learning_rate = 0.0001
tf.train.AdamOptimizer.epsilon = 0.0003125

create_gym_environment.environment_name = 'LunarLander'
create_gym_environment.version = 'v2'
create_agent.agent_name = 'dqn'
Runner.create_environment_fn = @gym_lib.create_gym_environment
#TrainRunner.create_environment_fn = @gym_lib.create_gym_environment
Runner.num_iterations = 100
Runner.training_steps = 1000
Runner.evaluation_steps = 1000
Runner.max_steps_per_episode = 1000  # Default max episode length.

WrappedReplayBuffer.replay_capacity = 100000
WrappedReplayBuffer.batch_size = 64
"""
gin.parse_config(dqn_config, skip_unknown=False)

for run in range(10):
    DQN_PATH = os.path.join(BASE_PATH, 'llv2_standard_eval_decay/run_' + str(run))

    dqn_runner = run_experiment.create_runner(DQN_PATH, schedule='continuous_train_and_eval')
    print('Will train DQN agent, please be patient, may be a while...')
    dqn_runner.run_experiment()
    print('Done training!')

    data = colab_utils.read_experiment(DQN_PATH, verbose=True,
                                       summary_keys=['train_episode_returns',
                                                     'train_episode_actual_returns',
                                                     'train_episode_lengths',
                                                     'eval_episode_returns',
                                                     'eval_episode_actual_returns',
                                                     'eval_episode_lengths'])
    data['agent'] = 'DQN'
    data['run'] = run

    print(data)