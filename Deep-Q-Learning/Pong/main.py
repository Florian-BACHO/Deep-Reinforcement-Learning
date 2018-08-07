#!/usr/bin/python3

import tensorflow
import numpy as np
import gym
import gym.spaces
import atari_wrappers
import collections
import copy
import time
from DCQN import *
from ExperienceBuffer import *
from Agent import *

ENV_NAME = "PongNoFrameskip-v4"
METHOD_NAME = "Deep Q-Learning"

# Stop training if the model mean rewards on <REWARD_MEMORY_SIZE> last episodes is greater than <MEAN_REWARD_BOUND>
MEAN_REWARD_BOUND = 19.5
REWARD_MEMORY_SIZE = 100

LEARNING_RATE = 1e-2
DISCOUNT_RATE = 0.99 # Gamma in the Bellman equation
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
TRAINING_START_FRAME = 10000 # The count of frames we wait for before starting the training
SYNC_TARGET_FRAMES = 1000 # How frequently we sync model weights from the training to the target model

# Epsilon is the probability of choosing random action. It is linearly decayed from 1.0 to 0.02
# on <EPSILON_DECAY_LAST_FRAME> frames
EPSILON_DECAY_LAST_FRAME = 1e5
EPSILON_START = 1.
EPSILON_FINAL = 0.02

def makeAgent():
    env = atari_wrappers.make_env(ENV_NAME)

    net = DCQN(env.observation_space.shape, env.action_space.n)
    net.buildLearningTensors(LEARNING_RATE)

    replayBuffer = ExperienceBuffer(REPLAY_BUFFER_SIZE)

    return Agent(env, net, replayBuffer)

def trainAgent(session, agent):
    targetNet = DCQN(agent.env.observation_space.shape, agent.env.action_space.n)
    agent.net.copyWeights(session, targetNet)

    done = False
    frame_idx = 0
    epsilon = EPSILON_START
    total_rewards = []
    ts_frame = 0
    ts = time.time()
    played_games = 0

    while not done:
        frame_idx += 1

        # Calculate epsilon and execute step
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.step(session, epsilon, render=True)

        # If episode is done, print some stats and check if the problem is solved
        if reward != None:
            played_games += 1
            agent.reset()

            total_rewards.append(reward)
            if len(total_rewards) > REWARD_MEMORY_SIZE:
                total_rewards.pop(0)

            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            mean_reward = np.mean(total_rewards)
            if mean_reward > MEAN_REWARD_BOUND:
                done = True

            print("%d: done %d games, mean reward %.3f, epsilon %.2f, speed %.2f fps" % \
                  (frame_idx, played_games, mean_reward, epsilon, speed))

        # Don't train as long as we don't have enought experience in buffer
        if len(agent.replayBuffer) < TRAINING_START_FRAME:
            continue

        # Copy net to target periodicaly
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            agent.net.copyWeights(session, targetNet)

        batch = agent.replayBuffer.sample(BATCH_SIZE)
        agent.net.train(session, batch, targetNet, DISCOUNT_RATE)

if __name__ == "__main__":
    print("Environment:", ENV_NAME)
    print("Method:", METHOD_NAME)

    agent = makeAgent()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    trainAgent(session, agent)
