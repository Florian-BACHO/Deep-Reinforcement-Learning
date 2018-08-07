#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import gym

ENV_NAME = "CartPole-v0"
METHOD_NAME = "Cross Entropy"

OBSERVATION_SPACE = 4 # Entry layer
HIDDEN_LAYERS = [30] # One hidden layer
ACTION_SPACE = 2 # Output layer
LEARNING_RATE = 1e-2
BATCH_SIZE = 16
PERCENTILE = 70
SOLVED_LIMIT = 199

class NeuralNet:
    def __init__(self, nbEntry, nbHiddens, nbOut, learningRate=1e-2):
        # NN tensors
        self.entry = tf.placeholder(tf.float32, (None, nbEntry))
        self.hiddens = []
        for i, it in enumerate(nbHiddens):
            tmp_layer = tf.layers.dense((self.entry if i == 0 else self.hiddens[-1]), \
                                 it, activation=tf.nn.relu)
            self.hiddens.append(tmp_layer)
        self.out = tf.layers.dense(self.entry if len(self.hiddens) == 0 else self.hiddens[-1], \
                            nbOut)

        self.multinomialChoice = tf.multinomial(tf.log(tf.nn.softmax(self.out)), num_samples=1)
        self.argmaxChoice = tf.argmax(self.out)

        # Train tensors
        self.expectedOutputs = tf.placeholder(tf.int32, (None))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, \
                                                                   labels=self.expectedOutputs)
        self.optimizer = tf.train.AdamOptimizer(learningRate)
        self.trainOperation = self.optimizer.minimize(self.loss)

    # Activate Net
    def forward(self, session, x, random=False):
        if random:
            out = session.run(self.multinomialChoice, feed_dict={self.entry: x})[0]
        else:
            out = session.run(self.argmaxChoice, feed_dict={self.entry: x})
        return out

    def train(self, session, x, y):
        session.run(self.trainOperation, feed_dict={self.entry: x, \
                                                    self.expectedOutputs: y})

def executeEpisode(session, env, nn, render=True):
    done = False
    obs = env.reset()
    episode_reward = 0.
    episode_steps = []
    while not done:
        if render:
            env.render()
        action = nn.forward(session, [obs], True)[0]
        # Make step tuple
        episode_steps.append((obs, action))

        # Execute action
        obs, reward, done, info = env.step(action)

        # Cumulate rewards during this episode
        episode_reward += reward

    # Return tuple of cumulated reward and episode steps
    return (episode_reward, episode_steps)

# Execute <size> episods and return episodes
def makeTrainBatch(session, env, nn, size):
    batch = []
    for i in range(size):
        batch.append(executeEpisode(session, env, nn))
    return batch

# This is the core of the cross entropy method in RL
# This function return the filtered batch by applying percentile on episods
def filterBatch(batch, percentile):
    all_rewards = [it[0] for it in batch]
    reward_bound = np.percentile(all_rewards, percentile)
    reward_mean = float(np.mean(all_rewards))

    # Make train examples
    x = []
    y = []
    for episode in batch:
        # continue if the reward of the current episode is less than the bound reward given by
        # pourcentile
        if episode[0]< reward_bound:
            continue
        x.extend([step[0] for step in episode[1]])
        y.extend([step[1] for step in episode[1]])

    return x, y, reward_bound, reward_mean

if __name__ == "__main__":
    print("Environment:", ENV_NAME)
    print("Method:", METHOD_NAME)

    env = gym.make(ENV_NAME)
    nn = NeuralNet(OBSERVATION_SPACE, HIDDEN_LAYERS, ACTION_SPACE, LEARNING_RATE)
    session = tf.Session()

    # Initialize tensorflow global variables
    session.run(tf.global_variables_initializer())
    reward_mean = 0.
    i = 0.

    while reward_mean < SOLVED_LIMIT:

        begin = time.time()
        batch = makeTrainBatch(session, env, nn, BATCH_SIZE)
        end = time.time()
        x, y, reward_bound, reward_mean = filterBatch(batch, PERCENTILE)
        nn.train(session, x, y)
        print("%d: reward_mean=%.1f, reward_bound=%.1f" % (i, reward_mean, reward_bound))
        i += 1
