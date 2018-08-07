import numpy as np
from ExperienceBuffer import *

class Agent:
    # Set replayBuffer to None if you don't want to store experience,
    # Else, set it with an ExperienceBuffer object
    def __init__(self, env, net, replayBuffer=None):
        self.env = env
        self.net = net
        self.replayBuffer = replayBuffer
        self.reset()

    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.

    # Epsilon is the probability of choosing random action
    def step(self, session, epsilon, render=False):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            QValues = self.net.activate(session, [self.state])
            action = np.argmax(QValues)

        next_state, reward, done, infos = self.env.step(action)
        if render:
            self.env.render()
        self.total_reward += reward

        if self.replayBuffer != None:
            experience = Experience(self.state, action, reward, done, next_state)
            self.replayBuffer.append(experience)

        self.state = next_state

        return self.total_reward if done else None
