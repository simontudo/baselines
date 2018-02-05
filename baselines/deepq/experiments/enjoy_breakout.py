import gym
import numpy as np
from baselines import deepq


def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = deepq.wrap_atari_dqn(env)
    act = deepq.load("breakout_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(np.array(obs)[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
