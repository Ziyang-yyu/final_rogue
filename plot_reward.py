import matplotlib.pyplot as plt
import numpy as np
ep_dqn = np.loadtxt('rewards/dh_dqn_1600000_2.txt')


'''
ep_reward1 = np.loadtxt('rewards/agent_reward_duel_eps_800000.txt')
ep_reward2 = np.loadtxt('rewards/agent_reward_history_duel_eps_800000.txt')
'''
ave_episode = 1000
n = len(ep_dqn) // ave_episode
avg_reward = np.mean(np.reshape(ep_dqn[: ave_episode * n], [n, ave_episode]), 1)


'''
avg_reward2 = np.mean(np.reshape(ep_reward1[: ave_episode * n], [n, ave_episode]), 1)
avg_reward3 = np.mean(np.reshape(ep_reward2[: ave_episode * n], [n, ave_episode]), 1)
'''

plt.title("Agent Reward")

plt.plot(avg_reward,label='dh')


'''
plt.plot(avg_reward2,"y-",label='duel')
plt.plot(avg_reward3,"c-",label='all')
'''
plt.show()
