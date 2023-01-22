import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 23})

fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14, 6), dpi=300)

# score VS oracle
num_env_origin = [200, 500, 1000, 5000]


# train results
train_seed_0 = np.array([0.9506, 0.9917, 0.9718, 0.9185])
train_seed_9875 = np.array([0.9506, 0.9917, 0.9718, 0.9185])
train_seed_12569 = np.array([0.9399, 0.9836, 0.9841, 0.9702])

# test results
test_seed_0 = np.array([0.8582, 0.9664, 0.9869, 0.9475])
test_seed_9875 =  np.array([0.8582, 0.9664, 0.9869, 0.9475])
test_seed_12569 = np.array([0.9355, 0.9075, 0.9537, 0.9273])

train_average = np.mean([train_seed_0, train_seed_9875, train_seed_12569], axis=0)
train_stddev = np.std([train_seed_0, train_seed_9875, train_seed_12569], axis=0)

num_env = list(range(4))
axs[0].plot(num_env, train_average, 'orange', label='Train')
axs[0].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_9875, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_12569, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_9875, test_seed_12569], axis=0)
test_stddev = np.std([test_seed_0, test_seed_9875, test_seed_12569], axis=0)

axs[0].plot(num_env, test_average, 'royalblue', label='Test')
axs[0].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_9875, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_12569, c='royalblue', alpha=0.8)

# axs[0].set_title('Maze')
axs[0].set_xlabel('# Environments')
axs[0].set_ylabel('Score VS. Oracle')
axs[0].set_ylim(0, 1.1)
# plt.xticks(num_env, num_env_origin)
# axs[0].set_xticks(num_env, num_env_origin)
# axs[0].legend(loc='lower right')
# axs[0].show()


# success rate

# train results
train_seed_0 = np.array([0.735, 0.8125, 0.875, 0.9375])
train_seed_9875 = np.array([0.735, 0.8125, 0.875, 0.9375])
train_seed_12569 = np.array([0.9399, 0.9836, 0.9841, 0.9702])

# test results
test_seed_0 = np.array([0.6875, 0.9375, 0.8438, 0.9688])
test_seed_9875 = np.array([0.6875, 0.9375, 0.8438, 0.9688])
test_seed_12569 = np.array([0.9355, 0.9075, 0.9537, 0.9273])

train_average = np.mean([train_seed_0, train_seed_9875, train_seed_12569], axis=0)
train_stddev = np.std([train_seed_0, train_seed_9875, train_seed_12569], axis=0)

num_env = list(range(4))
axs[1].plot(num_env, train_average, 'orange', label='Train')
axs[1].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_9875, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_12569, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_9875, test_seed_12569], axis=0)
test_stddev = np.std([test_seed_0, test_seed_9875, test_seed_12569], axis=0)

axs[1].plot(num_env, test_average, 'royalblue', label='Test')
axs[1].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_9875, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_12569, c='royalblue', alpha=0.8)

# axs[1].set_title('Maze')
axs[1].set_xlabel('# Environments')
axs[1].set_ylabel('Success Rate')
axs[1].set_ylim(0, 1.1)
# plt.xticks(num_env, num_env_origin)
plt.setp(axs, xticks=num_env, xticklabels=num_env_origin)
plt.legend(loc='lower right', prop={'size': 14})
plt.savefig('plot_entropy.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()