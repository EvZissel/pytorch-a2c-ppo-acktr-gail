import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 23})

fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14, 6), dpi=300)

# score VS oracle
num_env_origin = [200, 500, 1000, 5000]

# score VS oracle without RNN
# train results
train_seed_0 = np.array([0.56, 0.5611, 0.5654, 0.5542])
train_seed_12345 = np.array([0.5716, 0.5645, 0.5646, 0.5209])
train_seed_54789 = np.array([0.5778, 0.5689, 0.5773, 0.5679])
train_seed_9784 = np.array([0.5752, 0.5854, 0.5761, 0.5648])

# test results
test_seed_0 = np.array([0.4906, 0.4996, 0.5194, 0.5355])
test_seed_12345 =  np.array([0.4657, 0.4986, 0.5199, 0.5319])
test_seed_54789 = np.array([0.4444, 0.4706, 0.5773, 0.5099])
test_seed_9784 = np.array([0.4766, 0.4877, 0.5193, 0.5163])

train_average = np.mean([train_seed_0, train_seed_12345, train_seed_54789, train_seed_9784], axis=0)
train_stddev = np.std([train_seed_0, train_seed_12345, train_seed_54789, train_seed_9784], axis=0)

num_env = list(range(4))
axs[0].plot(num_env, train_average, 'orange', label='Train')
axs[0].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_12345, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_54789, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_9784, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_12345, test_seed_54789, test_seed_9784], axis=0)
test_stddev = np.std([test_seed_0, test_seed_12345, test_seed_54789, test_seed_9784], axis=0)

axs[0].plot(num_env, test_average, 'royalblue', label='Test')
axs[0].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_12345, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_54789, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_9784, c='royalblue', alpha=0.8)


# score VS oracle with RNN
# train results
train_seed_0 = np.array([0.9642, 0.9464, 0.9603, 0.9774])
train_seed_9875 = np.array([0.979, 0.9504, 0.9694, 0.9641])
train_seed_12569 = np.array([0.9291, 0.9314, 0.9563, 0.9667])
train_seed_58967 = np.array([0.9401, 0.9731, 0.961, 0.9815])

# test results
test_seed_0 = np.array([0.914, 0.9281, 0.9413, 0.9728])
test_seed_9875 =  np.array([0.8991, 0.9371, 0.9452, 0.9596])
test_seed_12569 = np.array([0.9062, 0.9073, 0.9404, 0.966])
test_seed_58967 = np.array([0.8774, 0.9428, 0.9505, 0.9693])

train_average = np.mean([train_seed_0, train_seed_9875, train_seed_12569, train_seed_58967], axis=0)
train_stddev = np.std([train_seed_0, train_seed_9875, train_seed_12569, train_seed_58967], axis=0)

num_env = list(range(4))
axs[0].plot(num_env, train_average, 'orange', linestyle = 'dashed', label='Train with RNN')
axs[0].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_9875, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_12569, c='orange', alpha=0.6)
axs[0].scatter(num_env, train_seed_58967, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_9875, test_seed_12569, test_seed_58967], axis=0)
test_stddev = np.std([test_seed_0, test_seed_9875, test_seed_12569, test_seed_58967], axis=0)

axs[0].plot(num_env, test_average, 'royalblue', linestyle = 'dashed', label='Test with RNN')
axs[0].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[0].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_9875, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_12569, c='royalblue', alpha=0.8)
axs[0].scatter(num_env, test_seed_58967, c='royalblue', alpha=0.8)

# axs[0].set_title('Maze')
axs[0].set_xlabel('# Environments')
axs[0].set_ylabel('Score VS. Oracle')
axs[0].set_ylim(0, 1.1)
# plt.xticks(num_env, num_env_origin)
# axs[0].set_xticks(num_env, num_env_origin)
# axs[0].legend(loc='lower right')
# axs[0].show()


# success rate without RNN

# train results
train_seed_0 = np.array([0.2044, 0.2088, 0.2086, 0.1946])
train_seed_12345 = np.array([0.2097, 0.2178, 0.2028, 0.2014])
train_seed_54789 = np.array([0.2097, 0.2115, 0.195, 0.1894])
train_seed_9784 = np.array([0.2204, 0.238, 0.1889, 0.2091])

# test results
test_seed_0 = np.array([0.1814, 0.1525, 0.1887, 0.1758])
test_seed_12345 =  np.array([0.1646, 0.1841, 0.1683, 0.1722])
test_seed_54789 = np.array([0.1192, 0.1133, 0.1237, 0.1104])
test_seed_9784 = np.array([0.147, 0.1707, 0.1588, 0.1526])

train_average = np.mean([train_seed_0, train_seed_12345, train_seed_54789, train_seed_9784], axis=0)
train_stddev = np.std([train_seed_0, train_seed_12345, train_seed_54789, train_seed_9784], axis=0)

num_env = list(range(4))
axs[1].plot(num_env, train_average, 'orange', label='Train')
axs[1].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_12345, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_54789, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_9784, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_12345, test_seed_54789, test_seed_9784], axis=0)
test_stddev = np.std([test_seed_0, test_seed_12345, test_seed_54789, test_seed_9784], axis=0)

axs[1].plot(num_env, test_average, 'royalblue', label='Test')
axs[1].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_12345, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_54789, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_9784, c='royalblue', alpha=0.8)


# success rate with RNN

# train results
train_seed_0 = np.array([0.8487, 0.8352, 0.8745, 0.9216])
train_seed_9875 = np.array([0.8345, 0.7813, 0.9187, 0.8989])
train_seed_12569 = np.array([0.832, 0.7964, 0.8782, 0.9398])
train_seed_58967 = np.array([0.8192, 0.9038, 0.8779, 0.9435])

# test results
test_seed_0 = np.array([0.66, 0.7764, 0.8026, 0.9194])
test_seed_9875 = np.array([0.6762, 0.8077, 0.8429, 0.8666])
test_seed_12569 = np.array([0.7255, 0.7669, 0.8068, 0.8903])
test_seed_58967 = np.array([0.6213, 0.7875, 0.8167, 0.8977])

train_average = np.mean([train_seed_0, train_seed_9875, train_seed_12569, train_seed_58967], axis=0)
train_stddev = np.std([train_seed_0, train_seed_9875, train_seed_12569, train_seed_58967], axis=0)

num_env = list(range(4))
axs[1].plot(num_env, train_average, 'orange', linestyle = 'dashed', label='Train with RNN')
axs[1].fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, train_seed_0, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_9875, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_12569, c='orange', alpha=0.6)
axs[1].scatter(num_env, train_seed_58967, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_0, test_seed_9875, test_seed_12569, test_seed_58967], axis=0)
test_stddev = np.std([test_seed_0, test_seed_9875, test_seed_12569, test_seed_58967], axis=0)

axs[1].plot(num_env, test_average, 'royalblue', linestyle = 'dashed', label='Test with RNN')
axs[1].fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
axs[1].scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_9875, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_12569, c='royalblue', alpha=0.8)
axs[1].scatter(num_env, test_seed_58967, c='royalblue', alpha=0.8)

# axs[1].set_title('Maze')
axs[1].set_xlabel('# Environments')
axs[1].set_ylabel('Success Rate')
axs[1].set_ylim(0, 1.1)
# plt.xticks(num_env, num_env_origin)
plt.setp(axs, xticks=num_env, xticklabels=num_env_origin)
plt.legend(loc='lower right', prop={'size': 14})
plt.savefig('plot_entropy.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()