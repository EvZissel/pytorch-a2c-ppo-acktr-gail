import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 23})
# fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14, 6), dpi=300)

# number of training environments seed 0
num_env_origin = [200, 500, 1000, 5000]


# train results
train_seed_10000 = np.array([9.680, 0.000, 0.000, 0.000])
train_seed_20000 = np.array([9.740, 0.000, 0.000, 0.000])
train_seed_30000 = np.array([9.770, 0.000, 0.000, 0.000])
train_seed_40000 = np.array([9.740, 0.000, 0.000, 0.000])

# test results
test_seed_10000 = np.array([9.070, 0.000, 0.000, 0.000])
test_seed_20000 = np.array([8.730, 0.000, 0.000, 0.000])
test_seed_30000 = np.array([9.160, 0.000, 0.000, 0.000])
test_seed_40000 = np.array([8.560, 0.000, 0.000, 0.000])

train_average = np.mean([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000], axis=0)
train_stddev = np.std([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000], axis=0)

max_train = np.maximum.reduce([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000]) - train_average
min_train = train_average - np.minimum.reduce([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000])
interval_train = np.maximum.reduce([max_train,min_train])

num_env = list(range(4))
plt.plot(num_env, train_average, 'orange', label='Train')
plt.fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, train_seed_10000, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_20000, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_30000, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_40000, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000], axis=0)
test_stddev = np.std([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000], axis=0)

max_test = np.maximum.reduce([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000]) - test_average
min_test = test_average - np.minimum.reduce([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000])
interval_test = np.maximum.reduce([max_test,min_test])

plt.plot(num_env, test_average, 'royalblue', label='Test')
plt.fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, test_seed_10000, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_20000, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_30000, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_40000, c='royalblue', alpha=0.8)

# plt.title('Maze')
# axs[0].set_xlabel('# Environments')
# axs[0].set_ylabel('Score')
# axs[0].set_ylim(0, 11)
# axs[0].xticks(num_env, num_env_origin)
# axs[0].legend(loc='lower right')
# plt.savefig('plot_reward', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()

#############
# Random policy
#############

# train results
train_seed_10000_rand = np.array([9.720, 0.000, 0.000, 0.000])
train_seed_20000_rand = np.array([9.650, 0.000, 0.000, 0.000])
train_seed_30000_rand = np.array([9.750, 0.000, 0.000, 0.000])
train_seed_40000_rand = np.array([9.750, 0.000, 0.000, 0.000])

# test results
test_seed_10000_rand = np.array([8.620, 0.000, 0.000, 0.000])
test_seed_20000_rand = np.array([8.250, 0.000, 0.000, 0.000])
test_seed_30000_rand = np.array([8.200, 0.000, 0.000, 0.000])
test_seed_40000_rand = np.array([8.700, 0.000, 0.000, 0.000])

train_average = np.mean([train_seed_10000_rand, train_seed_20000_rand, train_seed_30000_rand, train_seed_40000_rand], axis=0)
train_stddev = np.std([train_seed_10000_rand, train_seed_20000_rand, train_seed_30000_rand, train_seed_40000_rand], axis=0)

max_train = np.maximum.reduce([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000]) - train_average
min_train = train_average - np.minimum.reduce([train_seed_10000, train_seed_20000, train_seed_30000, train_seed_40000])
interval_train_RNN = np.maximum.reduce([max_train,min_train])

num_env = list(range(4))
plt.plot(num_env, train_average, 'orange', linestyle = 'dashed', label='Train with RNN')
plt.fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, train_seed_10000_rand, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_20000_rand, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_30000_rand, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_40000_rand, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_10000_rand, test_seed_20000_rand, test_seed_30000_rand, test_seed_40000_rand], axis=0)
test_stddev = np.std([test_seed_10000_rand, test_seed_20000_rand, test_seed_30000_rand, test_seed_40000_rand], axis=0)

max_test = np.maximum.reduce([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000]) - test_average
min_test = test_average - np.minimum.reduce([test_seed_10000, test_seed_20000, test_seed_30000, test_seed_40000])
interval_test_RNN = np.maximum.reduce([max_test,min_test])

plt.plot(num_env, test_average, 'royalblue',  linestyle = 'dashed', label='Test with RNN')
plt.fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, test_seed_10000_rand, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_20000_rand, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_30000_rand, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_40000_rand, c='royalblue', alpha=0.8)

# axs[1].set_title('Maze')
plt.xlabel('# Environments')
plt.ylabel('Score')
plt.ylim(0, 11)
plt.xticks(num_env, num_env_origin)
# plt.setp(axs, xticks=num_env, xticklabels=num_env_origin)
plt.legend(loc='lower right', prop={'size': 14})
plt.savefig('plot_reward.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()