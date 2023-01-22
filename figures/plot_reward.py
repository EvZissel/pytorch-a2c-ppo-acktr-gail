import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 23})
# fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14, 6), dpi=300)

# number of training environments seed 0
num_env_origin = [200, 500, 1000, 5000]


# train results
train_seed_9658 = np.array([10.0, 9.668, 10.0, 10.0])
train_seed_3256 = np.array([9.922, 10.0, 10.0, 10.0])
train_seed_0 = np.array([10.0, 10.0, 10.0, 9.922])
train_seed_28965 = np.array([9.792, 9.757, 9.965, 9.757])

# test results
test_seed_9658 = np.array([5.625, 5.859, 7.734, 9.379])
test_seed_3256 = np.array([4.766, 6.953, 8.047, 8.984])
test_seed_0 = np.array([4.844, 6.563, 7.5, 8.75])
test_seed_28965 = np.array([6.076, 7.771, 8.785, 9.375])

train_average = np.mean([train_seed_9658, train_seed_3256, train_seed_0, train_seed_28965], axis=0)
train_stddev = np.std([train_seed_9658, train_seed_3256, train_seed_0, train_seed_28965], axis=0)

num_env = list(range(4))
plt.plot(num_env, train_average, 'orange', label='Train')
plt.fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, train_seed_9658, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_3256, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_0, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_28965, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_9658, test_seed_3256, test_seed_0, test_seed_28965], axis=0)
test_stddev = np.std([test_seed_9658, test_seed_3256, test_seed_0, test_seed_28965], axis=0)

plt.plot(num_env, test_average, 'royalblue', label='Test')
plt.fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, test_seed_9658, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_3256, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_28965, c='royalblue', alpha=0.8)

# plt.title('Maze')
# axs[0].set_xlabel('# Environments')
# axs[0].set_ylabel('Score')
# axs[0].set_ylim(0, 11)
# axs[0].xticks(num_env, num_env_origin)
# axs[0].legend(loc='lower right')
# plt.savefig('plot_reward', format='pdf', dpi=300, bbox_inches='tight')
# plt.show()

#############
# train with RNN
#############

# train results
train_seed_9658_RNN = np.array([9.27, 9.668, 9.758, 9.6875])
train_seed_89657_RNN = np.array([8.906, 9.602, 9.844, 10.0])
train_seed_0 = np.array([10.0, 10.0, 10.0, 9.922]) #needed change
train_seed_28965 = np.array([9.792, 9.757, 9.965, 9.757]) #needed change

# test results
test_seed_9658_RNN = np.array([5.347, 7.815, 8.784, 9.340])
test_seed_89657_RNN = np.array([3.969, 7.273, 8.693, 9.716])
test_seed_0 = np.array([4.844, 6.563, 7.5, 8.75]) #needed change
test_seed_28965 = np.array([6.076, 7.771, 8.785, 9.375]) #needed change

train_average = np.mean([train_seed_9658_RNN, train_seed_89657_RNN, train_seed_0, train_seed_28965], axis=0)
train_stddev = np.std([train_seed_9658_RNN, train_seed_89657_RNN, train_seed_0, train_seed_28965], axis=0)

num_env = list(range(4))
plt.plot(num_env, train_average, 'orange', linestyle = 'dashed', label='Train with RNN')
plt.fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, train_seed_9658_RNN, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_89657_RNN, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_0, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_28965, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_9658_RNN, test_seed_89657_RNN, test_seed_0, test_seed_28965], axis=0)
test_stddev = np.std([test_seed_9658_RNN, test_seed_89657_RNN, test_seed_0, test_seed_28965], axis=0)

plt.plot(num_env, test_average, 'royalblue',  linestyle = 'dashed', label='Test with RNN')
plt.fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, test_seed_9658_RNN, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_89657_RNN, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_0, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_28965, c='royalblue', alpha=0.8)

# axs[1].set_title('Maze')
plt.xlabel('# Environments')
plt.ylabel('Score')
plt.ylim(0, 11)
plt.xticks(num_env, num_env_origin)
# plt.setp(axs, xticks=num_env, xticklabels=num_env_origin)
plt.legend(loc='lower right', prop={'size': 14})
plt.savefig('plot_reward.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()