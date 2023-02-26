import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 23})
# fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14, 6), dpi=300)

# number of training environments seed 0
num_env_origin = [200, 500, 1000, 5000]


# train results
train_seed_9658  = np.array([9.67, 9.74, 9.813, 9.893])
train_seed_3256  = np.array([9.857, 9.892, 9.764, 9.787])
train_seed_0     = np.array([9.686, 9.883, 9.842, 9.424])
train_seed_28965 = np.array([9.802, 9.691, 9.922, 9.645])

# test results
test_seed_9658  = np.array([5.847, 7.936, 8.673, 9.534])
test_seed_3256  = np.array([5.789, 7.64, 8.468, 9.538])
test_seed_0     = np.array([6.334, 8.015, 8.887, 9.191])
test_seed_28965 = np.array([6.003, 7.855, 8.828, 9.332])

train_average = np.mean([train_seed_9658, train_seed_3256, train_seed_0, train_seed_28965], axis=0)
train_stddev = np.std([train_seed_9658, train_seed_3256, train_seed_0, train_seed_28965], axis=0)

max_train = np.maximum.reduce([train_seed_9658,train_seed_3256,train_seed_0,train_seed_28965]) - train_average
min_train = train_average - np.minimum.reduce([train_seed_9658,train_seed_3256,train_seed_0,train_seed_28965])
interval_train = np.maximum.reduce([max_train,min_train])

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

max_test = np.maximum.reduce([test_seed_9658,test_seed_3256,test_seed_0,test_seed_28965]) - test_average
min_test = test_average - np.minimum.reduce([test_seed_9658,test_seed_3256,test_seed_0,test_seed_28965])
interval_test = np.maximum.reduce([max_test,min_test])

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
train_seed_9658_RNN = np.array([9.281, 9.85, 9.826, 9.683])
train_seed_89657_RNN = np.array([8.918, 9.53, 9.834, 9.84])
train_seed_68579_RNN = np.array([8.888, 9.709, 9.775, 9.869])
train_seed_3568_RNN = np.array([8.438, 9.726, 9.69, 9.837])

# test results
test_seed_9658_RNN = np.array([5.253, 7.882, 8.727, 9.332])
test_seed_89657_RNN = np.array([4.173, 7.208, 8.665, 9.668])
test_seed_68579_RNN = np.array([4.877, 7.872, 9.05, 9.565])
test_seed_3568_RNN = np.array([5.221, 7.67, 8.332, 9.782])

train_average = np.mean([train_seed_9658_RNN, train_seed_89657_RNN, train_seed_68579_RNN, train_seed_3568_RNN], axis=0)
train_stddev = np.std([train_seed_9658_RNN, train_seed_89657_RNN, train_seed_68579_RNN, train_seed_3568_RNN], axis=0)

max_train = np.maximum.reduce([train_seed_9658,train_seed_3256,train_seed_0,train_seed_28965]) - train_average
min_train = train_average - np.minimum.reduce([train_seed_9658,train_seed_3256,train_seed_0,train_seed_28965])
interval_train_RNN = np.maximum.reduce([max_train,min_train])

num_env = list(range(4))
plt.plot(num_env, train_average, 'orange', linestyle = 'dashed', label='Train with RNN')
plt.fill_between(num_env, train_average + train_stddev,
                 train_average - train_stddev, color='orange',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, train_seed_9658_RNN, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_89657_RNN, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_68579_RNN, c='orange', alpha=0.6)
plt.scatter(num_env, train_seed_3568_RNN, c='orange', alpha=0.6)

# test
test_average = np.mean([test_seed_9658_RNN, test_seed_89657_RNN, test_seed_68579_RNN, test_seed_3568_RNN], axis=0)
test_stddev = np.std([test_seed_9658_RNN, test_seed_89657_RNN, test_seed_68579_RNN, test_seed_3568_RNN], axis=0)

max_test = np.maximum.reduce([test_seed_9658,test_seed_3256,test_seed_0,test_seed_28965]) - test_average
min_test = test_average - np.minimum.reduce([test_seed_9658,test_seed_3256,test_seed_0,test_seed_28965])
interval_test_RNN = np.maximum.reduce([max_test,min_test])

plt.plot(num_env, test_average, 'royalblue',  linestyle = 'dashed', label='Test with RNN')
plt.fill_between(num_env, test_average + test_stddev,
                 test_average - test_stddev, color='royalblue',
                 alpha=0.25, linewidth=0)
plt.scatter(num_env, test_seed_9658_RNN, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_89657_RNN, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_68579_RNN, c='royalblue', alpha=0.8)
plt.scatter(num_env, test_seed_3568_RNN, c='royalblue', alpha=0.8)

# axs[1].set_title('Maze')
plt.xlabel('# Environments')
plt.ylabel('Score')
plt.ylim(0, 11)
plt.xticks(num_env, num_env_origin)
# plt.setp(axs, xticks=num_env, xticklabels=num_env_origin)
plt.legend(loc='lower right', prop={'size': 14})
plt.savefig('plot_reward.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()