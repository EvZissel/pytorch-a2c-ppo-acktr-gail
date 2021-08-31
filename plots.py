import matplotlib.pyplot as plt

# number of training environments seed 0
num_env_fully_obs = [100, 500, 1000, 5000, 10000, 50000]
num_env_masked_obs = [100, 500, 1000, 2000, 5000, 8000, 10000, 50000]
entropy_800 = [800, 800, 800, 800]

# train results
train_fully_obs = [9.791, 9.992, 9.922, 9.998, 10, 9.96]
train_masked_obs = [8.3, 4, 3.1, 3, 2.8, 3.7, 3.4, 3.4]
entropy_800_train = [9.94, 800, 800, 800]

# test results
test_fully_obs = [5, 7, 8.5, 9.6, 9.8, 9.95]
test_masked_obs = [3, 3.6, 3, 2.96, 2.7, 3.6, 3.2, 3.5]
entropy_800_test = [7.5, 800, 800, 800]


plt.plot(num_env_fully_obs[:-1], train_fully_obs[:-1], 'bo--', label='Fully obs train')
plt.plot(num_env_fully_obs[:-1], test_fully_obs[:-1], 'go--', label='Fully obs test')
plt.plot(num_env_masked_obs[:-1], train_masked_obs[:-1], 'ys--', label='Masked obs train')
plt.plot(num_env_masked_obs[:-1], test_masked_obs[:-1], 'ms--', label='Masked obs test')
plt.legend()
plt.show()

# Entropy at 800
entropy =       [0.001, 0.01, 0.05, 0.1, 0.2, 0.3,  0.4, 0.6, 0.8, 1, 10]
entropy_train = [9.88,  9.9,  9.8,  9.9, 9.9, 9.92, 9.8, 9.8, 9.7, 9.7, 5]
entropy_test =  [7.6,   7.5,  8,    8.2, 8.3, 8.4,  8.4, 7.8, 8,   8, 3.5]

entropy2 =       [0.01, 0.05, 0.1, 0.2,  0.3,  0.4, 0.6, 0.8, 1]
entropy_train2 = [9.9,  9.9,  10,  9.98, 9.9,  9.8, 9.8, 9.8, 8]
entropy_test2 =  [8,    8.4,  8.2, 8.3,  8.2,  8.2, 8,   7.8, 6]

entropy3 =       [0.01,  0.05,  0.1,   0.2,  0.3,  0.4, 0.6, 0.8, 1]
entropy_train3 = [9.9,   9.994, 9.94,  9.97, 9.84, 9.8, 9.8, 9.8, 9]
entropy_test3 =  [8,     8.4  , 8.1,   8.2,  8.2,  8.2, 8.1, 8.0, 7.5]

plt.plot(entropy[:-1], entropy_train[:-1], 'bo-', label='train seed 0')
plt.plot(entropy[:-1], entropy_test[:-1], 'bo--', label='test seed 0')
plt.plot(entropy2, entropy_train2, 'yo-', label='train seed 2578')
plt.plot(entropy2, entropy_test2, 'yo--', label='test seed 2578')
plt.plot(entropy3, entropy_train3, 'ro-', label='train seed 35431')
plt.plot(entropy3, entropy_test3, 'ro--', label='test seed 35431')
plt.xlabel('Entropy')
plt.legend()
plt.show()