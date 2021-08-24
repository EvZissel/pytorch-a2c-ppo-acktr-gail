import matplotlib.pyplot as plt

# 25 environments with different entropy coefficient
entropy =      [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_0_train = [0.84,   0.96,  0.96, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_0_test =  [0.35,   0.40,  0.50, 0.16, 0.14, 0.12, 0.11, 0.1, 0.13, 0.11, 0.12, 0.12]

seed_1235_entropy =      [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_1235_train =        [0.88,   1.0,   0.96, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_1235_test =         [0.35,   0.60,  0.50, 0.13, 0.11, 0.1,  0.1,  0.12, 0.11, 0.11, 0.12, 0.13]

seed_354_entropy =      [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_354_train =        [0.68,   0.96,  0.84, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_354_test =         [0.50,   0.41,  0.35, 0.1,  0.1,  0.12, 0.13, 0.11, 0.15, 0.12, 0.15, 0.1]

plt.plot(entropy, seed_0_train, 'bo--', label='train seed 0')
plt.plot(entropy, seed_0_test, 'go--', label='test seed 0')
plt.plot(seed_1235_entropy, seed_1235_train, 'yo--', label='train seed 1235')
plt.plot(seed_1235_entropy, seed_1235_test, 'mo--', label='test seed 1235')
plt.plot(seed_354_entropy, seed_354_train, 'ko--', label='train seed 354')
plt.plot(seed_354_entropy, seed_354_test, 'ro--', label='test seed 354')
plt.xlabel('Entropy')
plt.legend()
plt.show()