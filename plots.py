import matplotlib.pyplot as plt

# 25 environments with different entropy coefficient
entropy      = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_0_train = [0.84, 0.84, 0.96,  0.96, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_0_test  = [0.39, 0.35, 0.40,  0.50, 0.16, 0.14, 0.12, 0.11, 0.1, 0.13, 0.11, 0.12, 0.12]

seed_1235_entropy = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_1235_train   = [0.52, 0.88,   1.0,   0.96, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_1235_test    = [0.2, 0.35,   0.60,  0.50, 0.13, 0.11, 0.1,  0.1,  0.12, 0.11, 0.11, 0.12, 0.13]

seed_354_entropy = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8,  1]
seed_354_train   = [0.96, 0.68,   0.96,  0.84, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0]
seed_354_test    = [0.58, 0.50,   0.41,  0.35, 0.1,  0.1,  0.12, 0.13, 0.11, 0.15, 0.12, 0.15, 0.1]

seed_6872_entropy = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4, 0.5,  0.6,  0.8]
seed_6872_train   = [0.96, 0.92,   0.92,  0.92, 1.0,  1.0,  1.0,  1.0,  1.0, 1.0,  1.0,  1.0]
seed_6872_test    = [0.44, 0.37,   0.36,  0.25, 0.1,  0.12,  0.11, 0.11, 0.12, 0.12, 0.13, 0.12]

seed_7962_entropy = [0.0,  0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4,   0.5,  0.6,  0.8]
seed_7962_train   = [0.96, 0.96,   0.96,  0.96, 1.0,  1.0,  1.0,  1.0,  1.0,   1.0,  1.0,  1.0]
seed_7962_test    = [0.46, 0.45,   0.56,  0.53, 0.12, 0.12,  0.14, 0.12, 0.11, 0.12, 0.10, 0.12]

plt.plot(entropy, seed_0_train, 'bo-', label='train seed 0')
plt.plot(entropy, seed_0_test, 'bo--', label='test seed 0')
plt.plot(seed_1235_entropy, seed_1235_train, 'yo-', label='train seed 1235')
plt.plot(seed_1235_entropy, seed_1235_test, 'yo--', label='test seed 1235')
plt.plot(seed_354_entropy, seed_354_train, 'ro-', label='train seed 354')
plt.plot(seed_354_entropy, seed_354_test, 'ro--', label='test seed 354')
plt.plot(seed_6872_entropy, seed_6872_train, 'mo-', label='train seed 6872')
plt.plot(seed_6872_entropy, seed_6872_test, 'mo--', label='test seed 6872')
plt.plot(seed_7962_entropy, seed_7962_train, 'ko-', label='train seed 7962')
plt.plot(seed_7962_entropy, seed_7962_test, 'ko--', label='test seed 7962')
plt.xlabel('Entropy')
plt.ylabel('Reward')
plt.legend()
plt.show()

# 25 environments with different L2 coefficient
L2      =         [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4]
seed_0_train_L2 = [0.88,   0.88,  0.96, 0.96, 0.96, 1.0,  1.0,  1.0]
seed_0_test_L2  = [0.32,   0.40,  0.50, 0.49, 0.16, 0.14, 0.17, 0.14]

L2      =            [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4]
seed_7962_train_L2 = [0.96,   0.96,  0.96, 1.0,  1.0,  1.0,  1.0,  1.0]
seed_7962_test_L2  = [0.52,   0.41,  0.51, 0.39, 0.17, 0.15, 0.15, 0.155]

L2      =            [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4]
seed_1235_train_L2 = [0.71,   0.92,  0.96, 0.96, 0.96, 1.0,  1.0,  1.0]
seed_1235_test_L2  = [0.38,   0.51,  0.57, 0.34, 0.11, 0.13, 0.16, 0.12]

L2      =            [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4]
seed_354_train_L2 =  [0.88,   0.72,  0.92, 1.0,  0.96, 1.0,  1.0,  1.0]
seed_354_test_L2  =  [0.43,   0.38,  0.4,  0.37, 0.15, 0.11, 0.1,  0.1]

L2      =            [0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4]
seed_6872_train_L2 = [0.92,   0.92,  0.76, 0.92, 0.96, 1.0,  1.0,  0.96]
seed_6872_test_L2  = [0.47,   0.28,  0.23, 0.31, 0.12, 0.1,  0.11, 0.1]

plt.plot(L2, seed_0_train_L2, 'bo-', label='train seed 0')
plt.plot(L2, seed_0_test_L2, 'bo--', label='test seed 0')
plt.plot(L2, seed_7962_train_L2, 'go-', label='train seed 7962')
plt.plot(L2, seed_7962_test_L2, 'go--', label='test seed 7962')
plt.plot(L2, seed_1235_train_L2, 'ro-', label='train seed 1235')
plt.plot(L2, seed_1235_test_L2, 'ro--', label='test seed 1235')
plt.plot(L2, seed_354_train_L2, 'yo-', label='train seed 354')
plt.plot(L2, seed_354_test_L2, 'yo--', label='test seed 354')
plt.plot(L2, seed_6872_train_L2, 'ko-', label='train seed 6872')
plt.plot(L2, seed_6872_test_L2, 'ko--', label='test seed 6872')
plt.xlabel('L2')
plt.ylabel('Reward')
plt.legend()
plt.show()

# Rotation!
# reward for different number of envs
num_env_rot = [25,   50,   100,  150,  208, 312, 416]
train_rot   = [1.,   0.94, 0.92, 1.,   1.,  1.,   1.]
test_rot    = [0.14, 0.2,  0.81, 0.99, 1.,  1.,   1.]
plt.plot(num_env_rot, train_rot, 'bo-', label='train seed 0')
plt.plot(num_env_rot, test_rot, 'bo--', label='test seed 0')
plt.xlabel('num level')
plt.ylabel('Reward')
plt.legend()
plt.show()

# 25 environments with different entropy coefficient
entropy      = [0.0,  0.0001, 0.001, 0.01, 0.05, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6, 0.7, 0.8]
seed_0_train = [0.96, 0.88,   0.88,  0.92, 1.,   1.,   1.,   1.,   1.,   1.,   1.,  1.,  1. ]
seed_0_test  = [0.35, 0.3,    0.28,  0.33, 0.14, 0.11, 0.14, 0.13, 0.12, 0.13, 0.12,0.1, 0.12]

seed_1235_train = [0.92, 0.88, 0.92,  0.92, 1.,  1.,   1.,   1.,   1.,   1.,   1.,  1.,  1. ]
seed_1235_test  = [0.43, 0.35, 0.34,  0.33, 0.14,0.11, 0.11, 0.1, 0.12, 0.12, 0.1,0.11, 0.08]

seed_354_train = [0.92, 0.88, 0.8,   0.76,  1.,  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1. ]
seed_354_test  = [0.36, 0.35, 0.28,  0.30,  0.12,0.09, 0.13, 0.11, 0.13, 0.11, 0.15, 0.14, 0.12]

seed_7962_train = [0.88, 0.92, 0.84, 0.88,  1.,  1.,   0.96,   1.,   1.,   1.,   1.,   1.,   1. ]
seed_7962_test  = [0.36, 0.43, 0.36, 0.34,  0.17,0.16, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.15]

plt.plot(entropy, seed_0_train, 'bo-', label='train seed 0')
plt.plot(entropy, seed_0_test, 'bo--', label='test seed 0')
plt.plot(entropy, seed_1235_train, 'yo-', label='train seed 1235')
plt.plot(entropy, seed_1235_test, 'yo--', label='test seed 1235')
plt.plot(entropy, seed_354_train, 'ro-', label='train seed 354')
plt.plot(entropy, seed_354_test, 'ro--', label='test seed 354')
plt.plot(entropy, seed_7962_train, 'ko-', label='train seed 7962')
plt.plot(entropy, seed_7962_test, 'ko--', label='test seed 7962')
plt.title('With rotation')
plt.xlabel('Entropy')
plt.ylabel('Reward')
plt.legend()
plt.show()