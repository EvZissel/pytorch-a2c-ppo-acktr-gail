import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

from gym import Env, spaces
from gym.utils import seeding

R_DISTS = {
    'fixed': np.array(
        [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1]]),
    'train': np.array(
        [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
         [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
         [0.1, 0.2, 0.3, 0.4, 0.8, 0.7]]),
    'train_5': np.array(
        [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
         [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
         [0.1, 0.2, 0.3, 0.4, 0.8, 0.7],
         [0.4, 0.3, 0.5, 0.2, 0.1, 0.6],
         [0.7, 0.5, 0.5, 0.8, 0.7, 0.5]]),
    'train_10': np.array(
        [[0.1, 0.7, 0.8, 0.3, 0.2, 0.1],
         [0.8, 0.6, 0.5, 0.4, 0.1, 0.2],
         [0.1, 0.2, 0.3, 0.4, 0.8, 0.7],
         [0.4, 0.3, 0.5, 0.2, 0.1, 0.6],
         [0.23, 0.4, 0.4, 0.12, 0.15, 0.53],
         [0.56, 0.1, 0.3, 0.34, 0.85, 0.32],
         [0.45, 0.3, 0.9, 0.1, 0.3, 0.74],
         [0.7, 0.5, 0.5, 0.8, 0.7, 0.5]]),
    'train_5_b': np.array(
        [[0.421, 0.256, 0.202, 0.376, 0.783, 0.872],
         [0.196, 0.519, 0.883, 0.608, 0.483, 0.577],
         [0.531, 0.955, 0.529, 0.394, 0.422, 0.119],
         [0.428, 0.813, 0.385, 0.736, 0.102, 0.474],
         [0.674, 0.604, 0.057, 0.961, 0.551, 0.289]]),
    'train_10_b': np.array(
        [[0.076, 0.159, 0.321, 0.06, 0.49, 0.944],#6
         [0.375, 0.728, 0.414, 0.184, 0.603, 0.383],#2
         [0.525, 0.841, 0.249, 0.355, 0.57, 0.934],#6
         [0.834, 0.466, 0.295, 0.038, 0.988, 0.073],#5
         [0.36, 0.805, 0.499, 0.113, 0.476, 0.717],#2
         [0.981, 0.411, 0.533, 0.572, 0.437, 0.778],#1
         [0.456, 0.084, 0.752, 0.925, 0.888, 0.037],#4
         [0.211, 0.353, 0.992, 0.538, 0.261, 0.603],#3
         [0.213, 0.774, 0.928, 0.257, 0.844, 0.062],#3
         [0.323, 0.625, 0.743, 0.216, 0.918, 0.337]]),#5
    'train_15': np.array(
        [[0.01891801, 0.78047335, 0.13120012, 0.14672056, 0.05945085, 0.01512077],  # 2
         [0.55368842, 0.52939036, 0.72235884, 0.9893397, 0.13359386, 0.75881976],  # 4
         [0.27498474, 0.93254546, 0.96660071, 0.04061222, 0.54954941, 0.55050314],  # 3
         [0.60444392, 0.91122862, 0.00310513, 0.27571081, 0.11575216, 0.02846643],  # 2
         [0.26380986, 0.73890683, 0.52585752, 0.61425811, 0.27416517, 0.99662264],  # 6
         [0.58043721, 0.13505389, 0.54580814, 0.21496861, 0.78752624, 0.53733551],  # 5
         [0.81661991, 0.59362704, 0.84309718, 0.51686857, 0.00501944, 0.03844539],  # 3
         [0.64545611, 0.02579325, 0.18808368, 0.89585609, 0.58277973, 0.84766255],  # 4
         [0.10375267, 0.11166068, 0.3657824, 0.15392747, 0.52210355, 0.56073071],  # 6
         [0.99626614, 0.02869837, 0.04904237, 0.50727304, 0.90120693, 0.51891313],  # 1
         [0.02470811, 0.69273466, 0.57286804, 0.36748555, 0.34783015, 0.31934225],  # 2
         [0.91138232, 0.79204389, 0.81667496, 0.03328246, 0.28039698, 0.35511668],  # 1
         [0.09022578, 0.21121132, 0.11206563, 0.97331993, 0.87026452, 0.96314169],  # 4
         [0.05946778, 0.89392153, 0.18829576, 0.18404594, 0.96824312, 0.69029678],  # 5
         [0.97775388, 0.47861151, 0.09594251, 0.43096626, 0.29108497, 0.52482397]]),  # 1
    'train_25': np.array(
        [[0.076, 0.159, 0.321, 0.06, 0.49, 0.944],  # 6
         [0.375, 0.728, 0.414, 0.184, 0.603, 0.383],  # 2
         [0.525, 0.841, 0.249, 0.355, 0.57, 0.934],  # 6
         [0.834, 0.466, 0.295, 0.038, 0.988, 0.073],  # 5
         [0.36, 0.805, 0.499, 0.113, 0.476, 0.717],  # 2
         [0.981, 0.411, 0.533, 0.572, 0.437, 0.778],  # 1
         [0.456, 0.084, 0.752, 0.925, 0.888, 0.037],  # 4
         [0.211, 0.353, 0.992, 0.538, 0.261, 0.603],  # 3
         [0.213, 0.774, 0.928, 0.257, 0.844, 0.062],  # 3
         [0.323, 0.625, 0.743, 0.216, 0.918, 0.337],  # 5
         [0.01891801, 0.78047335, 0.13120012, 0.14672056, 0.05945085, 0.01512077],  # 2
         [0.55368842, 0.52939036, 0.72235884, 0.9893397, 0.13359386, 0.75881976],  # 4
         [0.27498474, 0.93254546, 0.96660071, 0.04061222, 0.54954941, 0.55050314],  # 3
         [0.60444392, 0.91122862, 0.00310513, 0.27571081, 0.11575216, 0.02846643],  # 2
         [0.26380986, 0.73890683, 0.52585752, 0.61425811, 0.27416517, 0.99662264],  # 6
         [0.58043721, 0.13505389, 0.54580814, 0.21496861, 0.78752624, 0.53733551],  # 5
         [0.81661991, 0.59362704, 0.84309718, 0.51686857, 0.00501944, 0.03844539],  # 3
         [0.64545611, 0.02579325, 0.18808368, 0.89585609, 0.58277973, 0.84766255],  # 4
         [0.10375267, 0.11166068, 0.3657824, 0.15392747, 0.52210355, 0.56073071],  # 6
         [0.99626614, 0.02869837, 0.04904237, 0.50727304, 0.90120693, 0.51891313],  # 1
         [0.02470811, 0.69273466, 0.57286804, 0.36748555, 0.34783015, 0.31934225],  # 2
         [0.91138232, 0.79204389, 0.81667496, 0.03328246, 0.28039698, 0.35511668],  # 1
         [0.09022578, 0.21121132, 0.11206563, 0.97331993, 0.87026452, 0.96314169],  # 4
         [0.05946778, 0.89392153, 0.18829576, 0.18404594, 0.96824312, 0.69029678],  # 5
         [0.97775388, 0.47861151, 0.09594251, 0.43096626, 0.29108497, 0.52482397]]),  # 1
    'val_25': np.array(
        [[0.98948062, 0.37708537, 0.331013, 0.69333325, 0.13124786, 0.88920003],
         [0.68237267, 0.08225824, 0.21834965, 0.87589398, 0.70003436, 0.00747251],
         [0.1609193, 0.2854119, 0.7559782, 0.56310084, 0.31743964, 0.63907489],
         [0.35077164, 0.45320645, 0.83024209, 0.2019887, 0.00781196, 0.65621887],
         [0.0633701, 0.51787492, 0.86057012, 0.1542816, 0.34795482, 0.76183472],
         [0.05661702, 0.81537868, 0.54159523, 0.24150988, 0.80161598, 0.73641051],
         [0.08267289, 0.54077583, 0.92198802, 0.22542473, 0.68597108, 0.18538359],
         [0.02403415, 0.54196871, 0.84891156, 0.6448765, 0.48396221, 0.81369304],
         [0.20072841, 0.77927205, 0.60569877, 0.79944944, 0.4583432, 0.85380735],
         [0.35281025, 0.34826374, 0.78951095, 0.3102897, 0.75779163, 0.10942438],
         [0.13450107, 0.97161668, 0.96107628, 0.96341468, 0.45336276, 0.71177643],
         [0.6525207, 0.02784637, 0.53778085, 0.86305474, 0.57162949, 0.66869977],
         [0.37037574, 0.25950368, 0.62888508, 0.49180031, 0.02295407, 0.72735831],
         [0.83755545, 0.54925098, 0.81712092, 0.60140652, 0.64817518, 0.18898978],
         [0.02032885, 0.01263567, 0.92663379, 0.61574462, 0.86945788, 0.94492315],
         [0.21675219, 0.67281203, 0.57920967, 0.09366986, 0.5876348, 0.69001061],
         [0.63990541, 0.90458126, 0.01816806, 0.15577517, 0.6541077, 0.38570712],
         [0.86316116, 0.36810602, 0.64426344, 0.37772334, 0.15496514, 0.90347212],
         [0.38393111, 0.51326142, 0.38421059, 0.50533497, 0.02014076, 0.80952414],
         [0.83281928, 0.59629474, 0.01652585, 0.85030557, 0.74296672, 0.94735433],
         [0.40128429, 0.55985262, 0.26350271, 0.21997737, 0.42450969, 0.7212289],
         [0.13821577, 0.07893783, 0.21247979, 0.93528661, 0.02002084, 0.31862179],
         [0.63407434, 0.98519196, 0.98856102, 0.381836, 0.43946035, 0.95008078],
         [0.5446136, 0.24205709, 0.76318612, 0.31159283, 0.29714141, 0.60059216],
         [0.72662477, 0.92363499, 0.73680628, 0.4617172, 0.82390518, 0.26765032]]),
    'test': np.array(
        [[0.811, 0.42, 0.301, 0.9, 0.421, 0.59],
         [0.047, 0.556, 0.778, 0.302, 0.147, 0.771],
         [0.832, 0.727, 0.052, 0.188, 0.922, 0.002],
         [0.708, 0.069, 0.435, 0.315, 0.813, 0.063],
         [0.573, 0.382, 0.805, 0.539, 0.975, 0.83],
         [0.822, 0.371, 0.46, 0.357, 0.276, 0.535],
         [0.441, 0.021, 0.614, 0.914, 0.446, 0.25],
         [0.201, 0.365, 0.702, 0.205, 0.764, 0.179],
         [0.004, 0.242, 0.612, 0.8, 0.589, 0.639],
         [0.897, 0.651, 0.364, 0.986, 0.777, 0.277],
         [0.883, 0.246, 0.936, 0.474, 0.369, 0.499],
         [0.871, 0.523, 0.668, 0.044, 0.923, 0.09],
         [0.014, 0.183, 0.664, 0.606, 0.601, 0.327],
         [0.728, 0.49, 0.203, 0.809, 0.464, 0.409],
         [0.46, 0.379, 0.989, 0.117, 0.621, 0.323],
         [0.851, 0.893, 0.052, 0.736, 0.662, 0.761],
         [0.391, 0.912, 0.934, 0.365, 0.479, 0.371],
         [0.765, 0.769, 0.216, 0.254, 0.045, 0.17],
         [0.172, 0.697, 0.215, 0.848, 0.47, 0.974],
         [0.098, 0.652, 0.707, 0.229, 0.791, 0.377],
         [0.044, 0.85, 0.351, 0.949, 0.561, 0.836],
         [0.515, 0.002, 0.164, 0.626, 0.816, 0.67],
         [0.298, 0.919, 0.235, 0.568, 0.357, 0.522],
         [0.304, 0.766, 0.887, 0.245, 0.737, 0.116],
         [0.765, 0.218, 0.112, 0.164, 0.025, 0.702],
         [0.252, 0.814, 0.513, 0.877, 0.561, 0.244],
         [0.924, 0.87, 0.036, 0.424, 0.108, 0.331],
         [0.724, 0.736, 0.14, 0.676, 0.217, 0.507],
         [0.833, 0.642, 0.176, 0.856, 0.88, 0.944],
         [0.325, 0.927, 0.142, 0.661, 0.66, 0.765],
         [0.93, 0.716, 0.767, 0.292, 0.993, 0.181],
         [0.133, 0.746, 0.892, 0.792, 0.143, 0.215],
         [0.928, 0.872, 0.187, 0.792, 0.882, 0.7],
         [0.315, 0.758, 0.308, 0.696, 0.537, 0.436],
         [0.708, 0.629, 0.704, 0.769, 0.026, 0.808],
         [0.552, 0.972, 0.205, 0.898, 0.982, 0.715],
         [0.031, 0.003, 0.688, 0.898, 0.772, 0.112],
         [0.621, 0.362, 0.513, 0.93, 0.518, 0.167],
         [0.286, 0.099, 0.844, 0.303, 0.858, 0.568],
         [0.232, 0.565, 0.154, 0.137, 0.42, 0.269],
         [0.153, 0.561, 0.063, 0.855, 0.166, 0.476],
         [0.65, 0.153, 0.001, 0.608, 0.373, 0.28],
         [0.591, 0.951, 0.68, 0.248, 0.652, 0.217],
         [0.614, 0.073, 0.671, 0.489, 0.157, 0.033],
         [0.004, 0.213, 0.533, 0.746, 0.763, 0.171],
         [0.79, 0.807, 0.552, 0.736, 0.965, 0.748],
         [0.725, 0.135, 0.571, 0.309, 0.958, 0.233],
         [0.518, 0.583, 0.107, 0.275, 0.574, 0.961],
         [0.913, 0.287, 0.382, 0.678, 0.136, 0.612],
         [0.125, 0.72, 0.833, 0.405, 0.683, 0.67],
         [0.734, 0.45, 0.232, 0.024, 0.546, 0.854],
         [0.097, 0.016, 0.342, 0.861, 0.218, 0.364],
         [0.746, 0.4, 0.708, 0.704, 0.369, 0.331],
         [0.494, 0.252, 0.21, 0.568, 0.021, 0.592],
         [0.423, 0.924, 0.079, 0.789, 0.973, 0.417],
         [0.33, 0.218, 0.422, 0.191, 0.362, 0.147],
         [0.819, 0.225, 0.309, 0.878, 0.461, 0.579],
         [0.72, 0.01, 0.064, 0.583, 0.825, 0.779],
         [0.309, 0.165, 0.204, 0.173, 0.197, 0.225],
         [0.293, 0.348, 0.306, 0.843, 0.743, 0.82],
         [0.806, 0.246, 0.025, 0.373, 0.071, 0.683],
         [0.131, 0.859, 0.108, 0.899, 0.88, 0.011],
         [0.325, 0.044, 0.13, 0.661, 0.05, 0.07],
         [0.794, 0.157, 0.366, 0.965, 0.475, 0.935],
         [0.008, 0.395, 0.589, 0.27, 0.189, 0.032],
         [0.94, 0.615, 0.445, 0.953, 0.677, 0.062],
         [0.213, 0.032, 0.948, 0.146, 0.857, 0.318],
         [0.695, 0.887, 0.14, 0.332, 0.137, 0.825],
         [0.027, 0.668, 0.966, 0.086, 0.567, 0.758],
         [0.223, 0.67, 0.101, 0.256, 0.699, 0.9],
         [0.125, 0.884, 0.01, 0.21, 0.124, 0.929],
         [0.114, 0.675, 0.155, 0.946, 0.897, 0.946],
         [0.067, 0.956, 0.463, 0.19, 0.1, 0.119],
         [0.585, 0.731, 0.925, 0.355, 0.482, 0.454],
         [0.479, 0.339, 0.621, 0.482, 0.395, 0.556],
         [0.557, 0.098, 0.937, 0.506, 0.169, 0.701],
         [0.952, 0.02, 0.759, 0.027, 0.827, 0.253],
         [0.167, 0.817, 0.81, 0.097, 0.26, 0.177],
         [0.662, 0.929, 0.767, 0.129, 0.278, 0.483],
         [0.604, 0.357, 0.985, 0.104, 0.787, 0.561],
         [0.859, 0.955, 0.094, 0.427, 0.189, 0.167],
         [0.677, 0.136, 0.32, 0.208, 0.444, 0.891],
         [0.87, 0.128, 0.174, 0.612, 0.268, 0.89],
         [0.909, 0.613, 0.816, 0.486, 0.534, 0.904],
         [0.165, 0.488, 0.522, 0.86, 0.127, 0.698],
         [0.557, 0.426, 0.842, 0.358, 0.792, 0.94],
         [0.178, 0.501, 0.825, 0.478, 0.567, 0.267],
         [0.008, 0.162, 0.956, 0.03, 0.772, 0.302],
         [0.379, 0.082, 0.626, 0.368, 0.566, 0.705],
         [0.417, 0.615, 0.887, 0.745, 0.183, 0.751],
         [0.577, 0.675, 0.749, 0.711, 0.804, 0.637],
         [0.045, 0.574, 0.559, 0.139, 0.24, 0.935],
         [0.16, 0.199, 0.042, 0.062, 0.118, 0.283],
         [0.371, 0.611, 0.345, 0.715, 0.393, 0.833],
         [0.263, 0.968, 0.483, 0.821, 0.886, 0.346],
         [0.117, 0.257, 0.223, 0.711, 0.085, 0.865],
         [0.37, 0.889, 0.922, 0.514, 0.401, 0.791],
         [0.417, 0.661, 0.454, 0.599, 0.532, 0.888],
         [0.646, 0.652, 0.749, 0.403, 0.477, 0.309],
         [0.569, 0.32, 0.082, 0.793, 0.249, 0.591]]
    )
}

def generate_random_rewards(size=6):
    """Generates a random assignments of reward
    :param size: number of arms
    """
    r = np.random.uniform(size=size)
    return r


class HBanditEnv(Env):
    """
    MAB env
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n=6, arms=None, steps=10, r_dist_name="6_v0", normalize_r=True, free_exploration=0,
                 recurrent=False, obs_recurrent=False):
        self.n = 6
        self.seed()
        if arms is None and r_dist_name == "random":
            self.arms = RandArms(n=n)
        elif arms == 'rand_obs':
            self.arms = ObsRandChooseArms(n=n, r_dist_name=r_dist_name)
        elif arms is None:
            self.arms = RandChooseArms(n=n, r_dist_name=r_dist_name)
        # elif arms is None:
        #     self.arms = ConstantArms(n=n, r_dist_name=r_dist_name)
        self.reward_range = (0, 1)
        self.t = 0
        self.steps = steps
        self.normalize_r = normalize_r
        self.free_exploration = free_exploration
        self.recurrent = recurrent
        self.obs_recurrent = obs_recurrent

        self.action_space = spaces.Discrete(self.n)
        if self.recurrent:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=[2], dtype=np.float32)
        elif self.obs_recurrent:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=[2+self.n], dtype=np.float32)
            self.obs = self.arms.arms
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=[2 * self.steps], dtype=np.float32)

        self.s = np.zeros(2 * self.steps)
        self.lastaction = None

        self.task_id = -1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.arms.reset()
        self.t = 0
        self.s = np.zeros(2 * self.steps)
        self.lastaction = None
        if self.recurrent:
            next_state = np.array([0, 0])
        elif self.obs_recurrent:
            self.obs = self.arms.arms
            next_state = np.concatenate((self.obs, np.array([0, 0])))
        else:
            next_state = self.s.copy()
        return next_state

    def step(self, a):
        r, r_rank = self.arms.eval(a)
        self.s[2 * self.t] = a
        self.s[2 * self.t+1] = r
        self.lastaction = a
        self.t += 1
        done = (self.t == self.steps)
        reward = r
        if self.t < self.free_exploration:
            reward = 0
        elif self.normalize_r:
            reward = r_rank
        if self.recurrent:
            next_state = np.array([a, r_rank])
        elif self.obs_recurrent:
            next_state = np.concatenate((self.obs, np.array([a, r_rank])))
        else:
            next_state = self.s.copy()
        return next_state, reward, done, {"time": self.t}

    def render(self, mode='human'):
        print(self.arms)
        print(self.t)
        print(self.s)

    def set_task_id(self, task_id):
        task_id = task_id % self.arms.max_ind()
        self.arms.task_id = task_id
        self.arms.reset()
        self.task_id = task_id


class ConstantArms():
    def __init__(self, n=6, r_dist_name="6_v0"):
        self.arms = np.array(R_DISTS[r_dist_name])
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        return


class RandChooseArms():
    def __init__(self, n=6, r_dist_name='train'):
        self.r_list = R_DISTS[r_dist_name]
        self.arms = np.array(self.r_list[np.random.randint(len(self.r_list))])
        self.n = len(self.arms)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n-1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def max_ind(self):
        return len(self.r_list)

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        if self.task_id < 0:
            task_id = np.random.randint(len(self.r_list))
        else:
            task_id = self.task_id
        self.arms = np.array(self.r_list[task_id])
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        return


# random observation vector (not related to the actual arms
class ObsRandChooseArms():
    def __init__(self, n=6, r_dist_name='train'):
        self.r_list = R_DISTS[r_dist_name]
        self.arms = np.array(self.r_list[np.random.randint(len(self.r_list))])
        self.n = len(self.arms)
        best_arm = np.random.randint(self.n)
        self.ranks = np.zeros(self.n)
        self.ranks[best_arm] = 1
        self.task_id = -1

    def max_ind(self):
        return len(self.r_list)

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        if self.task_id < 0:
            task_id = np.random.randint(len(self.r_list))
        else:
            task_id = self.task_id
        self.arms = np.array(self.r_list[task_id])
        self.n = len(self.arms)
        best_arm = task_id % self.n
        self.ranks = np.zeros(self.n)
        self.ranks[best_arm] = 1
        return


class RandArms():
    def __init__(self, n=6):
        self.arms = generate_random_rewards()
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        self.task_id = -1

    def max_ind(self):
        return 1e8

    def eval(self, arm):
        return self.arms[arm], self.ranks[arm]

    def reset(self):
        self.arms = generate_random_rewards()
        self.n = len(self.arms)
        # temp = self.arms.argsort()
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(self.n)
        inds = self.arms.argsort()
        ranks = np.empty_like(inds)
        # tmp = np.arange(self.n)
        tmp = np.zeros(self.n)
        tmp[-1] = self.n - 1
        ranks[inds] = tmp
        self.ranks = ranks / (self.n-1)
        return
