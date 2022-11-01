
# All environmental numerical settings

import os
import random
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPLAY_MEMORY_TYPE = "Uniform"
# wireless network parameters
BANDWIDTH = 5e6
CARRIER_FREQUENCY = 2.4e9
_NOISE_dBm_perHz = -130
NOISE_POWER = np.power(10, ((_NOISE_dBm_perHz-30)/10)) * BANDWIDTH
TX_HEIGHT = 1.5
RX_HEIGHT = 1.5
_TX_POWER_dBm = 30
TX_POWER = np.power(10, (_TX_POWER_dBm - 30) / 10)
_ANTENNA_GAIN_dB = 2.5
ANTENNA_GAIN = np.power(10, (_ANTENNA_GAIN_dB/10))

# Set random seed
RANDOM_SEED = 234
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

delta_param = 0.5
MAX_REWARD = 10
Power_levels = 6
INITIAL_ENERGY = 100
num_packets = 1000
nodes_explored = 10
# num_packets = np.random.randint(2, 10)
data_size = [10, 100]
deadline_time = [20, 50]
