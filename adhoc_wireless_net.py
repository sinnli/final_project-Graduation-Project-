
# Class for Ad-hoc wireless network
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

import numpy as np
from scipy.spatial.distance import pdist, squareform
from system_parameters import *
from data_flow import Data_Flow

def prepare_tx_rx_location_ratios(n_flows, separation_fraction):
    n1 = int(np.ceil(n_flows/2))
    n2 = n_flows - n1
    tx_locs_ratios = []
    rx_locs_ratios = []
    for i in range(n1):
        loc_1, loc_2 = [0, i*separation_fraction], [1, 1 - i * separation_fraction]
        if i%2 == 0:
            tx_locs_ratios.append(loc_1)
            rx_locs_ratios.append(loc_2)
        else:
            tx_locs_ratios.append(loc_2)
            rx_locs_ratios.append(loc_1)
    for i in range(n2):
        loc_1, loc_2 = [(i+1) * separation_fraction, 0], [1-(i+1)*separation_fraction, 1]
        if i % 2 == 0:
            tx_locs_ratios.append(loc_1)
            rx_locs_ratios.append(loc_2)
        else:
            tx_locs_ratios.append(loc_2)
            rx_locs_ratios.append(loc_1)
    tx_locs_ratios, rx_locs_ratios = np.array(tx_locs_ratios), np.array(rx_locs_ratios)
    assert np.shape(tx_locs_ratios) == np.shape(rx_locs_ratios) == (n_flows, 2), "{}, {}".format(tx_locs_ratios, rx_locs_ratios)
    return tx_locs_ratios, rx_locs_ratios

AdHocLayoutSettings = {
    'A': {"field_length": 1000,
          "n_flows": 2,
          "n_bands": 8,
          "transmit_power": TX_POWER,
          "txs_rxs_length_ratios": prepare_tx_rx_location_ratios(n_flows=2, separation_fraction=1/5),
          "mobile_nodes_distrib": [6, 8, 7, 6, 5, 10, 8, 9, 6]
          },
    'B': {"field_length": 5000,
          "n_flows": 10,
          "n_bands": 32,
          "transmit_power": TX_POWER,
          "txs_rxs_length_ratios": prepare_tx_rx_location_ratios(n_flows=10, separation_fraction=1 / 20),
          "mobile_nodes_distrib": [19, 16, 21, 18, 14, 24, 17, 20, 19]
          },
    'C': {"field_length": 5000,
          "n_flows": 10,
          "n_bands": 32,
          "transmit_power": TX_POWER,
          "txs_rxs_length_ratios": prepare_tx_rx_location_ratios(n_flows=10, separation_fraction=1 / 20),
          "mobile_nodes_distrib": [36, 34, 42, 38, 46, 40, 54, 45, 42]
          },
    'D': {"field_length": 1000,
          "n_flows": 5,
          "n_bands": 8,
          "transmit_power": TX_POWER,
          "txs_rxs_length_ratios": prepare_tx_rx_location_ratios(n_flows=5, separation_fraction=1 / 5),
          "mobile_nodes_distrib": [6, 8, 7, 6, 5, 10, 8, 9, 6]
          }
}

# [54, 74, 62, 78, 86, 60, 64, 75, 58
class AdHoc_Wireless_Net():
    def __init__(self):
        self.counter = 0
        self.test_time = time.time()
        self.layout_setting = AdHocLayoutSettings['D']
        self.field_length = self.layout_setting['field_length']
        self.n_flows = self.layout_setting['n_flows']
        self.transmit_power = self.layout_setting['transmit_power']
        self.flows = []
        self.alt_flows = []
        for i in range(self.n_flows):
            destination = np.random.randint(i+1, self.n_flows*2-1)
            self.flows.append(Data_Flow(flow_id=i, src=i, dest=destination))
            self.alt_flows.append(Data_Flow(flow_id=i,src = 1,dest = 1))
            packet_id = 1
            for j in range(num_packets):  # initial packets for each flow
                amount = np.random.randint(data_size[0], data_size[1])
                deadline = np.random.randint(deadline_time[0], deadline_time[1])
                packet = [amount, deadline, packet_id]
                self.flows[i].add_packet(packet)
                packet_id += 1
        self.n_nodes = 2 * self.n_flows + sum(self.layout_setting["mobile_nodes_distrib"])
        self.n_bands = self.layout_setting['n_bands']
        self.nodes_on_bands = np.zeros([self.n_bands, self.n_nodes])
        self.used_bands = np.zeros(self.n_bands)
        self.powers = np.zeros([self.n_bands, self.n_nodes])
        self.energy = INITIAL_ENERGY
        self.update_layout()

    # Refreshing on a larger time scale
    def update_layout(self):
        # ensure the network is cleared
        assert np.all(self.powers == np.zeros([self.n_bands, self.n_nodes]))
        assert np.all(self.nodes_on_bands == np.zeros([self.n_bands, self.n_nodes])) #new error
        self.used_bands = np.zeros(self.n_bands)
        txs_locs = self.layout_setting['txs_rxs_length_ratios'][0] * self.field_length
        rxs_locs = self.layout_setting['txs_rxs_length_ratios'][1] * self.field_length
        assert np.shape(txs_locs) == np.shape(rxs_locs)  == (self.n_flows, 2)
        self.nodes_locs = np.concatenate([txs_locs, rxs_locs], axis=0)
        for index, (i, j) in enumerate(itertools.product(range(3), range(3))):
            x = np.random.uniform(low=i/3*self.field_length, high=(i+1)/3*self.field_length, size=[self.layout_setting["mobile_nodes_distrib"][index], 1])
            y = np.random.uniform(low=j/3*self.field_length, high=(j+1)/3*self.field_length, size=[self.layout_setting["mobile_nodes_distrib"][index], 1])
            self.nodes_locs = np.concatenate([self.nodes_locs, np.concatenate([x, y], axis=1)], axis=0)
        assert np.shape(self.nodes_locs) == (self.n_nodes, 2)
        self.nodes_distances = squareform(pdist(self.nodes_locs))
        assert np.min(np.eye(self.n_nodes) + self.nodes_distances) > 0
        # compute channel losses based on ITU-1411 path loss model
        nodes_distances_tmp = self.nodes_distances + np.eye(self.n_nodes)
        signal_lambda = 2.998e8 / CARRIER_FREQUENCY
        Rbp = 4 * TX_HEIGHT * RX_HEIGHT / signal_lambda
        Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * TX_HEIGHT * RX_HEIGHT)))
        sum_term = 20 * np.log10(nodes_distances_tmp / Rbp)
        Tx_over_Rx = Lbp + 6 + sum_term + ((nodes_distances_tmp > Rbp).astype(int)) * sum_term
        self.channel_losses = np.power(10, (-Tx_over_Rx / 10))  # convert from decibel to absolute
        # Set self-to-self path loss to zero, corresponding to no self-interference contribution
        self.channel_losses *= (1 - np.eye(self.n_nodes))
        assert np.shape(self.channel_losses) == np.shape(self.nodes_distances)
        return

    def get_available_bands(self, node_id):
        available_bands = np.where(self.nodes_on_bands[:,node_id]==0)[0]
        return available_bands

    def add_link(self, flow_id, tx, band, rx, state, action, alt_flag = 0):
        if alt_flag==1:
            flow = self.alt_flows[flow_id]
        else:
            flow = self.flows[flow_id]
        if flow.first_packet():
            # assert self.nodes_on_bands[band, tx] == self.nodes_on_bands[band, rx] == 0
            # assert self.powers[band, tx] == self.powers[band, rx] == 0
            if tx != rx:  # not a reprobe
                level = action[0] // nodes_explored
                power = self.transmit_power / (level+1)
                self.powers[band, tx] = power
                self.nodes_on_bands[band, tx] = 1
                self.nodes_on_bands[band, rx] = 1
                self.used_bands[band] += 1
        flow.add_link(tx, band, rx, state, action,alt_flag)
        return

    def get_remain_energy(self):
        link_factor = 0
        duration_time = []
        now = time.time()
        for flow in self.flows:
            links = flow.get_links()
            link_factor += len(links)
            duration_time.append(np.int(flow.get_start_time() - now))
        link_factor = np.power(10, -(link_factor/10))
        duration = np.mean(duration_time)
        duration_factor = np.power(10, -(duration / 10))
        return duration_factor * link_factor * self.energy

    def clear_flow(self, flow_id,alt_flag = 0):
        if (alt_flag==0):
            flow = self.flows[flow_id]
        else:
            flow = self.alt_flows[flow_id]
        for tx, band, rx, _, _ in flow.get_links():
            if tx == rx: # reprobe ->??
                continue
            self.powers[band, tx] = 0
            self.nodes_on_bands[band, tx] = 0
            self.nodes_on_bands[band, rx] = 0
        flow.reset(alt_flag)
        return

    def add_counter(self):
        self.counter += 1
        if self.counter % 5 == 0:
            print(time.time() - self.test_time)
            self.test_time = time.time()
        return

    def compute_link(self, tx, band, rx):
        signal = self.powers[band, tx] * self.channel_losses[tx][rx] * ANTENNA_GAIN
        interfere_powers = np.copy(self.powers[band])
        interfere_powers[tx] = 0
        interference = np.dot(self.channel_losses[rx], interfere_powers)
        SINR = signal / (interference + NOISE_POWER)
        rate = BANDWIDTH * np.log2(1 + SINR)
        return rate, SINR, interference, self.powers[band, tx]

    # centered at node 1, return the angle from node 1 to node 2 (zero angle is to the right)
    def obtain_angle(self, node_1, node_2):
        delta_x, delta_y = self.nodes_locs[node_2] - self.nodes_locs[node_1]
        angle = np.arctan2(delta_y, delta_x)
        angle = 2*np.pi + angle if angle < 0 else angle # convert to continuous 0~2pi range
        return angle

    def visualize_layout(self):
        fig, ax = plt.subplots()
        #for i in range(self.n_flows):
            #ax.scatter(self.nodes_locs[self.flows[i].src, 0], self.nodes_locs[self.flows[i].src, 1], color=self.flows[i].plot_color, marker="^", s=50)
            #ax.scatter(self.nodes_locs[self.flows[i].dest, 0], self.nodes_locs[self.flows[i].dest, 1], color=self.flows[i].plot_color, marker="*", s=50)
        for i in range(2*self.n_flows, self.n_nodes):
            ax.scatter(self.nodes_locs[i, 0], self.nodes_locs[i, 1], color='k', marker="o", s=20)
        ax.set_aspect('equal')
        fig.suptitle("Network 5000X5000")
        plt.show()
        return


    def move_layout(self):
        #print("The nodes locactions of the network:")
        # print(self.nodes_locs)
        epsilon1 = np.random.rand(75,2)*10
        epsilon2 = np.random.rand(75,2)*10
        epsilon = epsilon1-epsilon2
        self.nodes_locs+=epsilon

        for node in self.nodes_locs:
            for i in range(0,2):
                if (node[i]<0):
                    node[i] = 0
                elif (node[i]>self.field_length):
                    node[i] = self.field_length

        return

    def total_del(self):
        self.nodes_on_bands = np.zeros([self.n_bands, self.n_nodes])
        self.powers = np.zeros([self.n_bands, self.n_nodes])
        return


if __name__ == "__main__":
    adhocnet = AdHoc_Wireless_Net()
    ax = plt.gca()
    adhocnet.visualize_layout()
    plt.show()
