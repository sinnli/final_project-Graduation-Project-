
# DQN Agent: the agent acts on the perspective of each flow and packets within it

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from data_flow import Data_Flow

sys.path.append("DQN/")
import dqn
import os
from routing_benchmarks import *
from system_parameters import *
import time

BAND_LINESTYLES = ['-', '-.', '--', ':', (0, (5,10)), (0, (5,1)), (0, (3,1,1,1)), (0, (3,1,1,1,1,1))]

class Agent():
    def __init__(self, adhocnet, flow_id, alt_flag = 0 ):

        self.bottleneck_link_index =0
        self.adhocnet = adhocnet
        self.id = flow_id  # the data flow this agent corresponds to
        if (alt_flag==0):
             self.flow = adhocnet.flows[flow_id]
        else:
            self.flow = adhocnet.alt_flows[flow_id]
        self.counter = 0
        self.test_time = time.time()
        self.flow.exclude_nodes = np.delete(np.arange(self.adhocnet.n_flows*2), self.flow.dest) # can delete by index
        self.state_features_per_neighbor = 8
        self.state_dim = nodes_explored * self.state_features_per_neighbor
        # action: one for each node explored, the last one for reprobing
        self.n_actions = nodes_explored * Power_levels + 1
        self._main_net = dqn.Dueling_DQN_Model(self.state_dim, self.n_actions).to(DEVICE)
        self._target_net = dqn.Dueling_DQN_Model(self.state_dim, self.n_actions).to(DEVICE)
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "DQN/Trained_Models/", "agent_{}_neighbors.pt".format(nodes_explored))
        self.loss_func = nn.SmoothL1Loss(reduction='none')
        self.optimizer = optim.Adam(self._main_net.parameters(), lr=1e-4)
        if os.path.exists(self._model_path):
            print("Agent {}: Loading pre-trained model from: {}".format(self.id, self._model_path))
            self._main_net.load_state_dict(torch.load(self._model_path))
            self._target_net.load_state_dict(torch.load(self._model_path))
        else:
            if (alt_flag==0):
                print("Agent {}: No pre-trained model found. Working from scratch.".format(self.id))
        if (alt_flag==0):
            print("Initialized agent for flow {}!".format(flow_id))
        else:
            print("Inititalized alternative agent.")

    def reset(self,alt_flag = 0):
        if alt_flag==0:
            #regular flow
            self.adhocnet.clear_flow(self.id)
            self.flow.exclude_nodes = np.delete(np.arange(self.adhocnet.n_flows * 2),
                                                self.flow.dest)  # can delete by index
        else:
            #alternertavive flow
            self.adhocnet.clear_flow(self.id,1)
            self.flow.exclude_nodes = np.delete(np.arange(self.adhocnet.n_flows * 2),
                                                self.flow.dest)  # can delete by index

        return

    def main_net_predict(self, states):  # Expect states shape: (minibatch_size X state_dim)
        return self._main_net(torch.tensor(states, dtype=torch.float32).to(DEVICE))

    def target_net_predict(self, states):
        return self._target_net(torch.tensor(states, dtype=torch.float32).to(DEVICE))

    def sync_main_network_from_another_agent(self, agent_another):
        assert self is not agent_another
        self._main_net.load_state_dict(agent_another._main_net.state_dict())
        return

    def sync_target_network(self):
        self._target_net.load_state_dict(self._main_net.state_dict())
        return

    def save_dqn_model(self,alt_flag = 0):
        torch.save(self._main_net.state_dict(), self._model_path)
        if alt_flag==0:
            print("Agent {}: Saved DQN model.".format(self.id))
        else:
            print("Alternative Agent saved.")
        return

    def route_epsilon_greedy(self, epsilon):
        rand_val = np.random.uniform()
        if rand_val < epsilon:
            self.route_random()
        else:
            self.route_DDRQN()
        return

    def route_DDRQN(self, visualize_axis=None):
        available_bands = self.adhocnet.get_available_bands(self.flow.frontier_node)
        packet, _ = self.flow.deliver_packet()
        first_packet = self.flow.first_packet()
        if not first_packet:
            next_link = self.flow.next_link()
            self.adhocnet.add_link(flow_id=self.id, tx=next_link[0], band=next_link[1], \
                                    rx=next_link[2], state=next_link[3], action=next_link[4])
            #self.received_count_pck_link.
            #self.send_count_pck_link
            return
        assert np.size(available_bands) > 0, "Flow {}: Sending from already fully occupied node {}!".format(self.flow_id, self.node_id)
        probe_attempt = 0
        while True:
            available_neighbors = self.get_available_neighbors(available_bands, self.adhocnet.transmit_power) # include nodes available on the bands
            states = self.get_state(available_bands, available_neighbors)
            Qs = self.main_net_predict(states).squeeze().detach().cpu().numpy()
            Qs = Qs - np.concatenate([self.adhocnet.nodes_on_bands[available_bands,:][:,available_neighbors], np.zeros([np.size(available_bands),51])], axis=1)*1e6
            band_index, neighbor_index = np.unravel_index(np.argmax(Qs), np.shape(Qs))
            # Policy enhancement I: previously aborted close neighbors won't be considered again
            self.flow.exclude_nodes = np.append(self.flow.exclude_nodes, available_neighbors[:neighbor_index])
            if visualize_axis:
                self.visualize_step(available_bands, available_neighbors, states, Qs, neighbor_index, band_index, probe_attempt, visualize_axis)
            # Policy enhancement II: if destination explored, don't bother go to further neighbors
            if self.flow.dest in available_neighbors:
                dest_index = np.where(available_neighbors == self.flow.dest)[0][0]
                if dest_index < neighbor_index:
                    band_index = np.argmax(Qs[:,dest_index]) # the destination should always have all bands available
                    neighbor_index = dest_index
                break
            if (neighbor_index < (nodes_explored)): # a promising next hop
                break
            else:
                probe_attempt += 1
                action = [neighbor_index, packet]
                self.adhocnet.add_link(flow_id=self.id, tx=self.flow.frontier_node, band=available_bands[band_index],
                                       rx=self.flow.frontier_node, state=states[band_index], action=action)
        action = [neighbor_index, packet]
        self.adhocnet.add_link(flow_id=self.id, tx=self.flow.frontier_node, band=available_bands[band_index],
                               rx=available_neighbors[neighbor_index], state=states[band_index], action=action)
        return

    def get_available_neighbors(self, available_bands, transmit_power):
        available_neighbors = np.argsort(self.adhocnet.nodes_distances[self.flow.frontier_node])
        available_neighbors = self.remove_nodes_excluded(available_neighbors, self.flow.exclude_nodes)
        fully_occupied_neighbors = np.where(np.min(self.adhocnet.nodes_on_bands[available_bands, :], axis=0) == 1)[0]
        available_neighbors = self.remove_nodes_excluded(available_neighbors, fully_occupied_neighbors)
        # check if any bands link require higher signal than transmit_power
        available_bands = np.array(available_bands)
        for neighbor in available_neighbors:
            if type(available_bands) != 'ndarray':
                signal = self.adhocnet.powers[available_bands, self.flow.frontier_node] * self.adhocnet.channel_losses[self.flow.frontier_node][neighbor] * ANTENNA_GAIN
                if signal.any() > transmit_power:
                    available_neighbors = self.remove_nodes_excluded(available_neighbors, neighbor)
            else:
                for band in available_bands:
                    signal = self.adhocnet.powers[band, self.flow.frontier_node] * self.adhocnet.channel_losses[self.flow.frontier_node][neighbor] * ANTENNA_GAIN
                    if signal.any() > transmit_power:
                        available_neighbors = self.remove_nodes_excluded(available_neighbors, neighbor)
                        break
        available_neighbors = available_neighbors[:nodes_explored]
        if np.size(available_neighbors) == 0:
            print("get here")
        assert np.size(available_neighbors) > 0, "Shouldn't exhaust node ever (at least the destination node is there)!"
        while np.size(available_neighbors) < nodes_explored:
            available_neighbors = np.append(available_neighbors, available_neighbors[-1])
        assert np.shape(available_neighbors) == (nodes_explored,)
        return available_neighbors

    def get_state(self, available_bands, available_neighbors):
        states = []
        tmp_states = []
        packet, sort_loc = self.flow.deliver_packet()
        # flow states
        # State 1: packets deadline field.
        deadline = packet[1]
        tmp_states.append(deadline)
        # State 2: packet location in sorted packets by deadline.
        tmp_states.append(sort_loc)
        # device states
        # State 3: distance from device to packet destination.
        packet_distance = self.adhocnet.nodes_distances[self.flow.frontier_node, self.flow.dest]
        tmp_states.append(packet_distance)
        # State 4: device package queue length.
        tmp_states.append(len(self.flow.flow_packets))
        # state 5: device remaining energy.
        tmp_states.append(self.adhocnet.get_remain_energy())
        # neighbor states
        for neighbor in available_neighbors:
            neighbor_state = []
            tmp_neighbor = list(np.copy(tmp_states))  # first states equal for all neighbors
            # State 6: Distance between the neighbor and the frontier node.
            tmp_neighbor.append(self.adhocnet.nodes_distances[self.flow.frontier_node, neighbor])
            # State 7: Distance between the neighbor and the destination.
            tmp_neighbor.append(self.adhocnet.nodes_distances[self.flow.dest, neighbor])
            for band in available_bands:
                tmp_new = list(np.copy(tmp_neighbor))  # each band is different state
                _, _, power_exposed_at_band, _ = self.adhocnet.compute_link(tx=self.flow.frontier_node, band=band, rx=neighbor)
                # State 8: Total interference the neighbor is exposed to in each frequency band.
                tmp_new.append(self.normalize_power(max(power_exposed_at_band, 1e-15)))
                neighbor_state.append(tmp_new)
            neighbor_state = np.array(neighbor_state)
            states.append(neighbor_state)
        states = np.concatenate(states, axis=1)
        return states

    def remove_nodes_excluded(self, nodes, nodes_excluded):
        nodes = np.array(nodes) # np.where works only on array instead of list
        nodes_tmp = nodes.copy() # don't iterate on the original array while modifying it
        if nodes_excluded is None:
            return nodes
        for node in nodes_tmp:
            if node in nodes_excluded:
                nodes = np.delete(nodes, np.where(nodes==node))
        return nodes

    # require the interference feature stored as the last state feature for each node
    def obtain_interference_from_states(self, states, action_index):
        return states[:, (action_index+1)*self.state_features_per_neighbor-1]

    # compute the angle offset used to determine the direction deviation torwards the destination
    def compute_angle_offset(self, angle_1, angle_2):
        angle_diff_one_direction = abs(angle_1 - angle_2)
        assert 0 <= angle_diff_one_direction < 2*np.pi
        return min(angle_diff_one_direction, 2*np.pi - angle_diff_one_direction)

    # Called after all agents reaches the destination to compute the final link rate
    def process_links(self, memory):
        assert self.flow.destination_reached()
        links = self.flow.get_links()
        transactions, rates_onwards, SINRs_onwards = [], [], []
        # process the last hop
        tx, band, rx, state, action = links[-1]
        assert rx == self.flow.dest and tx != rx
        # process links before the last hop
        tot_power = 0
        for tx, band, rx, state, action in links[::-1]:
            if tx != rx: # not a reprobe transaction
                rate, SINR, _, power = self.adhocnet.compute_link(tx=tx, band=band, rx=rx)
                if SINR < 0.5:
                    packet_id = action[1][2]
                    self.flow.reach_in_time[packet_id] = False
                tot_power += power
                rates_onwards.append(rate)
                SINRs_onwards.append(SINR)
            reward = self.compute_reward(tx, rx, rate, action)
            whether_done = (rx==self.flow.dest)
            transactions.append((state, action, reward, whether_done))
        self.flow.bottleneck_rate = np.min(rates_onwards)
        self.flow.tot_power = tot_power
        if memory!=None: # have to compare to None, since replay memory object has __len__() defined, empty would result in false
            for transaction in transactions[::-1]: # ordering matters within the replay buffer
                memory.add(transaction)
        return

    def normalize_power(self, power):
        return np.log10(power / 1e-10)

    def compute_reward(self, tx, rx, rate, action):
        amount_data = action[1][0]
        deadline = action[1][1]
        start = self.flow.start
        transmit_level = action[0] // nodes_explored + 1
        distance_device_to_dest = self.adhocnet.nodes_distances[tx, self.flow.dest]
        distance_next_hop_to_dest = self.adhocnet.nodes_distances[rx, self.flow.dest]
        D_param = (distance_device_to_dest-distance_next_hop_to_dest)*rate*amount_data
        duration_time = time.time() - start
        tmp_reward = delta_param*D_param + (1-delta_param)*TX_POWER/transmit_level
        if duration_time > deadline:
            reward = -MAX_REWARD
        elif rx == self.flow.dest:
            reward = MAX_REWARD
        else:
            reward = tmp_reward
        return reward

    def train_Q_Novel(self, states, actions, rewards, importance_weights):
        minibatch_size = np.shape(states)[0]
        self._main_net.train()
        self.optimizer.zero_grad()
        actions = actions[:,0].astype(np.int)
        Qs = self.main_net_predict(states)[np.arange(minibatch_size), actions]
        errors = self.loss_func(Qs, torch.tensor(rewards,dtype=torch.float32).to(DEVICE))
        loss = torch.mean(errors*torch.tensor(importance_weights, dtype=torch.float32).to(DEVICE))
        loss.backward()
        self.optimizer.step()
        return loss.item(), errors.detach().cpu().numpy()

    def train(self, memory, priority_ImpSamp_beta):
        if REPLAY_MEMORY_TYPE=="Prioritized":
            states, actions, rewards, next_states, whether_done, importance_weights, idxes = memory.sample(priority_ImpSamp_beta)
            loss, errors = self.train_Q_Novel(states, actions, rewards, importance_weights)
            memory.update_priorities(idxes, errors)
        else:
            states, actions, rewards, next_states, whether_done = memory.sample()
            importance_weights = np.ones_like(rewards)
            loss, _ = self.train_Q_Novel(states, actions, rewards, importance_weights)
        return loss

    def visualize_step(self, available_bands, closest_neighbors, states, criteria, action_index, band_index, probe_attempt, ax):
        if np.shape(criteria) == (self.n_actions,):
            criteria = np.tile(np.reshape(criteria, [1, self.n_actions]), [np.size(available_bands), 1])
        assert np.shape(criteria) == (np.size(available_bands), self.n_actions)
        prefix = "[{}th REPROBE] ".format(probe_attempt) if probe_attempt>0 else ""
        print("<<<<<{}Agent {} {}th step>>>>>".format(prefix, self.id, len(self.flow.get_links()) + 1))
        for i in range(nodes_explored):
            print("{}th neighbor: {}".format(i+1, states[0, i*self.state_features_per_neighbor: (i+1)*self.state_features_per_neighbor-1]))
        for band_index_tmp, band in enumerate(available_bands):
            print("Band{} power exposed: ".format(band), end="")
            for i in range(nodes_explored):
                interferences_on_node = self.obtain_interference_from_states(states, i)
                print("{:.1f}".format(interferences_on_node[band_index_tmp]).rjust(6), end="")
            prefix = ">>" if band_index_tmp == band_index else ""
            print("\n" + "{}Criteria: ".format(prefix).rjust(21), end="")
            for i in range(nodes_explored):
                print("{:.1f}".format(criteria[band_index_tmp, i]).rjust(6), end="")
            print("{:.1f}".format(criteria[band_index_tmp, -1]).rjust(6))
        self.visualize_route(ax)
        ax.scatter(self.adhocnet.nodes_locs[self.flow.frontier_node, 0], self.adhocnet.nodes_locs[self.flow.frontier_node, 1], color=self.flow.plot_color, marker="2", s=65)
        for i, (criterion, neighbor) in enumerate(zip(criteria[band_index], closest_neighbors)):
            text_str = '({}){:.1f}*'.format(i+1,criterion) if i==action_index else '({}){:.1f}'.format(i+1,criterion)
            ax.text(self.adhocnet.nodes_locs[neighbor, 0], self.adhocnet.nodes_locs[neighbor, 1], s=text_str, color=self.flow.plot_color, fontsize=8)
        ax.text(self.adhocnet.nodes_locs[self.flow.frontier_node, 0], self.adhocnet.nodes_locs[self.flow.frontier_node, 1], s='Q:{:.1f}'.format(criteria[band_index][-1]), color=self.flow.plot_color, fontsize=8)
        return

    # Can be called on partial routes
    def visualize_route(self, ax):
        links = self.flow.get_links()
        if len(links) == 0:
            return
        for tx, band, rx, _, _ in links:
            if tx != rx:  # a non-reprobe action
                ax.arrow(x=self.adhocnet.nodes_locs[tx][0], y=self.adhocnet.nodes_locs[tx][1],
                         dx=self.adhocnet.nodes_locs[rx][0] - self.adhocnet.nodes_locs[tx][0],
                         dy=self.adhocnet.nodes_locs[rx][1] - self.adhocnet.nodes_locs[tx][1],
                         length_includes_head=True, head_width=6, linestyle=BAND_LINESTYLES[band], color=self.flow.plot_color, linewidth=1.7)
        return

    # Visualize the reward distribution in memory
    def visualize_non_zero_rewards(self, memory):
        rewards_in_memory = [transaction[2] for transaction in memory.transactions_storage if (transaction[2]!=0 and transaction[2]!=np.inf)]
        plt.title("Non-zero reward distribution in replay memory")
        plt.hist(rewards_in_memory, bins=50)
        plt.show()
        return

    # Benchmark Callers
    def route_close_neighbor_closest_to_destination(self):
        route_close_neighbor_closest_to_destination(self)
        return

    def route_random(self):
        route_random(self)
        return

    def route_close_neighbor_under_lowest_power(self):
        route_close_neighbor_under_lowest_power(self)
        return

    def route_strongest_neighbor(self):
        route_strongest_neighbor(self)
        return

    def route_close_neighbor_with_largest_forward_rate(self):
        route_close_neighbor_with_largest_forward_rate(self)
        return

    def route_close_neighbor_best_forwarding_direction(self):
        route_close_neighbor_best_forwarding_direction(self)
        return

    def route_destination_directly(self):
        route_destination_directly(self)
        return

    def route_neighbor_with_largest_reward(self,alt_flag = 0):
        route_neighbor_with_largest_reward(self,alt_flag)
        return


    def process_links_find_bottleneck(self):
      #  assert self.flow.destination_reached()
        links = self.flow.get_links()
        rates_onwards = []
        # process the last hop
        tx, band, rx, _, _ = links[-1]
        assert rx == self.flow.dest and tx != rx

        # process links before the last hop
        for tx, band, rx, _, _ in links[::]:
            if tx != rx:  # not a reprobe transaction
                rate, _, _, _ = self.adhocnet.compute_link(tx=tx, band=band, rx=rx)
                rates_onwards.append(rate)

        # self.flow.bottleneck_rate = np.min(rates_onwards)
        self.bottleneck_link_index = np.argmin(rates_onwards)
        return

    def get_bottlenecklink_index(self):
        return self.bottleneck_link_index

    def get_alt_flow(self):
        return self.flow

    def get_flow_id(self):
        return self.id


