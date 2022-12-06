
# Main script to train the agent

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange
import sys
import evaluate
sys.path.append("DQN/")
import adhoc_wireless_net
import agent
from replay_memory import Prioritized_Replay_Memory, Uniform_Replay_Memory
from system_parameters import *
import time

EXPONENT_VALUE = int(1e0)
INITIAL_EXPLORE_STEPS = int(30*EXPONENT_VALUE)
EPSILON_GREEDY_STEPS = int(300*EXPONENT_VALUE)
FINAL_CONVERGE_STEPS = int(50*EXPONENT_VALUE)
REPLAY_MEMORY_SIZE = int(100*EXPONENT_VALUE)
EVALUATE_FREQUENCY = int(5*EXPONENT_VALUE)
TARGET_NET_SYNC_FREQUENCY = int(5*EXPONENT_VALUE)

# Linear scheduler for hyper-parameters during training. Modified from the code from openai baselines:
#     https://github.com/openai/baselines/tree/master/baselines
class LinearSchedule():
    def __init__(self, initial_val, final_val, schedule_timesteps):
        self.initial_val = initial_val
        self.final_val = final_val
        self.schedule_timesteps = schedule_timesteps
    def value(self, timestep):
        fraction = max(min(float(timestep) / self.schedule_timesteps, 1.0),0.0)
        return self.initial_val + fraction * (self.final_val - self.initial_val)

if (__name__ == "__main__"):
    print("Training DQN with {} replay memory".format(REPLAY_MEMORY_TYPE))
    adhocnet = adhoc_wireless_net.AdHoc_Wireless_Net()
    agents = [agent.Agent(adhocnet, i) for i in range(adhocnet.n_flows)]
    memory = Prioritized_Replay_Memory(REPLAY_MEMORY_SIZE) if REPLAY_MEMORY_TYPE=="Prioritized" \
                else Uniform_Replay_Memory(REPLAY_MEMORY_SIZE) #is this the Di buffer mentioned in the pseudo code ?

    metrics = [] # store two metrics [Q_Loss, routing rate performance]
    policy_epsilon = LinearSchedule(initial_val=1.0, final_val=0.1, schedule_timesteps=EPSILON_GREEDY_STEPS) # how to choose the next step
    priority_ImpSamp_beta = LinearSchedule(initial_val=0.4, final_val=1.0, schedule_timesteps=EPSILON_GREEDY_STEPS)
    # 'Strongest Neighbor', 'Closest to Destination', 'Least Interfered', 'Largest Data Rate'
    # 'Best Direction', 'Max Reward'
    METHODS = ['Max Reward']
    for metrics_index, method in enumerate(METHODS): #starts to train the agents
        best_min_bottleneck_rate = -np.inf #we want to get the maximum
        metrics.append([])
        for i in trange(1, INITIAL_EXPLORE_STEPS+EPSILON_GREEDY_STEPS+FINAL_CONVERGE_STEPS+1):
            num = 0
            # refresh the layout
            adhocnet.update_layout() #before all powers and node bands
            for agent in agents[:-1]:
                while not agent.flow.first_packet():
                    agent.route_close_neighbor_closest_to_destination()
            for agent in agents[:-1]:
                num+=1
                while not agent.flow.destination_reached(): #building path for each flow
                    agent.route_close_neighbor_closest_to_destination()
            while not agents[-1].flow.destination_reached(): #why this done only in the last flow?
                # final settlement for Monte-Carlo estimatiuon based learning (MCMC estiate probability of what ?)
                epsilon_val = 0 if i>(INITIAL_EXPLORE_STEPS+EPSILON_GREEDY_STEPS) \
                                else policy_epsilon.value(i-1-INITIAL_EXPLORE_STEPS)
                agents[-1].route_epsilon_greedy(epsilon=epsilon_val)
            agents[-1].process_links(memory) #calculates the final link rate
            for agent in agents:
                agent.reset() #initialize
            if i >= INITIAL_EXPLORE_STEPS:  # Have gathered enough experiences, start training the agents
                # every 10 packets there is a need to check the reached packets procentage
                Q_loss = agents[-1].train(memory, priority_ImpSamp_beta.value(i-1-INITIAL_EXPLORE_STEPS))
                assert not np.isnan(Q_loss) #returns error if true (is Q_loss packet loss?)
                if (i % TARGET_NET_SYNC_FREQUENCY == 0):
                    agents[-1].sync_target_network() #synchronizes
                if (i % EVALUATE_FREQUENCY == 0):
                    for agent in agents[:-1]: # load the currently trained model parameters to evaluate
                        agent.sync_main_network_from_another_agent(agents[-1]) #why this is done ?
                    eval_results = evaluate.evaluate_routing(adhocnet, agents, method, n_layouts=3)
                    min_bottleneck_rate_avg = np.mean(np.min(eval_results[:, :, 0],axis=1))/1e6
                    Packets_reached = np.mean(eval_results[:, :, 3])
                    reach_packets = ((float)(Packets_reached / num_packets)) * 100 # checks procentage of reached packets
                    x5 = eval_results[:, :, 1]
                    num_links = np.mean(eval_results[:, :, 1])
                    tot_power = np.mean(eval_results[:, :, 4])
                    print("The results are:\n")
                    print("Q loss: {:.3f}; Min Bottleneck Rate(mbps): {:.3g}; Num Packets Reached(%): {:.2f}".format(
                        Q_loss, min_bottleneck_rate_avg, reach_packets))
                    metrics[metrics_index].append([i, Q_loss, min_bottleneck_rate_avg, reach_packets, num_links, tot_power])
                    if best_min_bottleneck_rate < min_bottleneck_rate_avg:
                        agents[-1].save_dqn_model()
                        best_min_bottleneck_rate = min_bottleneck_rate_avg
            adhocnet.total_del()





    #agents[-1].visualize_non_zero_rewards(memory)
    metrics = np.array(metrics)
    x_vals = metrics[0, :, 0] / 1e3
    with open('x_vals.txt', 'w') as f:
        for item in x_vals:
            f.write("%s\n" % item)
    colors = ['b', 'r', 'g']
    # fig1 - Q Loss
    fig, axes = plt.subplots()
    fig.suptitle("Q Loss for Individual Agent")
    axes.set_xlabel("Number of Layouts (1e3)")
    axes.set_ylabel("Q Loss")
    for index, method in enumerate(METHODS):
        axes.plot(x_vals, np.log(metrics[index, :, 1]), c=colors[index], label=method)
        with open('Q_Loss_%s.txt' % method, 'w') as f:
            for item in metrics[index, :, 1]:
                f.write("%s\n" % item)
    axes.legend(loc=1)
    plt.show()
    # fig2 - Min Bottleneck Rate
    fig, axes = plt.subplots()
    fig.suptitle("Min Bottleneck Rate for Individual Agent")
    axes.set_xlabel("Number of Layouts (1e3)")
    axes.set_ylabel("Min Bottleneck Rate (mbps)")
    for index, method in enumerate(METHODS):
        temp = metrics[index, :, 2]
        axes.plot(x_vals, metrics[index, :, 2], c=colors[index], label=method)
        with open('Min_Bottleneck_%s.txt' % method, 'w') as f:
            for item in metrics[index, :, 2]:
                f.write("%s\n" % item)
    axes.legend(loc=1)
    plt.show()
    # fig3 - Reached In Time Packets
    fig, axes = plt.subplots()
    fig.suptitle("Reached In Time Packets for Individual Agent")
    axes.set_xlabel("Number of Layouts (1e3)")
    axes.set_ylabel("Reached In Time Packets [%]")
    for index, method in enumerate(METHODS):
        axes.plot(x_vals, metrics[index, :, 3], c=colors[index])
        with open('Reach_Packets_%s.txt' % method, 'w') as f:
            for item in metrics[index, :, 3]:
                f.write("%s\n" % item)
    for index, method in enumerate(METHODS):
        axes.plot(np.NaN, np.NaN, c=colors[index], label=method)
    axes.legend(loc=1)
    plt.show()
    # fig4 - flow links length
    fig, axes = plt.subplots()
    fig.suptitle("flow links length for Individual Agent")
    axes.set_xlabel("Number of Layouts (1e3)")
    axes.set_ylabel("flow links length")
    for index, method in enumerate(METHODS):
        axes.plot(x_vals, metrics[index, :, 4], c=colors[index], label=method)
        with open('flow links_length_%s.txt' % method, 'w') as f:
            for item in metrics[index, :, 4]:
                f.write("%s\n" % item)
    axes.legend(loc=1)
    plt.show()
    # fig5 - Power Consumption
    fig, axes = plt.subplots()
    fig.suptitle("Power Consumption for Individual Agent")
    axes.set_xlabel("Number of Layouts (1e3)")
    axes.set_ylabel("Power Consumption")
    for index, method in enumerate(METHODS):
        axes.plot(x_vals, metrics[index, :, 5], c=colors[index], label=method)
        with open('Power_Consumption_%s.txt' % method, 'w') as f:
            for item in metrics[index, :, 5]:
                f.write("%s\n" % item)
    axes.legend(loc=1)
    plt.show()
    print("Script Finished Successfully!")

