
# Benchmarks for routing
# Each method accepts one data flow object
import numpy as np
from system_parameters import *

def route_close_neighbor_closest_to_destination(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:

        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    assert np.size(available_bands) > 0
    agent_dist_to_dest = agent.adhocnet.nodes_distances[agent.flow.frontier_node, agent.flow.dest]
    while True:
        level = Power_levels
        power = agent.adhocnet.transmit_power / level
        available_neighbors = agent.get_available_neighbors(available_bands, power)
        states = agent.get_state(available_bands, available_neighbors)
        dists_to_dest = agent.adhocnet.nodes_distances[available_neighbors, agent.flow.dest]
        dists_to_dest = np.append(dists_to_dest, agent_dist_to_dest)
        neighbor_index = np.argmin(dists_to_dest) # get the closest to terminal node one by one
        if neighbor_index < nodes_explored:  # found one neighbor closer to the destination
            next_hop = available_neighbors[neighbor_index]
            interferences = agent.obtain_interference_from_states(states, neighbor_index)
            band_index = np.argmin(interferences + agent.adhocnet.nodes_on_bands[available_bands,next_hop]*1e6) # use the power exposed feature to determine the optimal band
            action = [nodes_explored * (level - 1) + neighbor_index, packet]
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                                    rx=next_hop, state=states[band_index], action=action)
            break
        # reprobe as all the closest neighbors are further away from the destination
        action = [agent.n_actions-1, packet]
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[0],
                                rx=agent.flow.frontier_node, state=states[0], action=action)
        # This benchmark is not to be mixed with DQN routing, thus can exclude these nodes for further consideration
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, available_neighbors)
    return

def route_random(agent):
    # Generate a random node as next hop, transmit to it using a random band
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    assert np.size(available_bands) > 0
    band = np.random.choice(available_bands)
    band_index = np.where(band == available_bands)[0][0]
    # Generate a random node that's also available on the band
    available_nodes = np.where(agent.adhocnet.nodes_on_bands[band]==0)[0]
    available_nodes = agent.remove_nodes_excluded(available_nodes, agent.flow.exclude_nodes)
    assert np.size(available_nodes) > 0, "At least the destination node has to be there!"
    next_hop = np.random.choice(available_nodes)
    # Since random routing is to be used together with DQN, can't abort any nodes here
    exclude_nodes_original = np.copy(agent.flow.exclude_nodes)
    while True: # reproduce it as a sequence of actions by the agent
        # only probe on randomly selected bands (would be faster to find the node than probe over all available bands to the agent)
        power = agent.adhocnet.transmit_power
        available_neighbors = agent.get_available_neighbors(available_bands, power)
        states = agent.get_state(available_bands, available_neighbors)
        if next_hop in available_neighbors:
            neighbor_index = np.where(available_neighbors == next_hop)[0][0]
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=band,
                                    rx=next_hop, state=states[band_index], action=[neighbor_index, packet])
            break
        # reprobe (only store reprobe on the randomly selected band)
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=band,
                                rx=agent.flow.frontier_node, state=states[band_index], action=[agent.n_actions-1, packet])
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, available_neighbors)
    # restore the exclude nodes list
    agent.flow.exclude_nodes = np.append(exclude_nodes_original, next_hop)
    return

def route_close_neighbor_under_lowest_power(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    level = Power_levels
    power = agent.adhocnet.transmit_power / level
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    available_neighbors = agent.get_available_neighbors(available_bands, power)
    states = agent.get_state(available_bands, available_neighbors)
    interferences = [agent.obtain_interference_from_states(states, i)+agent.adhocnet.nodes_on_bands[available_bands,close_neighbor]*1e6
                            for i, close_neighbor in enumerate(available_neighbors)]
    interferences = np.transpose(interferences)
    assert np.shape(interferences) == (np.size(available_bands), np.size(available_neighbors))
    band_index, neighbor_index = np.unravel_index(np.argmin(interferences), np.shape(interferences))
    action = [nodes_explored * (level-1) + neighbor_index, packet]
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=available_neighbors[neighbor_index], state=states[band_index], action=action)
    return

# Route towards the neighbor with the strongest wireless channel from the frontier node
# Since using path-loss as the channel model, this is equivalent to selecting the closest neighbor.
# If the closest neighbor has more than one band available, go to the band with the lowest interference
def route_strongest_neighbor(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        #agent.adhocnet.add_counter()
        return
    level = Power_levels
    power = agent.adhocnet.transmit_power / level
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    available_neighbors = agent.get_available_neighbors(available_bands, power)
    states = agent.get_state(available_bands, available_neighbors)
    neighbor_index = 0  #  choose the closest neighbor
    interferences = agent.obtain_interference_from_states(states, neighbor_index)
    band_index = np.argmin(interferences+agent.adhocnet.nodes_on_bands[available_bands,available_neighbors[neighbor_index]]*1e6)
    action = [nodes_explored * (level-1) + neighbor_index, packet]
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=available_neighbors[neighbor_index], state=states[band_index], action=action)
    return

# Search within closest neighbors every available neighbor-band combination
def route_close_neighbor_with_largest_forward_rate(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    level = Power_levels
    power = agent.adhocnet.transmit_power / level
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    available_neighbors = agent.get_available_neighbors(available_bands, power)
    states = agent.get_state(available_bands, available_neighbors)
    rates = []
    for neighbor in available_neighbors:
        rates_one_neighbor = []
        for band in available_bands:
            if agent.adhocnet.nodes_on_bands[band, neighbor] == 1:
                rates_one_neighbor.append(-1)
            else:
                agent.adhocnet.powers[band, agent.flow.frontier_node] = TX_POWER # temporaily set
                rate, _, _, _ = agent.adhocnet.compute_link(tx=agent.flow.frontier_node, band=band, rx=neighbor)
                rates_one_neighbor.append(rate)
                agent.adhocnet.powers[band, agent.flow.frontier_node] = 0
        rates.append(rates_one_neighbor)
    rates = np.transpose(rates); assert np.shape(rates) == (np.size(available_bands), np.size(available_neighbors))
    assert np.max(rates) > 0, "Has to be one node/band available"
    band_index, neighbor_index = np.unravel_index(np.argmax(rates), np.shape(rates))
    action = [nodes_explored * (level - 1) + neighbor_index, packet]
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                            rx=available_neighbors[neighbor_index], state=states[band_index], action=action)
    return

# If all neighbors are behind the agent (i.e. the angle difference is greater than 90 degrees) then reprobe
def route_close_neighbor_best_forwarding_direction(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    level = Power_levels
    power = agent.adhocnet.transmit_power / level
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    angle_self_to_dest = agent.adhocnet.obtain_angle(node_1=agent.flow.frontier_node, node_2=agent.flow.dest)
    while True:
        available_neighbors = agent.get_available_neighbors(available_bands, power)
        states = agent.get_state(available_bands, available_neighbors)
        angles = []
        for neighbor in available_neighbors:
            angle_self_to_neighbor = agent.adhocnet.obtain_angle(node_1=agent.flow.frontier_node, node_2=neighbor)
            angles.append(agent.compute_angle_offset(angle_1=angle_self_to_dest, angle_2=angle_self_to_neighbor))
        angles.append(np.pi/2) # make sure we don't look for any neighbor going backwards
        neighbor_index = np.argsort(angles)[0]
        if neighbor_index < nodes_explored:  # a neighbor with forward angle
            interferences = agent.obtain_interference_from_states(states, neighbor_index)
            band_index = np.argmin(interferences + agent.adhocnet.nodes_on_bands[available_bands, available_neighbors[neighbor_index]]*1e6) # use the power exposed feature to determine the optimal band
            action = [nodes_explored * (level - 1) + neighbor_index, packet]
            agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[band_index],
                                    rx=available_neighbors[neighbor_index], state=states[band_index], action=action)
            break
        # reprobe as all the closest neighbors are with backward angle
        action = [agent.n_actions-1, packet]
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=available_bands[0],
                                    rx=agent.flow.frontier_node, state=states[0], action=action)
        agent.flow.exclude_nodes = np.append(agent.flow.exclude_nodes, available_neighbors)
    return

def route_destination_directly(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        return
    assert agent.flow.frontier_node == agent.flow.src
    available_bands = agent.adhocnet.get_available_bands(agent.flow.src)
    # For here, just directly compute power exposed field
    interferences = []
    for band in available_bands:
        _, _, interference, _ = agent.adhocnet.compute_link(tx=agent.flow.src, band=band, rx=agent.flow.dest)
        interferences.append(interference)
    band_index = np.argmin(interferences) # For the destination, no need to check for band availability
    # Just append pseudo state and action (not used in agent training)
    action = [0, packet]
    agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.src, band=available_bands[band_index],
                            rx=agent.flow.dest, state=np.zeros(agent.state_dim), action=action)
    return

def route_neighbor_with_largest_reward(agent):
    packet, _ = agent.flow.deliver_packet()
    first_packet = agent.flow.first_packet()
    if not first_packet:
        next_link = agent.flow.next_link()
        agent.adhocnet.add_link(flow_id=agent.id, tx=next_link[0], band=next_link[1], \
                                rx=next_link[2], state=next_link[3], action=next_link[4])
        #agent.adhocnet.add_counter()
        return
    available_bands = agent.adhocnet.get_available_bands(agent.flow.frontier_node)
    max_band = available_bands[0]  # set random default value
    max_reward, max_power, band_index, neighbor_index, max_neighbor = [-2*MAX_REWARD, 1, 0, 0, 0]
    max_states = None  # states for the best packet to deliver
    for tmp_power_level in range(Power_levels):
        tmp_power_level += 1  # range(1,Power_levels)
        tmp_power = agent.adhocnet.transmit_power/tmp_power_level
        available_neighbors = agent.get_available_neighbors(available_bands, tmp_power)
        for n_index, neighbor in enumerate(available_neighbors):
            for b_index, band in enumerate(available_bands):
                agent.adhocnet.powers[band, agent.flow.frontier_node] = tmp_power  # temporarily set
                states = agent.get_state(available_bands, available_neighbors)
                rate, _, _, _ = agent.adhocnet.compute_link(tx=agent.flow.frontier_node, band=band, rx=neighbor)
                tmp_action = [nodes_explored*tmp_power_level+n_index, packet]
                reward = agent.compute_reward(tx=agent.flow.frontier_node, rx=neighbor, rate=rate, action=tmp_action)
                if reward > max_reward or max_states is None:  # new max reward
                    max_reward, max_power, max_band, band_index = [reward, tmp_power_level, band, b_index]
                    neighbor_index, max_neighbor = [n_index, neighbor]
                    max_states = states
                agent.adhocnet.powers[band, agent.flow.frontier_node] = 0
    max_power -= 1  # range(0,Power_levels-1)
    next_action = [nodes_explored*max_power+neighbor_index, packet]
    if max_reward > 0:
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=max_band,\
            rx=max_neighbor, state=max_states[band_index], action=next_action)
    else:   # Miss the deadline, move to destination
        next_action = [agent.n_actions-1, packet]
        agent.adhocnet.add_link(flow_id=agent.id, tx=agent.flow.frontier_node, band=max_band, \
                               rx=agent.flow.dest, state=max_states[band_index], action=next_action)
    return
