
# Data flow object


import numpy as np
import time

FLOW_COLORS = ['r', 'g', 'b']*10

class Data_Flow():
    def __init__(self, flow_id, src, dest):
        self.flow_id = flow_id
        self.src = src
        self.dest = dest
        self._links = []
        self.flow_packets = []
        self.sorted_packets = []
        self.finished_packets = []
        self.start = time.time()
        self.deliver_index = 0  # counter of packets got to destination
        self.packet_flow_length = 0
        self.reach_in_time = []
        self.packet_link_index = 0
        self.exclude_nodes = None
        self.frontier_node = src
        self.bottleneck_rate = -1
        self.tot_power = 0
        self.test_time = time.time()
        self._n_reprobes = 0  # number of reprobes during routing this flow
        self.plot_color = FLOW_COLORS[flow_id]

    def add_link(self, tx, band, rx, state, action):  # return true if this link end a packet.
        assert tx == self.frontier_node
        deliver_packet, _ = self.deliver_packet()
        if self.first_packet():
            self._links.append((tx, band, rx, state, action))

        else:
            self.packet_link_index += 1
            #this should be an indicator of how many times the link was used
            #in order to know how many links are sending in the same time
        if tx == rx:  # it's a reprobe
            self._n_reprobes += 1
        else:
            self.frontier_node = rx
            if self.frontier_node == self.dest:  # packet reach dest, deliver next packet in the list
                reach = (time.time() - self.start) < deliver_packet[1]  # duration < deadline
                self.reach_in_time.append(reach)
                self.start = time.time()
                if self.deliver_index+1 < len(self.flow_packets):  # move to next packet to deliver
                    if self.deliver_index == 0:  # reduce deadline
                        self.change_deadline("div") # since the first packets has a longer deadline
                    self.deliver_index += 1
                    self.packet_link_index = 0
                    self.frontier_node = self.src
                    self.exclude_nodes = None
                return
            self.exclude_nodes = np.append(self.exclude_nodes, rx)
        return

    def get_links(self):
        return self._links.copy()  # make sure it's not altered from outside

    def destination_reached(self):
        return self.frontier_node == self.dest

    def get_number_of_reprobes(self):
        return self._n_reprobes

    def first_packet(self):
        return self.deliver_index == 0

    def add_packet(self, packet):
        self.flow_packets.append(packet)
        self.sorted_packets = self.sort_packets()
        return

    def get_start_time(self):
        return self.start

    def number_reached_packets(self):
        result = 0
        for reach_in_time in self.reach_in_time:
            if reach_in_time:
                result += 1
        return result

    def change_deadline(self, op):
        packets = self.flow_packets
        tmp = []
        for packet in packets:
            if op == "div":
                packet[1] /= 500000000
            elif op == "mul":
                packet[1] *= 500000000
            tmp.append(packet)
        self.flow_packets = tmp
        return

    def sort_packets(self):  # sort packets by deadline
        sorted_packets = []
        if self.flow_packets is None:
            return sorted_packets
        for packet in self.flow_packets:
            if len(sorted_packets) == 0:
                sorted_packets.append(packet)
            else:
                insert_flag = True
                for index, tmp in enumerate(sorted_packets):
                    if packet[1] < tmp[1]:  # check deadline field
                        sorted_packets.insert(index, packet)
                        insert_flag = False
                        break
                if insert_flag:
                    sorted_packets.append(packet)  # this packet have the highest deadline
        return sorted_packets

    def deliver_packet(self):
        sort_packets = self.sorted_packets
        packet = sort_packets[self.deliver_index]
        return packet, self.deliver_index

    def next_link(self):
        links = self._links
        return links[self.packet_link_index]

    def reset(self):
        self._links = []
        self.exclude_nodes = None
        self.change_deadline("mul")
        self.start = time.time()
        self.deliver_index = 0
        self.packet_flow_length = 0
        self.reach_in_time = []
        self.packet_link_index = 0
        self.frontier_node = self.src
        self.bottleneck_rate = None
        self._n_reprobes = 0
        return
