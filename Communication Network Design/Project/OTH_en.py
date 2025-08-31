class OTN1:
    def __init__(self):
        self.odu_10_in = 0 # Number of 10G ODUs received from the outside
        self.odu_100_in = 0 # Number of 100G ODUs received from the outside
        self.odu_10_out = 0 # Number of 10G ODUs sent to the outside
        self.odu_100_out = 0 # Number of 100G ODUs sent to the outside
        self.odu_10_forward_send = 0 # Number of 10G ODUs forwarded to OTN2
        self.odu_100_forward_send = 0 # Number of 100G ODUs forwarded to OTN2
        self.odu_10_forward_receive = 0 # Number of 10G ODUs received from OTN2
        self.odu_100_forward_receive = 0 # Number of 100G ODUs received from OTN2

    def receive_odu(self, odu_type, count):
        """Receive ODU from the outside world and update the counter according to the type"""
        if odu_type == '10':
            self.odu_10_in += count
        elif odu_type == '100':
            self.odu_100_in += count

    def send_odu(self, odu_type, count):
        """Send ODU to the outside world and update the counter according to the type"""
        if odu_type == '10':
            self.odu_10_out += count
        elif odu_type == '100':
            self.odu_100_out += count

    def forward_odu_to_otn2(self):
        """Automatically updates the number of ODUs forwarded to OTN2 based on the number of ODUs received from the outside world"""
        self.odu_10_forward_send = self.odu_10_in
        self.odu_100_forward_send = self.odu_100_in

    def forward_odu_from_otn2(self):
        """Automatically update the number of ODUs received from OTN2 based on the number of ODUs sent to the outside world"""
        self.odu_10_forward_receive = self.odu_10_out
        self.odu_100_forward_receive = self.odu_100_out

    def calculate_io_cards(self):
        """Calculate the number of I/O cards required to combine receive/send and forward send/receive ODUs"""
        io_cards = 0

        # Calculate the I/O cards required to receive and send ODUs from the outside world
        total_in_out_10 = self.odu_10_in + self.odu_10_out
        total_in_out_100 = self.odu_100_in + self.odu_100_out
        io_cards += (total_in_out_10 + 9) // 10  # Every 10 10G ODUs consume one I/O card
        io_cards += total_in_out_100  # Each 100G ODU consumes one I/O card

        # Calculate the I/O cards required for ODUs forwarded to and received from OTN2
        total_forward_10 = self.odu_10_forward_send + self.odu_10_forward_receive
        total_forward_100 = self.odu_100_forward_send + self.odu_100_forward_receive
        io_cards += (total_forward_10 + 9) // 10  # Every 10 10G ODUs consume one I/O card
        io_cards += total_forward_100  # Each 100G ODU consumes one I/O card

        return io_cards

    def calculate_capacity(self):
        """Calculate capacity consumption, each I/O card consumes 100Gb/s"""
        return self.calculate_io_cards() * 100

class OTN2:
    def __init__(self, node_name):
        self.node_name = node_name  # Node name
        # ODU exchange between OTN2 and OTN1
        self.odu_10_otn1 = 0  # Number of 10G ODUs exchanged with OTN1
        self.odu_100_otn1 = 0  # Number of 100G ODUs exchanged with OTN1

        # ODU exchange between OTN2 and the physical layer (external exchange)
        self.odu_10_physical_in = 0  # Number of 10G ODUs received from the outside
        self.odu_100_physical_in = 0  # Number of 100G ODUs received from the outside
        self.odu_10_physical_out = 0  # Number of 10G ODUs sent to the outside
        self.odu_100_physical_out = 0  # Number of 100G ODUs sent to the outside

        # ODU exchange between OTN2 and other nodes (optical path exchange)
        self.node_exchanges = {}  # Record ODU exchanges with other nodes, format: {node: {'10_in': x, '100_in': y, '10_out': z, '100_out': w}}

    def exchange_with_otn1(self):
        """Exchange ODUs with OTN1, automatically update based on the number of ODUs exchanged with the outside"""
        # Number of ODUs from OTN1 to OTN2 equals the number of ODUs received from the outside
        self.odu_10_otn1 = self.odu_10_physical_in
        self.odu_100_otn1 = self.odu_100_physical_in

        # Number of ODUs from OTN2 to OTN1 equals the number of ODUs sent to the outside
        self.odu_10_otn1 += self.odu_10_physical_out
        self.odu_100_otn1 += self.odu_100_physical_out

    def exchange_with_physical(self, odu_type, count, direction):
        """Exchange ODUs with the outside world, update counters"""
        if direction == 'in':
            if odu_type == '10':
                self.odu_10_physical_in += count
            elif odu_type == '100':
                self.odu_100_physical_in += count
        elif direction == 'out':
            if odu_type == '10':
                self.odu_10_physical_out += count
            elif odu_type == '100':
                self.odu_100_physical_out += count

    def exchange_with_node(self, odu_type, count, direction, target_node):
        """Exchange ODUs with other nodes, update counters"""
        if target_node not in self.node_exchanges:
            self.node_exchanges[target_node] = {'10_in': 0, '100_in': 0, '10_out': 0, '100_out': 0}

        if direction == 'in':
            if odu_type == '10':
                self.node_exchanges[target_node]['10_in'] += count
            elif odu_type == '100':
                self.node_exchanges[target_node]['100_in'] += count
        elif direction == 'out':
            if odu_type == '10':
                self.node_exchanges[target_node]['10_out'] += count
            elif odu_type == '100':
                self.node_exchanges[target_node]['100_out'] += count

    def calculate_io_cards(self):
        """Calculate the number of I/O cards required for OTN2"""
        io_cards = 0

        # Calculate I/O cards for exchange with OTN1
        io_cards_otn1 = (self.odu_10_otn1 + 9) // 10 + self.odu_100_otn1
        io_cards += io_cards_otn1

        # Calculate I/O cards for exchange with the outside world
        total_physical_10 = self.odu_10_physical_in + self.odu_10_physical_out
        total_physical_100 = self.odu_100_physical_in + self.odu_100_physical_out
        io_cards_physical = (total_physical_10 + 9) // 10 + total_physical_100
        io_cards += io_cards_physical

        # Calculate I/O cards for exchange with other nodes
        for node, exchanges in self.node_exchanges.items():
            total_node_10 = exchanges['10_in'] + exchanges['10_out']
            total_node_100 = exchanges['100_in'] + exchanges['100_out']
            io_cards_node = (total_node_10 + 9) // 10 + total_node_100
            io_cards += io_cards_node

        return io_cards

    def calculate_physical_connections(self):
        """Calculate the number of optical paths"""
        # Calculate optical paths for exchange with the outside world
        total_physical_bandwidth = (self.odu_10_physical_in + self.odu_10_physical_out) * 10 + \
                                   (self.odu_100_physical_in + self.odu_100_physical_out) * 100
        physical_connections = (total_physical_bandwidth + 499) // 500

        # Calculate optical paths for exchange with other nodes
        for node, exchanges in self.node_exchanges.items():
            total_node_bandwidth = (exchanges['10_in'] + exchanges['10_out']) * 10 + \
                                   (exchanges['100_in'] + exchanges['100_out']) * 100
            physical_connections += (total_node_bandwidth + 499) // 500

        return physical_connections

    def calculate_capacity(self):
        """Calculate the total capacity consumption for OTN2"""
        # Capacity consumption for exchange with OTN1
        capacity_otn1 = self.calculate_io_cards_otn1() * 100

        # Capacity consumption for optical paths
        capacity_physical = self.calculate_physical_connections() * 500

        return capacity_otn1 + capacity_physical

    def calculate_io_cards_otn1(self):
        """Calculate the number of I/O cards for exchange with OTN1"""
        return (self.odu_10_otn1 + 9) // 10 + self.odu_100_otn1
    
    def nb_of_odu(self):
        """Calculate the number of ODUs exchanged with OTN2"""
        nbOfOdu = self.odu_10_physical_in + self.odu_100_physical_in + self.odu_10_physical_out + self.odu_100_physical_out

        return nbOfOdu

class Network:
    def __init__(self):
        self.nodes_otn1 = {}  # Store OTN1 instances for all nodes
        self.nodes_otn2 = {}  # Store OTN2 instances for all nodes
        self.connections = set()  # Store connections between nodes, format: {(node1, node2)}

    def add_node(self, node_name):
        """Add a node to the network"""
        if node_name not in self.nodes_otn1:
            self.nodes_otn1[node_name] = OTN1()
        if node_name not in self.nodes_otn2:
            self.nodes_otn2[node_name] = OTN2(node_name)

    def add_connection(self, node1, node2):
        """Add a connection between nodes"""
        if node1 not in self.nodes_otn1:
            self.add_node(node1)
        if node2 not in self.nodes_otn1:
            self.add_node(node2)
        self.connections.add((node1, node2))
        self.connections.add((node2, node1))  # Bidirectional connection

    def process_services(self, services):
        """Process the list of services, update ODU exchange information for nodes"""
        for service in services:
            odu_size = service['odu_size']  # ODU size (10G or 100G)
            path = service['path']          # Service path (list of nodes)

            # Traverse each node in the path
            for i in range(len(path)):
                node_name = path[i]

                # If it's the starting node
                if i == 0:
                    # Receive ODU from the outside
                    self.nodes_otn1[node_name].receive_odu(odu_size, 1)
                    self.nodes_otn2[node_name].exchange_with_physical(odu_size, 1, 'in')

                # If it's the ending node
                if i == len(path) - 1:
                    # Send ODU to the outside
                    self.nodes_otn1[node_name].send_odu(odu_size, 1)
                    self.nodes_otn2[node_name].exchange_with_physical(odu_size, 1, 'out')

                # If it's not the last node, forward ODU to the next node
                if i < len(path) - 1:
                    next_node_name = path[i + 1]
                    # Send ODU to the next node
                    self.nodes_otn2[node_name].exchange_with_node(odu_size, 1, 'out', next_node_name)

    def propagate_odu_exchanges(self):
        """Automatically retrieve ODUs from connected nodes"""
        for node_name, otn2 in self.nodes_otn2.items():
            for target_node, exchanges in otn2.node_exchanges.items():
                if '10_out' in exchanges:
                    self.nodes_otn2[target_node].exchange_with_node('10', exchanges['10_out'], 'in', node_name)
                if '100_out' in exchanges:
                    self.nodes_otn2[target_node].exchange_with_node('100', exchanges['100_out'], 'in', node_name)

    def calculate_wdm_count(self):
        """Calculate the number of WDMs used in the entire network"""
        wdm_count = 0
        # Traverse all connections
        for connection in self.connections:
            node1, node2 = connection
            if node1 < node2:  # Avoid double-counting bidirectional connections
                # Calculate total bandwidth from node1 to node2
                bandwidth_1_to_2 = (self.nodes_otn2[node1].node_exchanges.get(node2, {'10_out': 0, '100_out': 0})['10_out'] * 10 + 
                                   self.nodes_otn2[node1].node_exchanges.get(node2, {'10_out': 0, '100_out': 0})['100_out'] * 100)
                # Calculate total bandwidth from node2 to node1
                bandwidth_2_to_1 = (self.nodes_otn2[node2].node_exchanges.get(node1, {'10_out': 0, '100_out': 0})['10_out'] * 10 + 
                                   self.nodes_otn2[node2].node_exchanges.get(node1, {'10_out': 0, '100_out': 0})['100_out'] * 100)
                # Total bandwidth
                total_bandwidth = bandwidth_1_to_2 + bandwidth_2_to_1
                # Calculate the number of WDMs
                wdm_count += (total_bandwidth + 499) // 500
        return wdm_count

    def calculate_total_io_cards(self):
        """Calculate the total number of I/O cards for OTN1 and OTN2 in the entire network"""
        total_io_cards = sum(otn1.calculate_io_cards() for otn1 in self.nodes_otn1.values())
        total_io_cards += sum(otn2.calculate_io_cards() for otn2 in self.nodes_otn2.values())
        return total_io_cards

    def calculate_total_capacity(self):
        """Calculate the total capacity consumption of the entire network"""
        total_capacity = sum(otn1.calculate_capacity() for otn1 in self.nodes_otn1.values())
        total_capacity += sum(otn2.calculate_capacity() for otn2 in self.nodes_otn2.values())
        return total_capacity

    def calculate_otn2_non_otn1_odu(self):
        """Calculate the size of ODUs exchanged by OTN2 (excluding ODUs exchanged with OTN1)"""
        total_odu_10 = 0
        total_odu_100 = 0
        for otn2 in self.nodes_otn2.values():
            for node, exchanges in otn2.node_exchanges.items():
                total_odu_10 += exchanges['10_in'] + exchanges['10_out']
                total_odu_100 += exchanges['100_in'] + exchanges['100_out']
        return total_odu_10 + total_odu_100

    def run_network(self, services):
        """Run the network"""
        # Process the list of services
        self.process_services(services)
        # Automatically retrieve ODUs from connected nodes --OTN2
        self.propagate_odu_exchanges()

        # Update the number of ODUs exchanged between OTN1 and OTN2
        for otn1 in self.nodes_otn1.values():
            otn1.forward_odu_to_otn2()
            otn1.forward_odu_from_otn2()
        for otn2 in self.nodes_otn2.values():
            otn2.exchange_with_otn1()
        can_use = 1
        # Calculate the number of I/O cards and capacity consumption for each node
        for node_name, otn1 in self.nodes_otn1.items():
            if(otn1.calculate_io_cards() > 70):
                # print(f"Node {node_name} I/O card count does not meet requirements")
                can_use = 0
            if(otn1.calculate_capacity() > 12288):
                # print(f"Node {node_name} OTN1 total capacity does not meet requirements")
                can_use = 0

        for node_name, otn2 in self.nodes_otn2.items():
            if(otn2.calculate_io_cards() > 70):
                # print(f"Node {node_name} OTN2 I/O card count does not meet requirements")
                can_use = 0
            if(otn2.calculate_capacity() > 12288):
                # print(f"Node {node_name} OTN2 total capacity does not meet requirements")
                can_use = 0
            if((otn2.nb_of_odu()) > 100):
                # print(f"Node {node_name} ODU count does not meet requirements")
                can_use = 0
            #print(f"Node {node_name} OTN2 I/O Number: {otn2.calculate_io_cards()}, Lightpaths: {otn2.calculate_physical_connections()}, Capacity: {otn2.calculate_capacity()} Gb/s")
        
        # total_io_cards = self.calculate_total_io_cards()
        # total_capacity = self.calculate_total_capacity()
        # otn2_odu = self.calculate_otn2_non_otn1_odu()
        
        # Calculate the total number and capacity of I/O cards
        # print(f"Total number of I/O cards in the node: {total_io_cards}")
        # print(f"Total capacity consumption of the node: {total_capacity} Gb/s")
        # print(f"Number of ODUs exchanged by the node: {otn2_odu}")

        # Calculate the number of WDMs used in the entire network
        wdm_count = self.calculate_wdm_count()
        # print(f"Number of WDM used in the entire network: {wdm_count}")
        return can_use, wdm_count

'''
# Example test
network = Network()

# Add nodes
network.add_node('A')
network.add_node('B')
network.add_node('C')

# Add connections between nodes
network.add_connection('A', 'B')
network.add_connection('B', 'C')

# List of services
services = [
    {'odu_size': '10', 'path': ['A', 'B', 'C']},  # Service1: 10Gb/s, from A to B to C
    {'odu_size': '100', 'path': ['A', 'B']},       # Service2: 100Gb/s, from A to B
    {'odu_size': '100', 'path': ['B', 'C']}        # Service3: 100Gb/s, from B to C
]

(can_use, wdm_count) = network.run_network(services)
print(f"Number of WDMs used in the entire network: {wdm_count}")
if(can_use == 1):
    print("Can be used")
else:
    print("Cannot be used")
'''