import random
import networkx as nx
from collections import deque
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from OTH_en import Network  # import Network class

max_path = 3
nbOfNode = 100

class TrafficSimulator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = [i for i in range(num_nodes)]  # Use numbers as node names
        self.graph = nx.Graph()
        self.path_cache = {}  # Cache calculated paths
        
    def create_network(self, topology_file=None, edge_probability=0.5):
        """
        Create a network topology, either reading edges from a file or randomly generating edges and writing them to a file.
        
        :param topology_file: Topology file path. If None, edges are randomly generated.
        :param edge_probability: The probability of randomly generating an edge (a floating point number between 0 and 1)
        """
        self.graph.add_nodes_from(self.nodes)
        
        if topology_file:
            # Reading topology from file
            with open(topology_file, 'r') as file:
                for line in file:
                    node1, node2 = map(int, line.strip().split())
                    self.graph.add_edge(node1, node2)
        else:
            # Randomly generate edges
            edges = []
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if random.random() < edge_probability:  # Generate edges with a certain probability
                        edges.append((i, j))
            # Adding edges to the graph
            self.graph.add_edges_from(edges)
    def find_k_paths(self, source, destination, max_path):
        """
        Find up to max_path paths from source to destination in an unweighted graph.
        
        :param graph: networkx Graph
        :param source: starting node
        :param destination: ending node
        :param max_path: maximum number of paths to find
        :return: list of paths (each path is a list of nodes)
        """
        if source == destination:
            return [[source]]
        
        queue = deque([[source]])  # Queue for BFS, storing paths
        paths = []  # List to store found paths
        visited = set()  # Set to track visited nodes
        
        while queue and len(paths) < max_path:
            path = queue.popleft()  # Get the next path to explore
            node = path[-1]  # Last node in the path
            
            if node == destination:
                paths.append(path)  # Found a valid path
                continue
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in path:  # Prevent cycles
                    new_path = path + [neighbor]
                    queue.append(new_path)
                    visited.add(neighbor)
        return paths[:max_path]  # Return only up to max_path results

    def generate_services(self, num_services=300):
        """Generates a specified number of random services"""
        services = []
        for _ in range(num_services):
            # Randomly select source and destination nodes
            source, destination = random.sample(self.nodes, 2)
            # Randomly select service rate (10G or 100G)
            rate = random.choice(['10', '100'])
            
            # Generate k possible paths for each service
            k_paths = self.find_k_paths(source, destination, max_path)
            services.append({
                'source': source,
                'destination': destination,
                'rate': rate,
                'possible_paths': k_paths
            })
        #print(k_paths)
        return services

    def calculate_no_grooming_lightpaths(self, services):
        """Calculate the number of light paths required without grooming"""
        lightpaths = {}  # Record the number of lightpaths on each link
        
        for service in services:
            # Use the shortest path
            path = service['possible_paths'][0]
            
            # Calculate the number of lightpaths required for each link on the path
            for i in range(len(path) - 1):
                link = tuple(sorted([path[i], path[i+1]]))
                if link not in lightpaths:
                    lightpaths[link] = {
                        '10G': 0,
                        '100G': 0
                    }
                
                # Add lightpaths based on service rate
                rate = service['rate']
                if rate == '10':
                    lightpaths[link]['10G'] += 1
                else:  # 100G
                    lightpaths[link]['100G'] += 1
        
        # Calculate the total number of optical paths (one optical path for every 500G capacity)
        total_lightpaths = 0
        for link_load in lightpaths.values():
            # For 10G services, one 500G optical path is required for every 50
            lightpath_10g = (link_load['10G'] + 49) // 50
            # For 100G services, one 500G optical path is required for every five
            lightpath_100g = (link_load['100G'] + 4) // 5
            total_lightpaths += lightpath_10g + lightpath_100g
            
        return total_lightpaths, lightpaths

def plot_results(results):
    import matplotlib.pyplot as plt

def plot_results(results):
    # Extracting data
    num_services = [entry['num_services'] for entry in results]
    no_grooming_lightpaths = [entry['no_grooming_lightpaths'] for entry in results]
    grooming_lightpaths = [entry['grooming_lightpaths'] for entry in results]
    blocked_percentage = [entry['blocked_percentage'] for entry in results]
    
    # Calculate Savings Ratio
    savings_ratio = [(ng - g) / ng if ng != 0 else 0 for ng, g in zip(no_grooming_lightpaths, grooming_lightpaths)]
    
    # Create multiple windows and display them simultaneously
    fig1 = plt.figure()
    plt.plot(num_services, no_grooming_lightpaths, marker='o', linestyle='-', color='b', label='No Grooming')
    plt.plot(num_services, grooming_lightpaths, marker='s', linestyle='-', color='g', label='Grooming')
    plt.xlabel('Number of Services')
    plt.ylabel('Lightpaths')
    plt.title('No Grooming vs Grooming Lightpaths')
    plt.legend()
    plt.grid(True)
    
    fig2 = plt.figure()
    plt.plot(num_services, savings_ratio, marker='d', linestyle='-', color='m')
    plt.xlabel('Number of Services')
    plt.ylabel('Savings Ratio')
    plt.title('Savings Ratio (Grooming vs No Grooming)')
    plt.grid(True)
    
    fig3 = plt.figure()
    plt.plot(num_services, blocked_percentage, marker='^', linestyle='-', color='r')
    plt.xlabel('Number of Services')
    plt.ylabel('Blocked Percentage')
    plt.title('Number of Services vs Blocked Percentage')
    plt.grid(True)
    
    # Shwo
    plt.show()

def main():
    # Creating a Simulation Instance
    simulator = TrafficSimulator(nbOfNode)  # Nodes of the network
    simulator.create_network()  # Create a network topology
    
    # Initialization results
    results = []
    num_services = 30  # Starting Serving Quantity
    blocked_percentage = 0
    
    while blocked_percentage < 0.01:  # Until 0.01% of the traffic is blocked
        print(f"\nNumber of testing services: {num_services}")
        
        # Generate Service
        services = simulator.generate_services(num_services)
        
        # Calculation without grooming
        no_grooming_lightpaths, link_details = simulator.calculate_no_grooming_lightpaths(services)
        


        # Processing Services
        grooming_services = []
        previce_services = []
        blocked_services = 0
        for service in services:
            for path_number in range(max_path):
                # Use the existing Network class to calculate grooming situations
                network = Network()
                
                # Adding nodes and connections
                for node in simulator.nodes:
                    network.add_node(node)
                for edge in simulator.graph.edges():
                    network.add_connection(edge[0], edge[1])

                previce_services = grooming_services
                grooming_services.append({
                    'odu_size': service['rate'],
                    'path': service['possible_paths'][path_number]
                })
                
                (can_use, wdm_count) = network.run_network(grooming_services)
                #print(grooming_services)
                if(can_use == 0):
                    if(path_number + 1 >= max_path):
                        #print(f"Blocked{grooming_services}")
                        grooming_services.remove({
                            'odu_size': service['rate'],
                            'path': service['possible_paths'][path_number]
                        })
                        #print(f"Deleted{grooming_services}")

                        blocked_services += 1
                        print("Blocked+1")
                        path_number = 0
                        break
                    else:
                        # print(f"Switch to Link {path_number+1}")
                        grooming_services.remove({
                            'odu_size': service['rate'],
                            'path': service['possible_paths'][path_number]
                        })
                else:
                    #print(f"Link {path_number} pass")
                    break
            grooming_lightpaths = wdm_count
        
        # Calculate the blocked ratio
        blocked_percentage = blocked_services / num_services
        
        # Calculate the number of light paths with grooming
        #grooming_lightpaths = network.calculate_wdm_count()
        
        # Recording Results
        results.append({
            'num_services': num_services,
            'no_grooming_lightpaths': no_grooming_lightpaths,
            'grooming_lightpaths': grooming_lightpaths,
            'blocked_percentage': blocked_percentage
        })
        
        # Output current result
        print(f"Number of lightpaths without grooming: {no_grooming_lightpaths}")
        print(f"Number of lightpaths with grooming: {grooming_lightpaths}")
        print(f"Save the number of lightpaths: {no_grooming_lightpaths - grooming_lightpaths}")
        print(f"Blocked service ratio: {blocked_percentage:.2%}")
        
        # Increase the number of services
        num_services += 10
    
    # Print the final result
    print("\nFinal simulation results:")
    for result in results:
        print(f"\nNumber of Services: {result['num_services']}")
        print(f"Number of lightpaths without grooming: {result['no_grooming_lightpaths']}")
        print(f"Number of lightpaths with grooming: {result['grooming_lightpaths']}")
        print(f"Savings ratio: {(result['no_grooming_lightpaths'] - result['grooming_lightpaths']) / result['no_grooming_lightpaths']:.2%}")
        print(f"Blocking rate: {result['blocked_percentage']:.2%}")
    # Plot
    plot_results(results)


if __name__ == "__main__":
    main()