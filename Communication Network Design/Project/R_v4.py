import random
import networkx as nx
from collections import deque
from itertools import combinations
from OTH_v3 import Network  # 导入已有的 Network 类

max_path = 3
nbOfNode = 100

class TrafficSimulator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = [i for i in range(num_nodes)]  # 使用数字作为节点名
        self.graph = nx.Graph()
        self.path_cache = {}  # 缓存已经计算过的路径
        
    def create_network(self, topology_file=None, edge_probability=0.5):
        """
        创建网络拓扑，可以从文件读取或随机生成边并写入文件。
        
        :param topology_file: 拓扑文件路径，如果为None则随机生成边
        :param edge_probability: 随机生成边的概率（0到1之间的浮点数）
        """
        self.graph.add_nodes_from(self.nodes)
        
        if topology_file:
            # 从文件读取拓扑
            with open(topology_file, 'r') as file:
                for line in file:
                    node1, node2 = map(int, line.strip().split())
                    self.graph.add_edge(node1, node2)
        else:
            # 随机生成边
            edges = []
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if random.random() < edge_probability:  # 以一定概率生成边
                        edges.append((i, j))
            # 将边添加到图中
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
        """生成指定数量的随机服务"""
        services = []
        for _ in range(num_services):
            # 随机选择源节点和目的节点
            source, destination = random.sample(self.nodes, 2)
            # 随机选择服务速率（10G或100G）
            rate = random.choice(['10', '100'])
            
            # 为每个服务生成k条可能的路径
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
        """计算不使用grooming时需要的光路数量"""
        lightpaths = {}  # 记录每条链路上的光路数量
        
        for service in services:
            # 使用最短路径
            path = service['possible_paths'][0]
            
            # 对路径上的每条链路计算需要的光路数量
            for i in range(len(path) - 1):
                link = tuple(sorted([path[i], path[i+1]]))
                if link not in lightpaths:
                    lightpaths[link] = {
                        '10G': 0,
                        '100G': 0
                    }
                
                # 根据服务速率添加光路
                rate = service['rate']
                if rate == '10':
                    lightpaths[link]['10G'] += 1
                else:  # 100G
                    lightpaths[link]['100G'] += 1
        
        # 计算总光路数量（每500G容量算一条光路）
        total_lightpaths = 0
        for link_load in lightpaths.values():
            # 10G业务，每50个需要一条500G光路
            lightpath_10g = (link_load['10G'] + 49) // 50
            # 100G业务，每5个需要一条500G光路
            lightpath_100g = (link_load['100G'] + 4) // 5
            total_lightpaths += lightpath_10g + lightpath_100g
            
        return total_lightpaths, lightpaths

def main():
    # 创建仿真实例
    simulator = TrafficSimulator(nbOfNode)  # 网络的节点
    simulator.create_network()  # 创建网络拓扑
    
    # 初始化结果统计
    results = []
    num_services = 30  # 起始服务数量
    blocked_percentage = 0
    
    while blocked_percentage < 0.01:  # 直到0.01%的业务被阻塞
        print(f"\n测试服务数量: {num_services}")
        
        # 生成服务
        services = simulator.generate_services(num_services)
        
        # 计算无grooming情况
        no_grooming_lightpaths, link_details = simulator.calculate_no_grooming_lightpaths(services)
        


        # 处理服务
        grooming_services = []
        previce_services = []
        blocked_services = 0
        for service in services:
            for path_number in range(max_path):
                # 使用已有的Network类计算有grooming情况
                network = Network()
                
                # 添加节点和连接
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
                        #print(f"被阻塞的{grooming_services}")
                        grooming_services.remove({
                            'odu_size': service['rate'],
                            'path': service['possible_paths'][path_number]
                        })
                        #print(f"删除后的{grooming_services}")

                        blocked_services += 1
                        print("被阻塞+1")
                        path_number = 0
                        break
                    else:
                        print(f"切换到链路{path_number+1}")
                        grooming_services.remove({
                            'odu_size': service['rate'],
                            'path': service['possible_paths'][path_number]
                        })
                else:
                    #print(f"链路{path_number}通过")
                    break
            grooming_lightpaths = wdm_count
        
        # 计算被阻塞的比例
        blocked_percentage = blocked_services / num_services
        
        # 计算有grooming情况下的光路数量
        #grooming_lightpaths = network.calculate_wdm_count()
        
        # 记录结果
        results.append({
            'num_services': num_services,
            'no_grooming_lightpaths': no_grooming_lightpaths,
            'grooming_lightpaths': grooming_lightpaths,
            'blocked_percentage': blocked_percentage
        })
        
        # 输出当前结果
        print(f"无grooming光路数量: {no_grooming_lightpaths}")
        print(f"有grooming光路数量: {grooming_lightpaths}")
        print(f"节省光路数量: {no_grooming_lightpaths - grooming_lightpaths}")
        print(f"被阻塞服务比例: {blocked_percentage:.2%}")
        
        # 增加服务数量
        num_services += 10
    
    # 打印最终结果
    print("\n仿真最终结果:")
    for result in results:
        print(f"\n服务数量: {result['num_services']}")
        print(f"无grooming光路数量: {result['no_grooming_lightpaths']}")
        print(f"有grooming光路数量: {result['grooming_lightpaths']}")
        print(f"节省比例: {(result['no_grooming_lightpaths'] - result['grooming_lightpaths']) / result['no_grooming_lightpaths']:.2%}")
        print(f"阻塞率: {result['blocked_percentage']:.2%}")

if __name__ == "__main__":
    main()