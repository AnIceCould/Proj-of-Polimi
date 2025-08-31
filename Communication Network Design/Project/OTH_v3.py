class OTN1:
    def __init__(self):
        self.odu_10_in = 0  # 从外界接收的10G ODU数量
        self.odu_100_in = 0  # 从外界接收的100G ODU数量
        self.odu_10_out = 0  # 发送到外界的10G ODU数量
        self.odu_100_out = 0  # 发送到外界的100G ODU数量
        self.odu_10_forward_send = 0  # 转发到OTN2的10G ODU数量
        self.odu_100_forward_send = 0  # 转发到OTN2的100G ODU数量
        self.odu_10_forward_receive = 0  # 从OTN2接收的10G ODU数量
        self.odu_100_forward_receive = 0  # 从OTN2接收的100G ODU数量

    def receive_odu(self, odu_type, count):
        """从外界接收ODU，根据类型更新计数器"""
        if odu_type == '10':
            self.odu_10_in += count
        elif odu_type == '100':
            self.odu_100_in += count

    def send_odu(self, odu_type, count):
        """发送ODU到外界，根据类型更新计数器"""
        if odu_type == '10':
            self.odu_10_out += count
        elif odu_type == '100':
            self.odu_100_out += count

    def forward_odu_to_otn2(self):
        """自动根据从外界接收的ODU数量更新转发到OTN2的ODU数量"""
        self.odu_10_forward_send = self.odu_10_in
        self.odu_100_forward_send = self.odu_100_in

    def forward_odu_from_otn2(self):
        """自动根据发送到外界的ODU数量更新从OTN2接收的ODU数量"""
        self.odu_10_forward_receive = self.odu_10_out
        self.odu_100_forward_receive = self.odu_100_out

    def calculate_io_cards(self):
        """计算所需的I/O卡数量，合并接收/发送和转发发送/接收的ODU"""
        io_cards = 0

        # 计算从外界接收和发送到外界的ODU所需的I/O卡
        total_in_out_10 = self.odu_10_in + self.odu_10_out
        total_in_out_100 = self.odu_100_in + self.odu_100_out
        io_cards += (total_in_out_10 + 9) // 10  # 每10个10G ODU消耗一个I/O卡
        io_cards += total_in_out_100  # 每个100G ODU消耗一个I/O卡

        # 计算转发到OTN2和从OTN2接收的ODU所需的I/O卡
        total_forward_10 = self.odu_10_forward_send + self.odu_10_forward_receive
        total_forward_100 = self.odu_100_forward_send + self.odu_100_forward_receive
        io_cards += (total_forward_10 + 9) // 10  # 每10个10G ODU消耗一个I/O卡
        io_cards += total_forward_100  # 每个100G ODU消耗一个I/O卡

        return io_cards

    def calculate_capacity(self):
        """计算容量消耗，每个I/O卡消耗100Gb/s"""
        return self.calculate_io_cards() * 100

class OTN2:
    def __init__(self, node_name):
        self.node_name = node_name  # 节点名称
        # OTN2与OTN1之间的ODU交换
        self.odu_10_otn1 = 0  # 与OTN1交换的10G ODU数量
        self.odu_100_otn1 = 0  # 与OTN1交换的100G ODU数量

        # OTN2与物理层之间的ODU交换（与外界交换）
        self.odu_10_physical_in = 0  # 从外界接收的10G ODU数量
        self.odu_100_physical_in = 0  # 从外界接收的100G ODU数量
        self.odu_10_physical_out = 0  # 发送到外界的10G ODU数量
        self.odu_100_physical_out = 0  # 发送到外界的100G ODU数量

        # OTN2与其他节点之间的ODU交换（光路交换）
        self.node_exchanges = {}  # 记录与其他节点的ODU交换，格式为 {节点: {'10_in': x, '100_in': y, '10_out': z, '100_out': w}}

    def exchange_with_otn1(self):
        """与OTN1交换ODU，自动根据与外界交换的ODU数量更新"""
        # 从OTN1到OTN2的ODU数量等于从外界接收的ODU数量
        self.odu_10_otn1 = self.odu_10_physical_in
        self.odu_100_otn1 = self.odu_100_physical_in

        # 从OTN2到OTN1的ODU数量等于发送到外界的ODU数量
        self.odu_10_otn1 += self.odu_10_physical_out
        self.odu_100_otn1 += self.odu_100_physical_out

    def exchange_with_physical(self, odu_type, count, direction):
        """与外界交换ODU，更新计数器"""
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
        """与其他节点交换ODU，更新计数器"""
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
        """计算OTN2所需的I/O卡数量"""
        io_cards = 0

        # 计算与OTN1交换的I/O卡
        io_cards_otn1 = (self.odu_10_otn1 + 9) // 10 + self.odu_100_otn1
        io_cards += io_cards_otn1

        # 计算与外界交换的I/O卡
        total_physical_10 = self.odu_10_physical_in + self.odu_10_physical_out
        total_physical_100 = self.odu_100_physical_in + self.odu_100_physical_out
        io_cards_physical = (total_physical_10 + 9) // 10 + total_physical_100
        io_cards += io_cards_physical

        # 计算与其他节点交换的I/O卡
        for node, exchanges in self.node_exchanges.items():
            total_node_10 = exchanges['10_in'] + exchanges['10_out']
            total_node_100 = exchanges['100_in'] + exchanges['100_out']
            io_cards_node = (total_node_10 + 9) // 10 + total_node_100
            io_cards += io_cards_node

        return io_cards

    def calculate_physical_connections(self):
        """计算光路数量"""
        # 计算与外界交换的光路数量
        total_physical_bandwidth = (self.odu_10_physical_in + self.odu_10_physical_out) * 10 + \
                                   (self.odu_100_physical_in + self.odu_100_physical_out) * 100
        physical_connections = (total_physical_bandwidth + 499) // 500

        # 计算与其他节点交换的光路数量
        for node, exchanges in self.node_exchanges.items():
            total_node_bandwidth = (exchanges['10_in'] + exchanges['10_out']) * 10 + \
                                   (exchanges['100_in'] + exchanges['100_out']) * 100
            physical_connections += (total_node_bandwidth + 499) // 500

        return physical_connections

    def calculate_capacity(self):
        """计算OTN2的总容量消耗"""
        # 与OTN1交换的容量消耗
        capacity_otn1 = self.calculate_io_cards_otn1() * 100

        # 光路的容量消耗
        capacity_physical = self.calculate_physical_connections() * 500

        return capacity_otn1 + capacity_physical

    def calculate_io_cards_otn1(self):
        """计算与OTN1交换的I/O卡数量"""
        return (self.odu_10_otn1 + 9) // 10 + self.odu_100_otn1
    
    def nb_of_odu(self):
        """ 计算与OTN2交换的ODU数量 """
        nbOfOdu = self.odu_10_physical_in + self.odu_100_physical_in + self.odu_10_physical_out + self.odu_100_physical_out

        return nbOfOdu

class Network:
    def __init__(self):
        self.nodes_otn1 = {}  # 存储所有节点的OTN1实例
        self.nodes_otn2 = {}  # 存储所有节点的OTN2实例
        self.connections = set()  # 存储节点之间的连接关系，格式为 {(节点1, 节点2)}

    def add_node(self, node_name):
        """添加节点到网络中"""
        if node_name not in self.nodes_otn1:
            self.nodes_otn1[node_name] = OTN1()
        if node_name not in self.nodes_otn2:
            self.nodes_otn2[node_name] = OTN2(node_name)

    def add_connection(self, node1, node2):
        """添加节点之间的连接"""
        if node1 not in self.nodes_otn1:
            self.add_node(node1)
        if node2 not in self.nodes_otn1:
            self.add_node(node2)
        self.connections.add((node1, node2))
        self.connections.add((node2, node1))  # 双向连接

    def process_services(self, services):
        """处理服务列表，更新节点的ODU交换信息"""
        for service in services:
            odu_size = service['odu_size']  # ODU大小（10G或100G）
            path = service['path']          # 服务的路径（节点列表）

            # 遍历路径中的每个节点
            for i in range(len(path)):
                node_name = path[i]

                # 如果是路径的起点（起始节点）
                if i == 0:
                    # 从外界接收ODU
                    self.nodes_otn1[node_name].receive_odu(odu_size, 1)
                    self.nodes_otn2[node_name].exchange_with_physical(odu_size, 1, 'in')

                # 如果是路径的终点（结束节点）
                if i == len(path) - 1:
                    # 发送ODU到外界
                    self.nodes_otn1[node_name].send_odu(odu_size, 1)
                    self.nodes_otn2[node_name].exchange_with_physical(odu_size, 1, 'out')

                # 如果不是路径的最后一个节点，需要转发ODU到下一个节点
                if i < len(path) - 1:
                    next_node_name = path[i + 1]
                    # 发送ODU到下一个节点
                    self.nodes_otn2[node_name].exchange_with_node(odu_size, 1, 'out', next_node_name)

    def propagate_odu_exchanges(self):
        """自动从相连节点获取ODU"""
        for node_name, otn2 in self.nodes_otn2.items():
            for target_node, exchanges in otn2.node_exchanges.items():
                if '10_out' in exchanges:
                    self.nodes_otn2[target_node].exchange_with_node('10', exchanges['10_out'], 'in', node_name)
                if '100_out' in exchanges:
                    self.nodes_otn2[target_node].exchange_with_node('100', exchanges['100_out'], 'in', node_name)

    def calculate_wdm_count(self):
        """计算整个网络使用的WDM数量"""
        wdm_count = 0
        # 遍历所有连接
        for connection in self.connections:
            node1, node2 = connection
            if node1 < node2:  # 避免重复计算双向连接
                # 计算节点1到节点2的总带宽
                bandwidth_1_to_2 = (self.nodes_otn2[node1].node_exchanges.get(node2, {'10_out': 0, '100_out': 0})['10_out'] * 10 + 
                                   self.nodes_otn2[node1].node_exchanges.get(node2, {'10_out': 0, '100_out': 0})['100_out'] * 100)
                # 计算节点2到节点1的总带宽
                bandwidth_2_to_1 = (self.nodes_otn2[node2].node_exchanges.get(node1, {'10_out': 0, '100_out': 0})['10_out'] * 10 + 
                                   self.nodes_otn2[node2].node_exchanges.get(node1, {'10_out': 0, '100_out': 0})['100_out'] * 100)
                # 总带宽
                total_bandwidth = bandwidth_1_to_2 + bandwidth_2_to_1
                # 计算WDM数量
                wdm_count += (total_bandwidth + 499) // 500
        return wdm_count

    def calculate_total_io_cards(self):
        """计算整个网络中OTN1和OTN2的I/O卡总数"""
        total_io_cards = sum(otn1.calculate_io_cards() for otn1 in self.nodes_otn1.values())
        total_io_cards += sum(otn2.calculate_io_cards() for otn2 in self.nodes_otn2.values())
        return total_io_cards

    def calculate_total_capacity(self):
        """计算整个网络的总容量消耗"""
        total_capacity = sum(otn1.calculate_capacity() for otn1 in self.nodes_otn1.values())
        total_capacity += sum(otn2.calculate_capacity() for otn2 in self.nodes_otn2.values())
        return total_capacity

    def calculate_otn2_non_otn1_odu(self):
        """计算 OTN2 交换的 ODU 大小（不包括与 OTN1 交换的 ODU）"""
        total_odu_10 = 0
        total_odu_100 = 0
        for otn2 in self.nodes_otn2.values():
            for node, exchanges in otn2.node_exchanges.items():
                total_odu_10 += exchanges['10_in'] + exchanges['10_out']
                total_odu_100 += exchanges['100_in'] + exchanges['100_out']
        return total_odu_10 + total_odu_100

    def run_network(self, services):
        """运行该网络"""
        # 处理服务列表
        #print(f"将要执行的路径{services}")
        self.process_services(services)
        # 自动从相连节点获取ODU --OTN2
        self.propagate_odu_exchanges()

        # 更新OTN1与OTN2交换的ODU数量
        for otn1 in self.nodes_otn1.values():
            otn1.forward_odu_to_otn2()
            otn1.forward_odu_from_otn2()
        for otn2 in self.nodes_otn2.values():
            otn2.exchange_with_otn1()
        can_use = 1
        # 计算每个节点的I/O卡数量和容量消耗
        for node_name, otn1 in self.nodes_otn1.items():
            if(otn1.calculate_io_cards() > 70):
                print(f"节点{node_name}I/O 卡总数不满足要求")
                can_use = 0
            if(otn1.calculate_capacity() > 12288):
                print(f"节点{node_name} OTN1 总容量不满足要求")
                can_use = 0
            #print(f"节点{node_name} OTN1 I/O卡数量: {otn1.calculate_io_cards()}, 容量消耗: {otn1.calculate_capacity()} Gb/s")

        for node_name, otn2 in self.nodes_otn2.items():
            if(otn2.calculate_io_cards() > 70):
                print(f"节点{node_name} OTN2 I/O 卡总数不满足要求")
                can_use = 0
            if(otn2.calculate_capacity() > 12288):
                print(f"节点{node_name} OTN2 总容量不满足要求")
                can_use = 0
            if((otn2.nb_of_odu()) > 100):
                print(f"节点{node_name}ODU不满足要求")
                can_use = 0
            #print(f"节点{node_name} OTN2 I/O卡数量: {otn2.calculate_io_cards()}, 光路数量: {otn2.calculate_physical_connections()}, 容量消耗: {otn2.calculate_capacity()} Gb/s")
        
        # total_io_cards = self.calculate_total_io_cards()
        # total_capacity = self.calculate_total_capacity()
        # otn2_odu = self.calculate_otn2_non_otn1_odu()
        
        # 计算总的 I/O 卡数量和容量
        # print(f"节点的 I/O 卡总数: {total_io_cards}")
        # print(f"节点的总容量消耗: {total_capacity} Gb/s")
        # print(f"节点交换的 ODU 数量: {otn2_odu}")

        # 计算整个网络使用的WDM数量
        wdm_count = self.calculate_wdm_count()
        #print(f"整个网络使用的WDM数量: {wdm_count}")
        return can_use, wdm_count

'''
# 示例测试
network = Network()

# 添加节点
network.add_node('A')
network.add_node('B')
network.add_node('C')

# 添加节点之间的连接
network.add_connection('A', 'B')
network.add_connection('B', 'C')

# 服务列表
services = [
    {'odu_size': '10', 'path': ['A', 'B', 'C']},  # Service1：10Gb/s，由A到B到C
    {'odu_size': '100', 'path': ['A', 'B']},       # Service2：100Gb/s，由A到B
    {'odu_size': '100', 'path': ['B', 'C']}        # Service3：100Gb/s，由B到C
]

(can_use, wdm_count) = network.run_network(services)
print(f"整个网络使用的WDM数量: {wdm_count}")
if(can_use == 1):
    print("可以使用")
else:
    print("不能使用")
'''