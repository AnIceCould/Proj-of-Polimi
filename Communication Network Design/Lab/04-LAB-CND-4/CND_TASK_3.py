import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from itertools import islice
from typing import List, Tuple
from ortools.linear_solver import pywraplp

#Function get_network_rwa_json(fp: string, n_wavelengths: integer) -> Graph:
def get_network_rwa_json(fp: str, n_wavelengths: int) -> nx.Graph:
    #Open the file at path fp in read mode as file
    with open(fp, "r") as file:
        #Load JSON data from file into data
        data = json.load(file)
    #Convert data into a graph G using node-link format
    G = nx.node_link_graph(data, edges="links")
    #Initialize an empty dictionary available_wavelengths
    available_wavelengths = {}
    #For each edge in G:
    for edge in G.edges():
        #Set available_wavelengths[edge] to an array of ones with length n_wavelengths and data type uint8
        available_wavelengths[edge] = np.ones((n_wavelengths,), dtype=np.uint8)
    #Set the "available_wavelengths" attribute for all edges in G to the available_wavelengths dictionary
    nx.set_edge_attributes(G, available_wavelengths, "available_wavelengths")
    #Set the graph attribute "n_wavelengths" in G to n_wavelengths
    G.graph["n_wavelengths"] = n_wavelengths
    #Return the graph G
    return G

#Function get_ksp(G: Graph, n_paths: integer, metric: string) -> Dictionary with tuple keys and list values:
def get_ksp(G: nx.Graph, n_paths: int, metric: str) -> dict[tuple[int, int]: list[int]]:
    #Initialize an empty dictionary ksp
    ksp = {}
    #For each node i in the graph G:
    for i in range(G.number_of_nodes()):
        #For each node j in the graph G:
        for j in range(G.number_of_nodes()):
            #If i is less than j:
            if i < j:
                #Find the first n_paths shortest paths between nodes i and j in G using the specified metric
                #Convert these paths to a list and store in paths
                paths = list(islice(nx.shortest_simple_paths(G, i, j, metric), n_paths))
                #Set ksp[(i, j)] to paths
                ksp[i, j] = paths
                #Set ksp[(j, i)] to paths (to store paths in both directions)
                ksp[j, i] = paths
    #Return the dictionary ksp
    return ksp

#Function sap_ff_rwa(G: Graph, demands: List of tuples, ksp: Dictionary) -> Tuple of integer and List:
def sap_ff_rwa(G: nx.Graph, demands: list, ksp: dict) -> tuple[int, list]:
    #Initialize n_routed_demands to 0
    n_routed_demands = 0
    #Initialize an empty list routed_demands
    routed_demands = []
    #For each source-destination pair (src, dst) in demands:
    for src, dst in demands:
        #For each path in ksp[(src, dst)]:
        for path in ksp[src, dst]:
            #For each wavelength wav from 0 to the number of wavelengths in G:
            for wav in range(G.graph["n_wavelengths"]):
                #Set is_wavelength_free to True
                is_wavelength_free = True
                #For each consecutive pair of nodes in the path:
                for i in range(len(path)-1):
                    #If the available wavelength for the edge between these nodes is not free (value is 0):
                    if G[path[i]][path[i+1]]["available_wavelengths"][wav] == 0:
                        #Set is_wavelength_free to False
                        is_wavelength_free = False
                        #Break out of the edge-checking loop
                        break
                #If is_wavelength_free is True:
                if is_wavelength_free:
                    #Break out of the wavelength loop
                    break

            #If is_wavelength_free is True:
            if is_wavelength_free:
                #For each consecutive pair of nodes in the path:
                for i in range(len(path)-1):
                    #Assert that the wavelength wav on the edge between these nodes is free (value is 1)
                    assert(G[path[i]][path[i+1]]["available_wavelengths"][wav] == 1)
                    #Set the wavelength wav on this edge to unavailable (value is 0)
                    G[path[i]][path[i+1]]["available_wavelengths"][wav] = 0
                #Increment n_routed_demands by 1
                n_routed_demands += 1
                #Append the tuple (src, dst, path, wav) to routed_demands
                routed_demands.append((src, dst, path, wav))
                #Break out of the path loop
                break

    #Return the tuple (n_routed_demands, routed_demands)
    return n_routed_demands, routed_demands

### ILP
def ILP(G: nx.Graph, demands: list, n_wavelengths: int) -> int:
    #G = get_network_rwa_json("./nsfnet.json", n_wavelengths = n_wavelengths)
    # Set of nodes
    V = list(G.nodes)
    # Set of edges
    E_u = list(G.edges())
    E = []
    for e in E_u:
        E.append(e)
        E.append((e[1], e[0]))
    # E = E_u

    # set of demands
    K = list(range(len(demands)))
    # source node and destination node of the demands
    s = {}
    t = {}
    for k in K:
        s[k] = demands[k][0]
        t[k] = demands[k][1]
    # set of wavelengths
    Lambda = list(range(n_wavelengths))

    # Prepare parameters of the ILP

    # Create the SCIP solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not found.")

    # Variable definition
    # f[i,λ,(u,v)] represents flow variable f_{i,λ}(u,v)
    f = {}
    for k in K:
        for l in Lambda:
            for e in E:
                f[k,l,e] = solver.IntVar(0, 1, f'f_{k}_{l}_{e}')

    # flow conservation constraints on intermediate ndoes
    for k in K:
        for l in Lambda:
            for v in V:
                if v != s[k] and v != t[k]:  # For transit nodes
                    solver.Add(
                        sum(f[k,l,(u,v)] for u in V if (u,v) in E) -
                        sum(f[k,l,(v,w)] for w in V if (v,w) in E) == 0
                    )

    # flow conservation constraints on source and destination nodes
    for k in K:
        for l in Lambda:
            solver.Add(
                sum(f[k,l,(s[k],w)] for w in V if (s[k],w) in E) -
                sum(f[k,l,(w,s[k])] for w in V if (w,s[k]) in E) ==
                sum(f[k,l,(w,t[k])] for w in V if (w,t[k]) in E) -
                sum(f[k,l,(t[k],w)] for w in V if (t[k],w) in E)
            )

    # Ensure each demand is assigned one path
    for k in K:
        solver.Add(
            sum(f[k,l,(s[k],w)] for w in V if (s[k],w) in E for l in Lambda) -
            sum(f[k,l,(w,s[k])] for w in V if (w,s[k]) in E for l in Lambda) == 1
        )

    # each demand can only be assigned to one wavelength
    for k in K:
        for e in E:
            solver.Add(
                sum(f[k,l,e] for l in Lambda) <= 1
            )

    # different demands cannot use the same wavelength on the same link
    for l in Lambda:
        for e in E:
            solver.Add(
                sum(f[k,l,e] for k in K) <= 1
            )

    # Objective function: Minimize wavelength consumption
    objective = solver.Objective()
    for k in K:
        for l in Lambda:
            for e in E:
                objective.SetCoefficient(f[k,l,e], 1)
    objective.SetMinimization()

    # print the solution status and objective function
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        ilp_result = solver.Objective().Value()
        print('Objective value =', ilp_result)
        
        # Print the path and wavelength assignment for each demand
        for k in K:
            print(f'\nDemand {k} (from {s[k]} to {t[k]}):')
            for l in Lambda:
                path = []
                for e in E:
                    if f[k,l,e].solution_value() > 0.5:  # Using 0.5 as threshold due to potential numerical issues
                        path.append(e)
                if path:
                    print(f'  Wavelength {l}: {path}')
    else:
        print('The problem does not have an optimal solution.')
    return ilp_result
    ## ilp_result ##

### heuristic
def heuristic(G: nx.Graph, demands: list, n_wavelengths: int) -> int:

    num_spectrum = 0

    #Load the graph G from the JSON file "./nsfnet.json" with n_wavelengths as the number of wavelengths per edge
    G = get_network_rwa_json("./nsfnet.json", n_wavelengths = n_wavelengths)
    #Generate a list of demands using the function generate_demands with G and a total of 12 demands
    #demands = generate_demands(G, 12)
    #Generate the k-shortest paths (ksp) dictionary for G with up to 5 paths per node pair, using 'length' as the metric
    ksp = get_ksp(G, 5, metric='length')

    #Print the number of demands in the demands list
    print(f'Number of demands: {len(demands)}')
    #Call the sap_ff_rwa function with G, demands, and ksp, and store the results in n_routed_demands and routed_demands
    n_routed_demands, routed_demands = sap_ff_rwa(G, demands, ksp)
    #Print the number of routed demands
    print(f'Number of routed demands: {n_routed_demands}')

    #For each demand in routed_demands:
    for demand in routed_demands:
        #Extract the route path from the demand
        route = demand[-2]
        #Create an edge list route_el from consecutive node pairs in the route path
        route_el = list(zip(route, route[1:]))
        #Add the length of route_el to num_spectrum
        num_spectrum += len(route_el)

    #Print "Spectrum occupation is: " followed by the value of num_spectrum
    print("Spectrum occupation is: " + str(num_spectrum))
    return num_spectrum
    ## num_spectrum ##

#Set variable
n_wavelengths = 3
#Load the graph G from the JSON file
G = get_network_rwa_json("./nsfnet.json", n_wavelengths = n_wavelengths)

#Small demand
demands_TM1 = [
    (0, 13), (2, 11), (4, 9),
    (6, 7), (1, 8)
]
#Medium quantity demand
demands_TM2 = [
    (0, 13), (1, 12), (2, 11),
    (3, 10), (4, 9), (5, 8),
    (6, 7), (0, 7), (7, 13)
]
#High load
demands_TM3 = [
    (0, 13), (1, 12), (2, 11),
    (0, 7), (1, 7), (2, 7),
    (7, 13), (7, 12), (7, 11),
    (0, 1), (1, 2), (2, 3)
]
#Centralized
demands_TM4 = [
    (0, 7), (1, 7), (2, 7), (3, 7),
    (7, 10), (7, 11), (7, 12), (7, 13)
]
#Chain
demands_TM5 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10)
]

### run
ilp_result_1 = ILP(G,demands_TM1,n_wavelengths)
heuristic_result_1 = heuristic(G,demands_TM1,n_wavelengths)
ilp_result_2 = ILP(G,demands_TM2,n_wavelengths)
heuristic_result_2 = heuristic(G,demands_TM2,n_wavelengths)
ilp_result_3 = ILP(G,demands_TM3,n_wavelengths)
heuristic_result_3 = heuristic(G,demands_TM3,n_wavelengths)
ilp_result_4 = ILP(G,demands_TM4,n_wavelengths)
heuristic_result_4 = heuristic(G,demands_TM4,n_wavelengths)
ilp_result_5 = ILP(G,demands_TM5,n_wavelengths)
heuristic_result_5 = heuristic(G,demands_TM5,n_wavelengths)

### Plot

# set data
ilp_results = [ilp_result_1, ilp_result_2, ilp_result_3, ilp_result_4, ilp_result_5]
heuristic_results = [heuristic_result_1, heuristic_result_2, heuristic_result_3, heuristic_result_4, heuristic_result_5]
# set plot
bar_width = 0.35
index = np.arange(len(ilp_results))
# barplots
fig, ax = plt.subplots()
bar1 = ax.bar(index, ilp_results, bar_width, label='ILP')
bar2 = ax.bar(index + bar_width, heuristic_results, bar_width, label='Heuristic')
# label and title
ax.set_xlabel('traffic matrixes')
ax.set_ylabel('objetive function')
ax.set_title('Comparison of ILP vs Heuristic Solutions')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['TM1', 'TM2', 'TM3', 'TM4', 'TM5'])
ax.legend()
# show
plt.show()