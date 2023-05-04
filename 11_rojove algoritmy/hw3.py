# %%
from collections import namedtuple
import math
import functools
import numpy as np
import csv
import pprint
import sys
import argparse

import matplotlib.pyplot as plt
from matplotlib import collections as mc

Vertex = namedtuple('Vertex', ['name', 'x', 'y', 'demand'])


@functools.lru_cache(maxsize=None)
def distance(v1, v2):
    return ((v1.x - v2.x)**2+(v1.y - v2.y)**2)**(1/2)


def fitness(vertices, distance, solution):
    solution_distance = 0
    for x, y in zip(solution, solution[1:]):
        solution_distance += distance(vertices[x], vertices[y])
    solution_distance += distance(vertices[solution[-1]], vertices[solution[0]])
    return solution_distance


def initialize_pheromone(N):
    return 0.01*np.ones(shape=(N,N))

def update_pheromone(pheromones_array, solutions, fits, Q=100, rho=0.6):
    pheromone_update = np.zeros(shape=pheromones_array.shape)
    for solution, fit in zip(solutions, fits):
        for x, y in zip(solution, solution[1:]):
            pheromone_update[x][y] += Q/fit
        pheromone_update[solution[-1]][solution[0]] += Q/fit
    
    return (1-rho)*pheromones_array + pheromone_update


def generate_solutions_VRP(vertices: list[Vertex.demand], pheromones, distance, N, capacity, alpha=1, beta=3):
    def compute_prob(v1, v2, capacity):
        if vertices[v2].demand > capacity or v1 == v2 or vertices[v2].demand == 0:
            return 1e-12
        dist = 1/distance(vertices[v1], vertices[v2])
        tau = pheromones[v1, v2]
        ret = pow(tau, alpha) * pow(dist,beta)
        return ret if ret > 1e-6 else 1e-6

    pheromones_shape = pheromones.shape[0]
    for i in range(N):
        available = list(range(1,pheromones_shape))
        solution = [0]
        cap = capacity
        while available:
            probs = np.array([compute_prob(solution[-1], x, cap) for x in [0]+available])
            selected = np.random.choice([0]+available, p=probs/sum(probs))
            if vertices[selected].demand == float("inf") or vertices[selected].demand > cap:
                solution.append(0)
                cap = capacity
                continue
            cap -= vertices[selected].demand
            solution.append(selected)
            available.remove(selected)
        yield solution

import xml.etree.ElementTree as ET
def parse_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = root.find("network").find("nodes")
    vertices = []
    for vertex in nodes:
        type = vertex.attrib['type']
        x = float(vertex.find('cx').text)
        y = float(vertex.find('cy').text)
        if type == '0':
            V = Vertex(name=vertex.attrib['id'], x=x,y=y,demand=(float("inf")))
            vertices = [V] + vertices
        else:
            V = Vertex(name=vertex.attrib['id'], x=x,y=y,demand=0.0)
            vertices.append(V)

    capacity = float(root.find("fleet").find("vehicle_profile").find("capacity").text)
    for req in root.find("requests"):
        quantity = float(req.find("quantity").text)
        ix_to_be_removed = list(map(lambda x: x.name == req.attrib['node'], vertices)).index(True)
        vertices[ix_to_be_removed] = Vertex(name=vertices[ix_to_be_removed].name, 
                                                          x=vertices[ix_to_be_removed].x,
                                                          y=vertices[ix_to_be_removed].y,
                                                          demand=quantity)
        # pprint.pprint(vertices[ix_to_be_removed])
    return vertices, capacity


def ant_solver(vertices, distance, capacity, ants=10, max_iterations=3000, alpha=1, beta=3, Q=100, rho=0.8):
    pheromones = initialize_pheromone(len(vertices))
    best_solution = None
    best_fitness = float('inf')
    best_sol_progress = []
    
    for i in range(max_iterations):
        solutions = list(generate_solutions_VRP(vertices, pheromones, distance, ants, capacity,
                                                 alpha=alpha, beta=beta))
        fits = [fitness(vertices, distance, s) for s in solutions]
        pheromones = update_pheromone(pheromones, solutions, fits, Q=Q, rho=rho)
        
        for s, f in zip(solutions, fits):
            if f < best_fitness:
                best_fitness = f
                best_solution = s
        
        best_sol_progress.append(best_fitness)
        
        # print(f'{i:4}, {np.min(fits):.4f}, {np.mean(fits):.4f}, {np.max(fits):.4f}')
        print(f"iteration {i} / {max_iterations}, best fitness: {best_fitness}", end="\r")
    return best_solution, pheromones, best_sol_progress




parser = argparse.ArgumentParser()
parser.add_argument("--size", help="size of the problem", default=32,)
parser.add_argument("--ants", help="number of ants", default=10, type=int)
parser.add_argument("--iterations", help="number of iterations", default=100, type=int)
parser.add_argument("--alpha", help="alpha", default=1, type=int)
parser.add_argument("--beta", help="beta", default=3, type=int)
parser.add_argument("--Q", help="Q", default=100)
parser.add_argument("--rho", help="rho", default=0.8, type=float)
args = parser.parse_args()

def main():

    path = f"domaci_ukol_data/data_{args.size}.xml"
    vertices, capacity = parse_xml(path)
    best_sol, pheromones, bs_progress = ant_solver(vertices, distance, capacity, ants=args.ants, 
                        max_iterations=args.iterations, alpha=args.alpha, beta=args.beta, Q=args.Q, rho=args.rho)

    lines = []
    colors = []
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            lines.append([(v1.x, v1.y), (v2.x, v2.y)])
            colors.append(pheromones[i][j])

    lc = mc.LineCollection(lines, linewidths=np.array(colors))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()

    solution = best_sol

    # best_sol_progress plot
    # plt.figure(figsize=(12, 8))
    # plt.plot(bs_progress)


    # tady muzeme zkouset vliv jednotlivych parametru na vygenerovane reseni
    # solution = list(generate_solutions(vertices, pheromones, distance, N=1, alpha=3, beta=1))[0]

    print('Fitness: ', fitness(vertices, distance, solution))

    solution_vertices = [vertices[i] for i in solution]
    pprint.pprint(solution_vertices)

    solution_lines = []
    for i, j in zip(solution, solution[1:]):
        solution_lines.append([(vertices[i].x, vertices[i].y), (vertices[j].x, vertices[j].y)])
    solution_lines.append([(vertices[solution[-1]].x, vertices[solution[-1]].y), (vertices[solution[0]].x, vertices[solution[0]].y)])
    solutions_lc = mc.LineCollection(solution_lines, colors='red')
    ax.add_collection(solutions_lc)

    plt.show()


if __name__ == "__main__":
    main()

