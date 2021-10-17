import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import tqdm
import time
from datetime import datetime
import copy
import random


filename = "tsp"
data = pd.read_csv(f'{filename}.txt', header = None, sep=",", names=["x", 'y'])
points = np.array(data)
num_evolution = 100000
num_tries = 100
num_population = 10


def merge(dict1, dict2):
    """merge two dicts"""
    res = {**dict1, **dict2}
    return res


def euclidean_distance(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    dist = np.linalg.norm(p1 - p2)
    return dist


def sum_all_pts(pts): 
    res = 0
    for i in range(1, len(pts)):
        dist = euclidean_distance(pts[i], pts[i-1])
        res += dist
    return res


def shuffle_sum(pts):
    res = 0
    shuffled_pts = copy.deepcopy(pts)
    np.random.shuffle(shuffled_pts)
    res = sum_all_pts(shuffled_pts)
    return [res, shuffled_pts]


def mutation(pts):
    new_pts = copy.deepcopy(pts)
    idx1 = np.random.randint(0, (len(new_pts)-1)//2)
    idx2 = np.random.randint(((len(new_pts)-1)//2)+1, len(new_pts)-1)
#     new_pts[[idx1, idx2]] = new_pts[[idx2, idx1]]
    tmp = new_pts[idx2]
    new_pts[idx2] = new_pts[idx1]
    new_pts[idx1] = tmp
    
    return new_pts

def mutate_dict(dict_, mutate_rate):
    copied_dict_ = copy.deepcopy(dict_)
    # {dist1 : [p1,p2,p3], dist2 : [p2,p3,p1]}
    times = mutate_rate * 100
    dists = []
    pts = []
    d = {}
    
    for ele in dict_:
        mutated_pts = mutation(dict_[ele])
        times -= 1
        dist = sum_all_pts(mutated_pts)
        dists.append(dist)
        pts.append(mutated_pts)
        if times < 0:
            break
    for i in range(len(dists)):
        d[dists[i]] = pts[i]
    
    merged_dicts = merge(d, dict_)
    return merged_dicts

def population_generation(start_pop: int, pts: list) -> dict:
    pool = {}
    
    for i in range(start_pop):
        sf = shuffle_sum(pts)
        pool[sf[0]] = sf[1]
    return pool


def rs(nums, pts):
    dists = []
    generation = 0
    evaluation = []
    f_pts = []
    for i in tqdm.tqdm(range(nums)):
        r = shuffle_sum(pts)
        if len(dists) == 0:
            dists.append(r[0])
            generation += 1
            evaluation.append(i)
            f_pts = r[1]

#         if  r[0] > dists[-1]:   # longest: >
        if  r[0] < dists[-1]: # shortest: <
            dists.append((r[0]))
            generation += 1
            evaluation.append(i)
            f_pts = r[1]

    return [evaluation, dists, f_pts]    


def rmhc(evaluation, pts):
    """
    random mutation hill climbing
    """
    
    first_evaluation = shuffle_sum(pts)
    dist = first_evaluation[0]
    pt = first_evaluation[1]
    
    res_f_pts = []
    res_eva = []
    res_dists = []
    

    
    for i in tqdm.tqdm(range(evaluation)):
        
        if (len(res_dists) == 0):
            res_dists.append(dist)
            res_eva.append(i)
            res_f_pts = pt
        
        # mutation - swap two cities
        new_pt = mutation(pt)
        new_dist = sum_all_pts(new_pt)
        
        if new_dist < dist: # shortest: <
#         if new_dist > dist: # longest: >
            dist = new_dist
            pt =  new_pt

            res_eva.append(i)
            res_f_pts = pt
            res_dists.append(dist)
        
    return [res_eva, res_dists, res_f_pts]


def rrhc(evaluation, pts, num_tries):
    
    res_f_pts = []
    res_eva = []
    res_dists = []
    counter = evaluation
    
    
    while counter > 0:
        
        init_eval = shuffle_sum(pts)
        dist = init_eval[0]
        pt = init_eval[1]

        counter_tries = num_tries
        counter -= 1
        
        if (len(res_dists) == 0):
            res_dists.append(dist)
            res_eva.append(evaluation - counter)
            res_f_pts = pt
            
        while counter_tries > 0 and counter > 0:
            
            new_pt = mutation(pt)
            new_dist = sum_all_pts(new_pt)
            print(counter)
            counter -= 1
            
            
            if new_dist < dist:
                dist = new_dist
                pt = new_pt

            else:
                counter_tries -= 1
                
        if (dist < res_dists[-1]):   # shortest: <
#         if (dist > res_dists[-1]):     # longest: >

            res_dists.append(dist)
            res_eva.append(evaluation - counter)
            res_f_pts = pt
        
    return [res_eva, res_dists, res_f_pts]


def pbs(evaluation, pts, start_pop):
    """
    parallel beam search
    generate start_pop population to start
    ...
    """
    
    res_f_pts = []
    res_eva = []
    res_dists = []
    
    pool = {}
    counter = evaluation

    
    # start poplulation
    pool = population_generation(start_pop, pts)
    """
    {dist1 : [p1,p2,p3], dist2 : [p2,p3,p1]}
    """
    num_keys = len(pool)
        
    while counter > 0 and num_keys >= 2:
        
        # mutate one and combine
        mutates = {}
        for ele in (pool):
            new_pts = mutation(pool[ele])
            mutates[sum_all_pts(new_pts)] = new_pts
            
        merge_pool = merge(mutates, pool)
        pool = merge_pool
        
        # rank 
        temp_pool = {}

#         keys = sorted(pool.keys(), reverse=True) # longest
        keys = sorted(pool.keys(), reverse=False) # shortest
        
        half_keys = keys[0:len(keys)//2]
        num_keys = len(half_keys)
        print(num_keys)
        # give pool's value to temp_pool
        
        for i in range(num_keys): 
            temp_pool[keys[i]] = pool[keys[i]]
            
        res_dist = min(half_keys) # min for shortest            
#         res_dist = max(half_keys) # max for longest


        if (len(res_dists) == 0) or (res_dist < res_dists[-1]): # shortest: <
#         if (len(res_dists) == 0) or (res_dist > res_dists[-1]): # longest: >

            res_f_pts = temp_pool[res_dist]
            res_eva.append(evaluation - counter)
            res_dists.append(res_dist)
             
        pool = temp_pool
        counter -= 1
        print(counter)
        
    return [res_eva, res_dists, res_f_pts]

def crossover(p1, p2):
    
    parent1 = [list(x) for x in p1]
    parent2 = [list(x) for x in p1]
    child1 = []
    child2 = []
    childP1 = []
    childP2 = []
    
    a = int(random.random() * len(parent1))
    b = int(random.random() * len(parent1))
    
    start = min(a, b)
    end = max(a, b)

    for i in range(start, end):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child1 = childP1 + childP2
    child2 = childP2 + childP1
    
    return [child1, child2]


def ga_no_m(evaluation, pts, start_pop):
    
    """
    genetic algorithm
    
    do:
    1. generate population
    2. crossover
    3. mutation (no)
    4. selection
    
    while time
    """
    res_f_pts = []
    res_eva = []
    res_dists = []
    
    pool = {}
    counter = evaluation

    
    # start poplulation
    pool = population_generation(start_pop, pts) # {dist1 : [p1,p2,p3], dist2 : [p2,p3,p1]}
    
    num_keys = len(pool)
    
    while counter > 0 and num_keys >= 2:
        
        # crossover one and combine
        children = {}
        
        keys = sorted(pool.keys(), reverse=False) # shortest
#         keys = sorted(pool.keys(), reverse=True) # longest
        
        for i in range(0, len(keys)-1, 2):
            new_pts = crossover(pool[keys[i]], pool[keys[i+1]])
            children[sum_all_pts(new_pts[0])] = new_pts[0] ## first child
            children[sum_all_pts(new_pts[1])] = new_pts[1] ## second child

        merge_pool = merge(children, pool)
        pool = merge_pool
        
        # rank 
        temp_pool = {}

#         keys = sorted(pool.keys(), reverse=True) # longest
        keys = sorted(pool.keys(), reverse=False) # shortest
        
        half_keys = keys[0:(len(keys)//2)]
        num_keys = len(half_keys)
        print(num_keys)
        
        # give pool's value to temp_pool
        
        for i in range(num_keys): 
            temp_pool[keys[i]] = pool[keys[i]]
            
        res_dist = min(half_keys) # min for shortest            
#         res_dist = max(half_keys) # max for longest


        if (len(res_dists) == 0) or (res_dist < res_dists[-1]): # shortest: <
#         if (len(res_dists) == 0) or (res_dist > res_dists[-1]): # longest: >

            res_f_pts = temp_pool[res_dist]
            res_eva.append(evaluation - counter)
            res_dists.append(res_dist)
             
        pool = temp_pool
        counter -= 1

        print(f"counter: {counter}")
        
    return [res_eva, res_dists, res_f_pts]



def ga_m(evaluation, pts, start_pop):
    
    """
    genetic algorithm
    
    do:
    1. generate population
    2. crossover
    3. mutation (yes)
    4. selection
    
    while time
    """
    res_f_pts = []
    res_eva = []
    res_dists = []
    
    pool = {}
    counter = evaluation

    
    # start poplulation
    pool = population_generation(start_pop, pts) # {dist1 : [p1,p2,p3], dist2 : [p2,p3,p1]}
    
    num_keys = len(pool)
    
    while counter > 0 and num_keys >= 2:
        
        # crossover one and combine
        children_pool = {}
        
        keys = sorted(pool.keys(), reverse=False) # shortest
#         keys = sorted(pool.keys(), reverse=True) # longest
        
        for i in range(0, len(keys)-1, 2):
            new_pts = crossover(pool[keys[i]], pool[keys[i+1]])
            children_pool[sum_all_pts(new_pts[0])] = new_pts[0] ## first child
            children_pool[sum_all_pts(new_pts[1])] = new_pts[1] ## second child

        merge_pool = merge(children_pool, pool)

        mutated_pool = mutate_dict(merge_pool, 0.01)
        
        pool = mutated_pool
        
        # rank 
        temp_pool = {}

#         keys = sorted(pool.keys(), reverse=True) # longest
        keys = sorted(pool.keys(), reverse=False) # shortest
        
        half_keys = keys[0:(len(keys)//2)]
        num_keys = len(half_keys)
        print(num_keys)
        
        # give pool's value to temp_pool
        for i in range(num_keys): 
            temp_pool[keys[i]] = pool[keys[i]]
            
        res_dist = min(half_keys) # min for shortest            
#         res_dist = max(half_keys) # max for longest


        if (len(res_dists) == 0) or (res_dist < res_dists[-1]): # shortest: <
#         if (len(res_dists) == 0) or (res_dist > res_dists[-1]): # longest: >

            res_f_pts = temp_pool[res_dist]
    
            x,y = zip(*res_f_pts)
            plt.figure(figsize=(10,10))
            plt.gca().set_aspect('equal')
            plt.plot(x,y)
            plt.savefig(f'/home/dx2222/ea_homework1/foo_{evaluation - counter}.png', 
                        bbox_inches='tight')
            plt.close()
            res_eva.append(evaluation - counter)
            res_dists.append(res_dist)
             
        pool = temp_pool
        counter -= 1

        print(f"counter: {counter}")
        
    return [res_eva, res_dists, res_f_pts]



def ga_tru(evaluation, pts, start_pop, mutation_rate):

   
    """
    ################# cut off rate 33% with 0.5 mutation rate #################
    genetic algorithm
    
    do:
    1. generate population
    2. crossover
    3. mutation (yes)
    4. selection
    
    while time
    """
    res_f_pts = []
    res_eva = []
    res_dists = []
    
    pool = {}
    counter = evaluation

    
    # start poplulation
    pool = population_generation(start_pop, pts) # {dist1 : [p1,p2,p3], dist2 : [p2,p3,p1]}
    
    num_keys = len(pool)
    
    while counter > 0 and num_keys >= 2:
        
        # crossover one and combine
        children_pool = {}
        
        keys = sorted(pool.keys(), reverse=False) # shortest
#         keys = sorted(pool.keys(), reverse=True) # longest
        
        for i in range(0, len(keys)-1, 2):
            new_pts = crossover(pool[keys[i]], pool[keys[i+1]])
            children_pool[sum_all_pts(new_pts[0])] = new_pts[0] ## first child
            children_pool[sum_all_pts(new_pts[1])] = new_pts[1] ## second child

        merge_pool = merge(children_pool, pool)

        mutated_pool = mutate_dict(merge_pool, mutation_rate)
        
        pool = mutated_pool
        
        # rank 
        temp_pool = {}

#         keys = sorted(pool.keys(), reverse=True) # longest
        keys = sorted(pool.keys(), reverse=False) # shortest
        
        half_keys = keys[0:(len(keys)//3)]
        num_keys = len(half_keys)
        print(num_keys)
        
        # give pool's value to temp_pool
        for i in range(num_keys): 
            temp_pool[keys[i]] = pool[keys[i]]
            
        res_dist = min(half_keys) # min for shortest            
#         res_dist = max(half_keys) # max for longest


        if (len(res_dists) == 0) or (res_dist < res_dists[-1]): # shortest: <
#         if (len(res_dists) == 0) or (res_dist > res_dists[-1]): # longest: >
        
            res_f_pts = temp_pool[res_dist]
            res_eva.append(evaluation - counter)
            res_dists.append(res_dist)
             
        pool = temp_pool
        counter -= 1

        print(f"counter: {counter}")
        
    return [res_eva, res_dists, res_f_pts]




def main():
	ga_m_res = ga_m(num_evolution, points, num_population)


if __name__ == "__main__":
	main()