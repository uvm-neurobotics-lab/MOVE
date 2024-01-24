# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

from mnk_landscapes import *



#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

def sort_by_values_n_obj(list1, *values):
    # Check if the length of each values list matches the length of list1
    for v in values:
        if len(v) != len(list1):
            raise ValueError("Length of values list must match length of list1")

    # Create a custom key function that returns a tuple of values for each element in list1
    def sort_key(index):
        return tuple(v[index] for v in values)

    # Sort list1 based on the custom key
    sorted_indices = sorted(range(len(list1)), key=sort_key)

    # Return the sorted indices as per list1
    return [list1[i] for i in sorted_indices]





#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

def fast_non_dominated_sort_n_obj(*objectives):
    # Number of solutions
    num_solutions = len(objectives[0])
    
    # Initialize S, n, and rank
    S = [[] for _ in range(num_solutions)]
    n = [0 for _ in range(num_solutions)]
    rank = [0 for _ in range(num_solutions)]

    # Initialize the fronts
    front = [[]]

    # Function to check if one solution dominates another
    def dominates(solution_p, solution_q):
        return all(p >= q for p, q in zip(solution_p, solution_q)) and any(p > q for p, q in zip(solution_p, solution_q))

    # Iterate over all pairs of solutions
    for p in range(num_solutions):
        for q in range(num_solutions):
            solution_p = [objectives[i][p] for i in range(len(objectives))]
            solution_q = [objectives[i][q] for i in range(len(objectives))]

            if dominates(solution_p, solution_q):
                S[p].append(q)
            elif dominates(solution_q, solution_p):
                n[p] += 1

        # Check if p is a non-dominated solution
        if n[p] == 0:
            rank[p] = 0
            front[0].append(p)

    # Construct the subsequent fronts
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        front.append(Q)

    # Remove the last empty front
    del front[-1]

    return front


#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance


def crowding_distance_n_obj(front, *objectives):
    num_solutions = len(front)
    num_objectives = len(objectives)
    
    # Initialize distance for each solution in the front
    distance = [0 for _ in range(num_solutions)]

    # Helper function to sort the front according to an objective
    def sort_by_values(obj):
        return sorted(front, key=lambda x: obj[x])

    # Set the distance to infinity for boundary solutions
    distance[0] = float('inf')
    distance[-1] = float('inf')

    # Iterate over each objective
    for obj in objectives:
        # Sort the front based on the objective
        sorted_front = sort_by_values(obj)

        # Calculate the objective range
        obj_min = min(obj)
        obj_max = max(obj)
        obj_range = obj_max - obj_min if obj_max != obj_min else 1

        # Calculate crowding distance for each solution in the front
        for i in range(1, num_solutions - 1):
            distance[i] += (obj[sorted_front[i + 1]] - obj[sorted_front[i - 1]]) / obj_range

    return distance



#Function to carry out the crossover
def crossover(genome_a, genome_b, config):
    a,b=genome_a.genome,genome_b.genome
    c = MNKGenome(config, do_init=False)
    
    r=random.random()
    if r>0.5:
        c.genome = mutation((a+b)/2)
    else:
        c.genome = mutation((a-b)/2)
    return c

#Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution += min_x+(max_x-min_x)*random.random()
        solution = torch.clamp(solution, min_x, max_x)
    return solution



if __name__ == "__main__":
    import os
    from move import MOVE
    from run_setup import run_setup
    from tqdm import tqdm

    for config, verbose in run_setup(config_class=MNKConfig):
        
        # Initialize RMNK problem
        # rho = Objective correlation coefficient
        # m = Number of objective functions
        # n = Length of solutions
        # k = Number of epistatic links (non-linearity)


        print("Loading RMNK instance from", config.rmnk_instance_file)
        rmnk = Rmnk(instance_file=config.rmnk_instance_file)
        config.rmnk_r = rmnk.rho
        config.rmnk_m = rmnk.m
        config.rmnk_n = rmnk.n
        config.rmnk_k = rmnk.k
        
        print("RMNK:", config.rmnk_r, config.rmnk_m, config.rmnk_n, config.rmnk_k)


        config.mutation_sigma = 0.4
        config.mutation_percent = 0.05


        fitness_functions = []

        for i in range(config.rmnk_m):
            fn = lambda genomes, i=i, config=config, rmnk=rmnk: -1.0 * evaluate_genomes(genomes, rmnk, i, config) # invert sign to maximize
            fn.__name__ = f"Objective {i}"
            fitness_functions.append(fn)
            ff.register_fitness_function(fn.__name__, fn, True)
            ff.NO_NORM.append(fn)
            ff.NO_MEAN.append(fn)
            
        
        if not hasattr(config, 'output_dir'):
            config.output_dir = "../results/nsga-mnk"
        config.mutate_sgd_lr_sigma = False
        config.objective_functions = fitness_functions
        config.sgd_steps = 0
        config.genome_type = MNKGenome
        config.target=None

        config.experiment_condition = config.experiment_condition + " (NSGA-II)"
       
        # 50,000 total evaluations
            
        # Main program starts here
        pop_size = 20
        max_evals = 50_000
        max_gen = (max_evals / (2*pop_size)) # parents + offspring
        max_gen = int(max_gen / config.rmnk_m) # divide by number of objectives
        
        print("Max gen", max_gen)
        
        evals = 0
        
        evals_by_batch = torch.zeros(max_gen)
        normed_fitness_by_batch = torch.zeros(len(fitness_functions), pop_size, max_gen)

        
        #Initialization
        min_x=0
        max_x=1
        solution=[MNKGenome(config) for _ in range(pop_size)]
        gen_no=0
        
        
        cond_dir = os.path.join(config.output_dir, "conditions", config.experiment_condition)
        os.makedirs(cond_dir, exist_ok=True)
        run_dir = os.path.join(cond_dir, f"run_{config.run_id:04d}")
        os.makedirs(run_dir, exist_ok=True)
        
        front_function_values = torch.ones(len(fitness_functions), 1000, max_gen)*-torch.inf
        
        pbar = tqdm(total=max_gen, desc="NSGA-II")
        while(gen_no<max_gen):
            function_values = []
            for fn in fitness_functions:
                function_values.append(fn(solution))
                evals += len(solution)
                
            non_dominated_sorted_solution = fast_non_dominated_sort_n_obj(*function_values)
            crowding_distance_values=[]
            for i in range(0,len(non_dominated_sorted_solution)):
                crowding_distance_values.append(crowding_distance_n_obj(non_dominated_sorted_solution[i][:],*function_values))
            solution2 = solution[:]
            #Generating offsprings
            while(len(solution2)!=2*pop_size):
                a1 = random.randint(0,pop_size-1)
                b1 = random.randint(0,pop_size-1)
                solution2.append(crossover(solution[a1],solution[b1],config))
            
            function_values2 = []
            for fn in fitness_functions:
                function_values2.append(fn(solution2))
                evals += len(solution2)
                
            
            non_dominated_sorted_solution2 = fast_non_dominated_sort_n_obj(*function_values2)
            crowding_distance_values2=[]
            for i in range(0,len(non_dominated_sorted_solution2)):
                crowding_distance_values2.append(crowding_distance_n_obj(non_dominated_sorted_solution2[i][:],*function_values2))
            new_solution= []
            for i in range(0,len(non_dominated_sorted_solution2)):
                non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front22 = sort_by_values_n_obj(non_dominated_sorted_solution2_1[:], *crowding_distance_values2[i][:])
                front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
                front.reverse()
                # fill in front function values
                fn_2 = torch.stack(function_values2)
                for j, idx in enumerate(front):
                    front_function_values[:, j, gen_no] = fn_2[:, idx]
                for value in front:
                    new_solution.append(value)
                    if(len(new_solution)==pop_size):
                        break
                if (len(new_solution) == pop_size):
                    break
            avg_fitness = sum(function_values[0]) / len(function_values[0])
            pbar.set_description(f"NSGA-II (gen {gen_no}, avg fitness {avg_fitness:.3f})")
            solution = [solution2[i] for i in new_solution]
            gen_no = gen_no + 1
            pbar.update(1)
            evals_by_batch[gen_no-1] = evals
            
            
            # normed_fitness_by_batch[:,:,gen_no-1] = torch.tensor(function_values)
            
        # Lets plot the final front now
        if len(fitness_functions) == 2:
            function1_values = [i * -1 for i in function_values[0]]
            function2_values = [j * -1 for j in function_values[1]]
            function1 = [i * -1 for i in function1_values]
            function2 = [j * -1 for j in function2_values]
            plt.xlabel('Function 1', fontsize=15)
            plt.ylabel('Function 2', fontsize=15)
            plt.scatter(function1, function2)
            plt.savefig('nsga-ii.png')
            plt.show()
   
    # reshape to:  torch.ones((max_gens, 4), device='cpu')*-torch.inf
    evals_by_batch = evals_by_batch.unsqueeze(1).repeat(1, 4) # line up with MOVE data (gross TODO)
    # save as evals_by_batch.pt and normed_fitness_by_batch.pt
    import torch
    torch.save(evals_by_batch, os.path.join(run_dir, "evals_by_batch.pt"))
    torch.save(front_function_values, os.path.join(run_dir, "normed_fitness_by_batch.pt"))

    