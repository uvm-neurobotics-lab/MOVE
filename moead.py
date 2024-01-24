from moead_framework.aggregation import Tchebycheff   
from moead_framework.algorithm.combinatorial import Moead   
from moead_framework.problem.combinatorial import Rmnk  
from moead_framework.tool.result import save_population
import os
 
output_dir = "../results/moead/"

instance_files = [
    "data/rmnk_-0.7_10_100_5_0.dat",
    "data/rmnk_0_10_100_0_0.dat",
    "data/rmnk_0_10_100_1_0.dat",
    "data/rmnk_0_10_100_2_0.dat",
    "data/rmnk_0_10_100_5_0.dat",
    "data/rmnk_0.7_10_100_5_0.dat",
]

for instance_file in instance_files:
    this_output_dir = os.path.join(output_dir, instance_file.split("/")[-1].split(".")[0])
    if not os.path.exists(this_output_dir):
        os.makedirs(this_output_dir)

    ###############################
    #   Initialize the problem    #
    ###############################
    # The file is available here : https://github.com/moead-framework/data/blob/master/problem/RMNK/Instances/rmnk_0_2_100_1_0.dat
    # Others instances are available here : https://github.com/moead-framework/data/tree/master/problem/RMNK/Instances
    rmnk = Rmnk(instance_file=instance_file)
        
        
    #####################################
    #      Initialize the algorithm     #
    #####################################
    number_of_weight = 10
    number_of_weight_neighborhood = 2
    number_of_evaluations = 1000
    # The file is available here : https://github.com/moead-framework/data/blob/master/weights/SOBOL-2objs-10wei.ws
    # Others weights files are available here : https://github.com/moead-framework/data/tree/master/weights
    weight_file = "SOBOL-" + str(rmnk.number_of_objective) + "objs-" + str(number_of_weight) + "wei.ws"
        
        
    ###############################
    #    Execute the algorithm    #
    ###############################
    moead = Moead(problem=rmnk,
                    max_evaluation=number_of_evaluations,
                    number_of_weight_neighborhood=number_of_weight_neighborhood,
                    weight_file=weight_file,
                    aggregation_function=Tchebycheff,
                    )
        
    population = moead.run()
        
        
    ###############################
    #       Save the result       #
    ###############################
    save_file = "moead-rmnk" + str(rmnk.number_of_objective) \
                    + "-N" + str(number_of_weight) \
                    + "-T" + str(number_of_weight_neighborhood) \
                    + "-iter" + str(number_of_evaluations) \
                    + ".txt"
    
    save_file = os.path.join(this_output_dir, save_file)
    
    save_population(save_file, population)