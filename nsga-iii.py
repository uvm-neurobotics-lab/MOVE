import os
import time
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import dill

import sys

sys.path.append("../")

from move_config import MoveConfig
from fitness.fitness_functions import *

from problem import CPPNProblem, NoCrossover, CPPNSampling, CPPNMutation, CPPNRepair, CPPNCrossover

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nsga-iii.py <config_file>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    
    with open(json_path, 'r') as infile:
        json_str = infile.read()
        infile.close()
    
    import json
    c_json = json.loads(json_str)['controls']
    
    config = MoveConfig.create_from_json(c_json, config_type=MoveConfig)
    


    config.sgd_steps = 1


    
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # config.norm_df_path = '../data/target_fitness_fn_ranges.csv'

    problem = CPPNProblem(config)

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", len(problem.fns), n_partitions=12)

    
    # create the algorithm object

    algorithm = NSGA3(
                pop_size=config.num_cells, # for move comparison
                n_offsprings=1,
                ref_dirs=ref_dirs,
                sampling=CPPNSampling(),
                # crossover=CPPNCrossover(2, 1),
                crossover=NoCrossover(1, 1),
                mutation=CPPNMutation(),
                repair=CPPNRepair(),
                eliminate_duplicates=False
    
                )
    algorithm.setup(problem)
    algorithm.start_time = time.time()
    res = algorithm.result()
    
    
    
    if len (sys.argv) > 2:
        # trying loading as a checkpoint instead:
        
        ckpt = sys.argv[2]
        with open(ckpt, 'rb') as f:
            checkpoint = dill.load(f)
            print("Loaded Checkpoint:", checkpoint)
        
        
        # res = minimize(problem,
        #     checkpoint,
        #     seed=config.seed,
        #     termination=('n_gen', 1),
        #     verbose=True
        #     )

    else:
        
        



        # execute the optimization
        res = minimize(problem,
                    algorithm,
                    seed=config.seed,
                    termination=('n_gen', 10),
                    verbose=True
                    )

        with open("checkpoint", "wb") as f:
            dill.dump(algorithm, f)
            
        with open("result", "wb") as f:
            dill.dump(res, f)

    
    # Scatter().add(res.F).show()

    # get the images
    # for i, x in enumerate(res.X):

    os.makedirs("tmp_results", exist_ok=True)
    out = {}
    problem._evaluate(res.X, out, return_image=True)
    print(out)
    out = out["imgs"]
    out = np.clip(out, 0, 1)
    for i in range(len(out)):
        plt.figure()
        plt.imshow(out[i])
        plt.axis("off")
        plt.savefig(f"tmp_results/{i}.png")
        plt.show()