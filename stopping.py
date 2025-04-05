import time
from typing import Any
import math

class StopCondition():
    def __init__(self,value: int = None) -> None:
        self.curr = 0
        self.value = value
    
    def __call__(self, alg) -> bool:
        self.curr = -1
        return False
    
    def n_batches(self, alg) -> int:
        return 10_000 # impossible to guess
    
    
class StopAfterGenerations(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.gen
        return self.curr >= self.value
    
    def n_batches(self, alg) -> int:
        batches_per_gen = alg.config.pop_size / alg.config.batch_size
        return math.ceil(self.value * batches_per_gen)
    
    
class StopAfterBatches(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.current_batch
        return self.curr >= self.value
    def n_batches(self, alg) -> int:
        return self.value
    
    
class StopAfterEvals(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_evals_incl_sgd
        return self.curr >= self.value
    
    def n_batches(self, alg) -> int:
        print("value", self.value)
        print("batch size", alg.config.batch_size)
        # maximum number of batches is if there is (2 + 2*early_stopping) eval per cppn per batch
        # 2 for fwd+backward
        num_evals_per_cppn = 2 + (2*(alg.config.sgd_early_stop if alg.config.with_grad and alg.config.sgd_steps>0 else 0))
        return math.ceil(self.value / (num_evals_per_cppn * alg.config.batch_size))
    
    
class StopAfterEvalsNoSGD(StopAfterEvals):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_evals
        return self.curr >= self.value
    
    def n_batches(self, alg) -> int:
        # maximum number of batches is if there is one eval per cppn per batch
        return math.ceil(self.value / (alg.config.batch_size))
    
    
class StopAfterFwdCalls(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_fwds_incl_sgd
        return  self.curr >= self.value
    def n_batches(self, alg) -> int:
        # maximum number of batches 
        num_fwds_per_cppn = 0
        if alg.config.with_grad and alg.config.sgd_steps>0:
            num_fwds_per_cppn = 2 + (1*alg.config.sgd_early_stop)
        else:
            num_fwds_per_cppn = 1
        return math.ceil(self.value / (num_fwds_per_cppn * alg.config.batch_size))
    
    
class StopAfterFwdCallsNoSGD(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_fwds
        return self.curr >= self.value
    def n_batches(self, alg) -> int:
        # maximum number of batches is if there is one eval per cppn per batch
        return math.ceil(self.value / (alg.batch_size))
    
    
class StopAfterSeconds(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.seconds = value
        self.start_time = time.time()
        
    def __call__(self, alg) -> bool:
        self.curr = time.time() - self.start_time
        return self.curr >= self.seconds
    def n_batches(self, alg) -> int:
        return 10_000 # impossible to guess
    
    
class StopAfterMeanFitness(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.mean_fit = value
        
    def __call__(self, alg) -> bool:
        self.curr = alg.solution_fitness
        return self.curr >= self.value
    def n_batches(self, alg) -> int:
        return 10_000 # impossible to guess
    

class StopAfterMaxStagnation(StopCondition):
    def __init__(self, patience: int):
        super().__init__()
        self.patience = patience
        self.stagnation = []
        self.last_fits = None
    
    def agg(self, stag):
        return max(stag)
        
    def __call__(self, alg) -> bool:
        if self.last_fits is None:
            self.last_fits = alg.fitnesses.max(dim=1)
            self.stagnation = [0] * len(self.last_fits)
            self.curr = 0
            return False
        else:
            new_fits = alg.fitnesses.max(dim=1)
            for i in range(len(self.last_fits)):
                if new_fits[i] > self.last_fits[i]:
                    self.stagnation[i] = 0
                else:
                    self.stagnation[i] += 1
            self.last_fits = new_fits
            self.curr = self.agg(self.stagnation)
            return self.curr >= self.patience
    def n_batches(self, alg) -> int:
        return 10_000 # impossible to guess
    

class StopAfterMinStagnation(StopCondition):
    def agg(self, stag):
        return min(stag)
    def n_batches(self, alg) -> int:
        return 10_000 # impossible to guess
    
    

name_to_stop_condition_map = {
    "none": StopCondition,
    None: StopCondition,
    "generations": StopAfterGenerations,
    "seconds": StopAfterSeconds,
    "mean_fitness": StopAfterMeanFitness,
    "max_stagnation": StopAfterMaxStagnation,
    "min_stagnation": StopAfterMinStagnation,
    "batches": StopAfterBatches,
    "evaluations": StopAfterEvals,
    "evals": StopAfterEvals,
    "evals_no_sgd": StopAfterEvalsNoSGD,
    "fwds": StopAfterFwdCalls,
    "fwds_no_sgd": StopAfterFwdCallsNoSGD,
}