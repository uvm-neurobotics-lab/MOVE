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
    
class StopAfterGenerations(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.gen
        return self.curr >= self.value
    
class StopAfterBatches(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.current_batch
        return self.curr >= self.value
    
class StopAfterEvals(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_evals_incl_sgd
        return self.curr >= self.value
    
class StopAfterEvalsNoSGD(StopAfterEvals):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_evals
        return self.curr >= self.value
    
class StopAfterFwdCalls(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_fwds_incl_sgd
        return  self.curr >= self.value
    
class StopAfterFwdCallsNoSGD(StopCondition):
    def __call__(self, alg) -> bool:
        self.curr = alg.record.n_fwds
        return self.curr >= self.value
    
class StopAfterSeconds(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.seconds = value
        self.start_time = time.time()
        
    def __call__(self, alg) -> bool:
        self.curr = time.time() - self.start_time
        return self.curr >= self.seconds
    
    
class StopAfterMeanFitness(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.mean_fit = value
        
    def __call__(self, alg) -> bool:
        self.curr = alg.solution_fitness
        return self.curr >= self.value
    

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


class StopAfterMinStagnation(StopCondition):
    def agg(self, stag):
        return min(stag)
        
    
    

name_to_stop_condition_map = {
    "none": StopCondition,
    None: StopCondition,
    "generations": StopAfterGenerations,
    "seconds": StopAfterSeconds,
    "mean_fitness": StopAfterMeanFitness,
    "max_stagnation": StopAfterMaxStagnation,
    "min_stagnation": StopAfterMinStagnation,
    "batches": StopAfterBatches,
    "evals": StopAfterEvals,
    "evals_no_sgd": StopAfterEvalsNoSGD,
    "fwds": StopAfterFwdCalls,
    "fwds_no_sgd": StopAfterFwdCallsNoSGD,
}