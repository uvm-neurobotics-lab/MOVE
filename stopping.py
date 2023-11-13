import time
from typing import Any
import math

class StopCondition():
    def __init__(self,value: int = None) -> None:
        self.value = value
    
    def __call__(self, alg) -> bool:
        return False
    
class StopAfterGenerations(StopCondition):
    def __call__(self, alg) -> bool:
        return alg.gen >= self.value
    
class StopAfterBatches(StopCondition):
    def __call__(self, alg) -> bool:
        return alg.current_batch >= self.value
    
class StopAfterEvals(StopCondition):
    def __call__(self, alg) -> bool:
        return alg.record.n_evals_incl_sgd >= self.value
    
class StopAfterEvalsNoSGD(StopAfterEvals):
    def __call__(self, alg) -> bool:
        return alg.record.n_evals >= self.value
    
class StopAfterFwdCalls(StopCondition):
    def __call__(self, alg) -> bool:
        return alg.record.n_fwds_incl_sgd >= self.value
    
class StopAfterFwdCallsNoSGD(StopCondition):
    def __call__(self, alg) -> bool:
        return alg.record.n_fwds >= self.value
    
class StopAfterSeconds(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.seconds = value
        self.start_time = time.time()
        
    def __call__(self, alg) -> bool:
        return time.time() - self.start_time >= self.seconds
    
    
class StopAfterMeanFitness(StopCondition):
    def __init__(self, value: float):
        super().__init__()
        self.mean_fit = value
        
    def __call__(self, alg) -> bool:
        return alg.solution_fitness >= self.value
    

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
            return False
        else:
            new_fits = alg.fitnesses.max(dim=1)
            for i in range(len(self.last_fits)):
                if new_fits[i] > self.last_fits[i]:
                    self.stagnation[i] = 0
                else:
                    self.stagnation[i] += 1
            self.last_fits = new_fits
            return self.agg(self.stagnation) >= self.patience


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