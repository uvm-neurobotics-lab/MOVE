from typing import Callable

from cppn.activation_functions import ACTIVATION_FUNCTIONS
# from move.fitness.fitness_functions import FITNESS_FUNCTIONS
FITNESS_FUNCTIONS = {}

def name_to_fn(name):
    """
    Converts a string to a function.
    params:
        name: The name of the function.
    returns:
        The function.
    """
    # raise NotImplementedError("")
    if isinstance(name, (Callable,)) or name is None:
        return name
    assert isinstance(name, str), f"name must be a string but is {type(name)}"
    if name == "":
        return None
    all_fns = ACTIVATION_FUNCTIONS
    all_fns.update(FITNESS_FUNCTIONS)
    if not name in all_fns:
        raise ValueError(f"Function {name} not found. Use register_activation_function or register_fitness_function to register it.")
    return all_fns[name]
    
    fns = inspect.getmembers(sys.modules[af.__name__])
    fns.extend(inspect.getmembers(sys.modules[ff.__name__]))
    
    fns.extend([("round", lambda x: torch.round(x))])
    
    if name == "Conv2d":
        return torch.nn.Conv2d
    
    try:
        return fns[[f[0] for f in fns].index(name)][1]
    except ValueError:
        raise ValueError(f"Function {name} not found.")