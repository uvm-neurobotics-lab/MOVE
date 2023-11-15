import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype is deprecated")
warnings.filterwarnings("ignore", "use_inf_as_na option is deprecated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to results directory")
    parser.add_argument('--agg', '-a', type=str, default=None, help="Aggregation function")
    parser.add_argument('--agg-value', '-v', type=int, nargs='*', default=None, help="Aggregation function value")
    parser.add_argument('--agg-dim', '-d', type=int, nargs='*', default=[0], help="Aggregation function dimension")
    parser.add_argument('--fitness', '-f', action='store_true', help="Plot fitness")
    parser.add_argument('--replacements', '-r', action='store_true', help="Plot replacements")
    parser.add_argument('--batch', '-b', type=int, default=None, help="Cut off batch")
    
    
    args = parser.parse_args()
    
    if len(args.agg_dim) == 1:
        args.agg_dim = args.agg_dim[0]
    if len(args.agg_value) == 1:
        args.agg_value = args.agg_value[0]
    
    loaded = torch.load(args.path).to(torch.float32)
    
    if args.batch is not None:
        loaded = loaded[...,:args.batch]

    if args.fitness:
        # load csv for names
        import csv
        with open(os.path.join(os.path.dirname(args.path), "function_names.csv")) as f:
            reader = csv.reader(f)
            names = next(reader)
    
    if args.replacements:
        # drop all in last dimension where there is a nan
        shape = loaded.shape
        loaded = loaded.reshape(-1, shape[-1]) # 64x64, 1000
        #Drop all rows containing any nan:
        loaded = loaded[~torch.isnan(loaded).any(dim=1)]
        # find the last row with sum>0
        last = (loaded.sum(dim=1) > 0).nonzero()[-1]
        # drop all rows after that
        loaded = loaded[:,:last+1]
        #Reshape back:
        loaded = loaded.reshape((shape[0], shape[1], -1))

        
        loaded = loaded.sum(dim=0).sum(dim=0)
    
    if len(loaded.shape) > 2:
        if args.agg_dim == 0:
            print("Data has shape", loaded.shape, "you may want to specify --agg-dim")
        loaded = torch.nanmean(loaded, dim=args.agg_dim).T
        
    if args.agg == 'cumsum':
        loaded = loaded.cumsum(dim=args.agg_dim)
    elif args.agg == 'mean':
        loaded = loaded.mean(dim=args.agg_dim if args.agg_value is None else args.agg_value)
    elif args.agg == 'rolling' or (args.agg == 'mean' and len(loaded.shape) < 2):
        orig_shape = loaded.shape
        win_size = 10 if args.agg_value is None else args.agg_value
        # pad with nans if not divisible by win_size
        pad = win_size - (loaded.shape[args.agg_dim] % win_size)
        if pad != win_size:
            pad = (0, pad) if args.agg_dim == 0 else (pad, 0)
            loaded = torch.nn.functional.pad(loaded, pad, value=float('nan'))
        
        for dim in range(3-len(loaded.shape)):
            loaded = loaded.unsqueeze(0)
        loaded = loaded.reshape(loaded.shape[0], -1, win_size)
        loaded = loaded.nanmean(dim=args.agg_dim+2)
        # restore original shape
        for dim in range(3-len(orig_shape)):
            loaded = loaded.squeeze(0)
    
    a = sns.lineplot(data=loaded, dashes=False)
    
    
    if args.fitness:
        # get legend
        handles, labels = a.get_legend_handles_labels()
        for i, l in enumerate(labels):
            labels[i] = names[i]
        a.legend(handles, labels)
    plt.show()
    plt.close()