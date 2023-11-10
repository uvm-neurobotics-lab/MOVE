import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to results directory")
    parser.add_argument('--agg', '-a', type=str, default=None, help="Aggregation function")
    parser.add_argument('--fitness', '-f', action='store_true', help="Plot fitness")
    parser.add_argument('--batch', '-b', type=int, default=None, help="Cut off batch")
    
    
    args = parser.parse_args()
    
    loaded = torch.load(args.path).to(torch.float32)
    
    if args.batch is not None:
        loaded = loaded[...,:args.batch]

    if args.fitness:
        # load csv for names
        import csv
        with open(os.path.join(os.path.dirname(args.path), "function_names.csv")) as f:
            reader = csv.reader(f)
            names = next(reader)
    
    if len(loaded.shape) > 2:
        loaded = torch.nanmean(loaded, dim=1).T
        
    if args.agg == 'cumsum':
        loaded = loaded.cumsum(dim=0)
    elif args.agg == 'mean':
        loaded = loaded.mean(dim=0)
        
    
    a = sns.lineplot(data=loaded, dashes=False)
    
    
    if args.fitness:
        # get legend
        handles, labels = a.get_legend_handles_labels()
        for i, l in enumerate(labels):
            labels[i] = names[i]
        a.legend(handles, labels)
    plt.show()
    plt.close()