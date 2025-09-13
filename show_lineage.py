import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from graphviz import Digraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lineages', type=str, help='Path to lineages json file.')
    args = parser.parse_args()
    
    def find_children(node, data):
        for k, v in data.items():
            if node in v and not k==node:
                yield k
        
    
    
    with open(args.lineages, 'r') as file:
        data = json.load(file)
        data = {str(k):[str(i) for i in v] for k,v in data.items()}
            

    # Create a Digraph object
    dot = Digraph(comment='Lineage Tree', format='pdf')


    # Function to recursively add nodes and edges to the graph
    def add_nodes(dot, parent, data):
        if parent not in dot:
            dot.node(str(parent))
        else:
            return
        children = list(find_children(parent, data))
        print(parent, children)
        for child in children:
            print(child)
            if child != '-1':  # Exclude the '-1' parent
                child_str = child
                if child_str not in dot:
                    dot.node(child_str)
                dot.edge(parent, child_str)
                add_nodes(dot, child, data)

    # Initialize the tree with the root node '-1'
    add_nodes(dot, '-1', data)

    # Save the dot file and render it as a PNG image
    dot.render('lineage_tree')

    print('DOT file and PNG image created successfully.')