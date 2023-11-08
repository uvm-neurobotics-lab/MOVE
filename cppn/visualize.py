"""Visualize a CPPN network, primarily for debugging"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_nodes(graph, pos, node_labels, node_size):
    """Draw nodes on the graph"""

    shapes = set((node[1]["shape"] for node in graph.nodes(data=True)))
    for shape in shapes:
        nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, graph.nodes(data=True))]
        colors = [nx.get_node_attributes(graph, 'color')[
            cNode] for cNode in nodes]
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=colors,
                               label=node_labels, node_shape=shape, nodelist=nodes)


def add_edges_to_graph(individual, visualize_disabled, graph, pos, config):
    """Add edges to the graph
    Args:
        individual (CPPN): The CPPN to visualize
        visualize_disabled (bool): Whether to visualize disabled nodes
        graph (Graph): The graph to add the edges to
        pos (dict): The positions of the nodes

    Returns:
        edge_labels (dict): labels of edges
    """
    connections = individual.connections
    max_weight = config.max_weight
    edge_labels = {}

    for cx_key in connections:
        cx = connections[cx_key]
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight.detach().cpu().numpy(), 0))):
            continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled\
            else ('--', 'grey', .5 + abs(cx.weight.item())/max_weight)

        if cx.enabled and cx.weight < 0:
            style = ('-', 'r', .5+abs(cx.weight.item())/max_weight)

        from_node,to_node = cx_key.split(',')
        graph.add_edge(from_node, to_node,
                       weight=f"{cx.weight.item():.4f}", pos=pos, style=style)
        edge_labels[(from_node, to_node)] = f"{cx.weight.item():.3f}"

    return edge_labels


def draw_edges(individual, graph, pos, show_weights, node_size, edge_labels):
    """Draw edges on the graph"""

    edge_styles = set((s[2] for s in graph.edges(data='style')))
    for style in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == style, graph.edges(data='style'))]
        nx.draw_networkx_edges(graph, pos,
                               edgelist=edges,
                               arrowsize=25, arrows=True,
                               node_size=[node_size]*1000,
                               style=style[0],
                               edge_color=[style[1]]*1000,
                               width=style[2],
                               connectionstyle="arc3"
                               )
    if show_weights:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, label_pos=.75)


def add_input_nodes(individual, node_labels, graph):
    """add input nodes to the graph

    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    for i, node in enumerate(individual.input_nodes):
        graph.add_node(node.id, color='lightsteelblue',
                       shape='d', layer=0)
        if len(individual.input_nodes) == 4:
            # includes bias and distance node
            input_labels = ['y', 'x', 'd', 'b']
        elif len(individual.input_nodes) == 3:
            # includes bias node or distance node
            input_labels = ['y', 'x', 'b/d']
        else:
            input_labels = ['y', 'x']
            # add other inputs
            for i in range(len(individual.input_nodes) - 2):
                input_labels.append(f"i{i+1}")
            


        label = f"{node.layer}.{node.id}\n{input_labels[i]}:"
        if isinstance(node.activation, torch.nn.Module):
            name = node.activation.__class__.__name__
        else:
            name = node.activation.__name__.replace('_activation', '')
        label += f"\n{name}"

        node_labels[node.id] = label


def add_hidden_nodes(individual, node_labels, graph):
    """add input nodes to the graph

    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    for node in individual.hidden_nodes:
        graph.add_node(node.id, color='lightsteelblue',
                       shape='o', layer=int(node.layer))
        label = f"{node.layer}.{node.id}"
        if isinstance(node.activation, torch.nn.Module):
            label += f"\n{node.activation.__class__.__name__}"
        else:
            label += f"\n{node.activation.__name__.replace('_activation', '')}"
        node_labels[node.id] = label


def add_output_nodes(individual, node_labels, graph, config):
    """add input nodes to the graph
    Args:
        individual (CPPN): CPPN to visualize
        node_labels (dictionary): labels of nodes
        graph (Graph): graph to add nodes to
    """
    color_mode = config.color_mode

    for i, node in enumerate(individual.output_nodes):
        title = color_mode[i] if i < len(color_mode) else 'XXX'
        graph.add_node(node.id, color='lightsteelblue',
                       shape='s', layer=int(node.layer))
        label = f"{node.layer}.{node.id}\n{title}:"
        if isinstance(node.activation, torch.nn.Module):
            label += f"\n{node.activation.__class__.__name__}"
        else:
            label += f"\n{node.activation.__name__.replace('_activation', '')}"
        node_labels[node.id] = label


def add_nodes_to_graph(individual, node_labels, graph, config):
    """Add nodes to the graph"""
    add_input_nodes(individual, node_labels, graph)
    add_hidden_nodes(individual, node_labels, graph)
    add_output_nodes(individual, node_labels, graph, config)


def visualize_network(individual, config, visualize_disabled=False, show_weights=False, save_name=None):
    """Visualize a CPPN network"""
    node_labels = {}
    node_size = 2000
    graph = nx.DiGraph()

    # configure plot
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0, bottom=0, right=1.25,
                        top=1.25, wspace=0, hspace=0)

    # nodes:
    add_nodes_to_graph(individual, node_labels, graph, config)
    pos = nx.multipartite_layout(graph, scale=4, subset_key='layer')
    
    draw_nodes(graph, pos, node_labels, node_size)

    edge_labels = add_edges_to_graph(
        individual, visualize_disabled, graph, pos, config)

    draw_edges(individual, graph, pos, show_weights, node_size, edge_labels)

    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    plt.tight_layout()
    
    if save_name is not None:
        plt.savefig(save_name)
    
    plt.show()
