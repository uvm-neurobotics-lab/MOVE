import logging
import torch 

def is_valid_connection(nodes, connections, key:tuple, config, warn:bool=False):
    """
    Checks if a connection is valid.
    params:
        from_node: The node from which the connection originates.
        to_node: The node to which the connection connects.
        config: The settings to check against
    returns:
        True if the connection is valid, False otherwise.
    """
    from_node, to_node = key
    from_node, to_node = nodes[from_node], nodes[to_node]
    
    # if from_node.layer == to_node.layer:
    #     if warn:
    #         logging.warning(f"Connection from node {from_node.id} (layer:{from_node.layer}) to node {to_node.id} (layer:{to_node.layer}) is invalid because they are on the same layer.")
    #     return False  # don't allow two nodes on the same layer to connect

    if not config.allow_recurrent and creates_cycle(connections, key):
        if warn:
            logging.warning(f"Connection from node {from_node.id} (layer:{from_node.layer}) to node {to_node.id} (layer:{to_node.layer}) is invalid because it creates a cycle.")
        return False
    # if not config.allow_recurrent and from_node.layer > to_node.layer:
    #     if warn:
    #         logging.warning(f"Connection from node {from_node.id} (layer:{from_node.layer}) to node {to_node.id} (layer:{to_node.layer}) is invalid because it is recurrent.")
    #     return False  # invalid

    return True


def get_ids_from_individual(individual):
    """Gets the ids from a given individual

    Args:
        individual (CPPN): The individual to get the ids from.

    Returns:
        tuple: (inputs, outputs, connections) the ids of the CPPN's nodes
    """
    
    inputs = [node.id for node in individual.input_nodes]
    outputs = [node.id for node in individual.output_nodes]
    connections = [(conn_key.split(',')[0], conn_key.split(',')[1]) for conn_key in individual.enabled_connections]
    return inputs, outputs, connections


def get_disjoint_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that do not share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if\
        (t_cx.key not in other_innovation and t_cx.key < other_innovation[-1])]


def get_excess_connections(this_cxs, other_innovation):
    """returns connections in this_cxs that share an innovation number with a
        connection in other_innovation"""
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if\
            (t_cx.key not in other_innovation and t_cx.key > other_innovation[-1])]


def get_matching_connections(cxs_1, cxs_2):
    """returns connections in cxs_1 that share an innovation number with a connection in cxs_2
       and     connections in cxs_2 that share an innovation number with a connection in cxs_1"""

    return sorted([c1 for c1 in cxs_1 if c1.key in [c2.key for c2 in cxs_2]],
                    key=lambda x: x.key),\
                    sorted([c2 for c2 in cxs_2 if c2.key in [c1.key for c1 in cxs_1]],
                    key=lambda x: x.key)



def genetic_difference(cppn, other) -> float:
    # only enabled connections, sorted by innovation id
    this_cxs = sorted(cppn.enabled_connections,
                        key=lambda c: c.key)
    other_cxs = sorted(other.enabled_connections,
                        key=lambda c: c.key)

    N = max(len(this_cxs), len(other_cxs))
    other_innovation = [c.key for c in other_cxs]

    # number of excess connections
    n_excess = len(get_excess_connections(this_cxs, other_innovation))
    # number of disjoint connections
    n_disjoint = len(get_disjoint_connections(this_cxs, other_innovation))

    # matching connections
    this_matching, other_matching = get_matching_connections(
        this_cxs, other_cxs)
    
    difference_of_matching_weights = [
        abs(o_cx.weight.item()-t_cx.weight.item()) for o_cx, t_cx in zip(other_matching, this_matching)]
    # difference_of_matching_weights = torch.stack(difference_of_matching_weights)
    
    if(len(difference_of_matching_weights) == 0):
        difference_of_matching_weights = 0
    else:
        difference_of_matching_weights = sum(difference_of_matching_weights) / len(difference_of_matching_weights)

    # Furthermore, the compatibility distance function
    # includes an additional argument that counts how many
    # activation functions differ between the two individuals
    n_different_fns = 0
    for t_node, o_node in zip(cppn.node_genome.values(), other.node_genome.values()):
        if(t_node.activation.__name__ != o_node.activation.__name__):
            n_different_fns += 1

    # can normalize by size of network (from Ken's paper)
    if(N > 0):
        n_excess /= N
        n_disjoint /= N

    # weight (values from Ken)
    n_excess *= 1
    n_disjoint *= 1
    difference_of_matching_weights *= .4
    n_different_fns *= 1
    
    difference = sum([n_excess,
                        n_disjoint,
                        difference_of_matching_weights,
                        n_different_fns])
    if torch.isnan(torch.tensor(difference)):
        difference = 0

    return difference


# Functions below are modified from other packages

###############################################################################################
# Functions below are from the NEAT-Python package https://github.com/CodeReclaimers/neat-python/

# LICENSE:
# Copyright (c) 2007-2011, cesar.gomes and mirrorballu2
# Copyright (c) 2015-2019, CodeReclaimers, LLC
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################################


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False

def get_candidate_nodes(s, connections):
    """Find candidate nodes c for the next layer.  These nodes should connect
    a node in s to a node not in s."""
    return set(b for (a, b) in connections if a in s and b not in s)


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    From: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """

    required = set(outputs) # outputs always required
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.

    Modified from: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    
    dangling_inputs = set()
    for n in required:
        has_input = False
        for (a, b) in connections:
            if b == n:
                has_input = True
                break
    
    # add dangling inputs to the input set
    s = s.union(dangling_inputs)
    
    while 1:

        c = get_candidate_nodes(s, connections)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            entire_input_set_in_s = all(a in s for (a, b) in connections if b == n)
            if n in required and entire_input_set_in_s:
                t.add(n)
        # t = set(a for (a, b) in connections if b in s and a not in s)
        if not t:
            break

        layers.append(t)
        s = s.union(t)
    return layers

