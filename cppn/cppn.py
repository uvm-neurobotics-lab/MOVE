"""Contains the CPPN, Node, and Connection classes."""
from itertools import count
import torch
from torch import nn
from torchviz import make_dot
from cppn.util import *
from cppn.graph_util import *
import cppn.activation_functions as af


class Node(nn.Module):
    def __init__(self, activation, id, bias=0.0):
        super().__init__()
        self.set_activation(activation)
        # self.bias = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.id:str = id
        self.layer = 999
    
    def set_activation(self, activation):
        
        if isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation
        
    def forward(self, x):
        # return self.activation(x + self.bias)
        return self.activation(x + self.bias)
    
    
class Connection(nn.Module):
    def __init__(self, weight, enabled=True):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.enabled = enabled
        
    def forward(self, x):
        return x * self.weight

    def add_to_weight(self, delta):
        self.weight = nn.Parameter(self.weight + delta)

class CPPN(nn.Module):
    """A CPPN Object with Nodes and Connections."""
    
    current_id = 1 # 0 reserved for 'random' parent
    current_node_id = 0
    
    @staticmethod
    def get_id():
        __class__.current_id += 1
        return __class__.current_id - 1
    
    @staticmethod
    def get_new_node_id():
        """Returns a new node id."""
        __class__.current_node_id += 1
        new_id = str(__class__.current_node_id-1)
        return new_id
    
    
    # TODO: remove deprecated
    @property
    def n_outputs(self):
        return self.n_output
    @property
    def n_in_nodes(self):
        return len(self.input_nodes)
    @property
    def node_genome(self):
        return {node_id: self.nodes[node_id].activation for node_id in self.nodes}
    @property
    def connection_genome(self):
        return {conn_key: self.connections[conn_key].weight for conn_key in self.connections}

    ###   
    
    
    def __init__(self, config, do_init=True):
        super().__init__()
        self.nodes = nn.ModuleDict()  # key: node_id (string)
        self.connections = nn.ModuleDict()  # key: (from, to) (string)
        self.input_nodes = nn.ModuleList()
        self.output_nodes = nn.ModuleList()
        self.hidden_nodes = nn.ModuleList()
        
        self.n_input = config.num_inputs
        self.n_output = config.num_outputs
        
        self.sgd_lr = config.sgd_learning_rate
        self.parents = (-1, -1)
        self.age = 0
        self.lineage = []
        self.cell_lineage = []
        self.n_cells = 0
        self.device = config.device
        self.id = -1
        
        if do_init:
            self.id = type(self).get_id()
            hidden_layers = self.initialize_node_genome(config)
            self.initialize_connection_genome(hidden_layers,
                                              config.init_connection_probability,
                                              config.init_connection_probability_fourier,
                                              config.weight_init_std)
            
            self.update_layers()
        
        self.to(self.device)
 
    
    def initialize_node_genome(self, config):
            n_hidden = config.hidden_nodes_at_start
            if isinstance(n_hidden, int):
                n_hidden = (n_hidden,)
            self.input_nodes = nn.ModuleList([Node(af.IdentityActivation,
                                                   str(n_i)) for n_i in range(-1, -self.n_input - 1, -1)])
            self.output_nodes = nn.ModuleList([Node(af.IdentityActivation,
                                                    str(n_i)) for n_i in range(-self.n_input - 1, -self.n_input - self.n_output - 1, -1)])
            
            for node in self.input_nodes + self.output_nodes:
                self.nodes[node.id] = node
            
            hidden_nodes = []
            
            hidden_layers = {}
            for i, layer in enumerate(n_hidden):
                hidden_layers[layer] = []
                for j in range(layer):
                    new_id = type(self).get_new_node_id()
                    node = Node(random_choice(config.activations), new_id)
                    hidden_nodes.append(node)
                    hidden_layers[layer].append(node)
                    
            
            self.hidden_nodes = nn.ModuleList(hidden_nodes)

            for node in self.hidden_nodes:
                self.nodes[node.id] = node
                
            return hidden_layers
            
    
    def initialize_connection_genome(self, hidden_layers, initial_connection_prob=1.0, init_connection_prob_fourier=1.0, weight_std=1.0):
        """Initializes the connection genome of the CPPN."""
        prev_layer = self.input_nodes
        for layer in hidden_layers.values():
            for node in layer:
                for prev_node in prev_layer:
                    prob = initial_connection_prob if (int(prev_node.id) > -4 and int(prev_node.id) < 0) else init_connection_prob_fourier
                    if torch.rand(1) < prob:
                        self.connections[f"{prev_node.id},{node.id}"] = Connection(self.rand_weight(weight_std))
            if len(layer) > 0:
                prev_layer = layer
        for node in self.output_nodes:
            for prev_node in prev_layer:
                prob = initial_connection_prob if (int(prev_node.id) > -4 and int(prev_node.id) < 0) else init_connection_prob_fourier
                if torch.rand(1) < prob:
                    self.connections[f"{prev_node.id},{node.id}"] = Connection(self.rand_weight(weight_std))
            
    

    def update_layers(self):
        input_node_ids = range(-1, -self.n_input - 1, -1)
        output_node_ids = range(-self.n_input - 1, -self.n_input - self.n_output - 1, -1)

        self.hidden_nodes = nn.ModuleList()
        for node_id in self.nodes.keys():
            if int(node_id) not in input_node_ids and int(node_id) not in output_node_ids: 
                self.hidden_nodes.append(self.nodes[node_id])
        
        
        self.enabled_connections = [conn_key for conn_key in self.connections if self.connections[conn_key].enabled]
        
        # inputs
        self.layers = [set([n.id for n in self.input_nodes])]

        self.layers.extend(feed_forward_layers([node.id for node in self.input_nodes],
                                          [node.id for node in self.output_nodes],
                                          [(conn_key.split(',')[0], conn_key.split(',')[1]) for conn_key in self.enabled_connections]))
        
        
        for layer_idx, layer in enumerate(self.layers):
            for node_id in layer:
                self.nodes[node_id].layer = layer_idx
                
        self.node_states = None

    def gather_inputs(self, node_id):
        for i in self.node_states:
            if f"{i},{node_id}" in self.enabled_connections:
                yield self.node_states[i] * self.connections[f"{i},{node_id}"].weight
             

    def forward(self, x, channel_first=True, force_recalculate=True, use_graph=False, act_mode='n/a'):
        #print(self.nodes.keys())
        
        #TODO
        all_nodes = list(self.nodes.values()) + list(self.output_nodes) + list(self.input_nodes)

        # Initialize node states
        if self.node_states is None or force_recalculate:
            self.node_states = {node.id: torch.zeros(x.shape[0:2],
                                                     device=x.device,
                                                     requires_grad=False) for node in all_nodes}
            
        # Set input node states
        for i, input_node in enumerate(self.input_nodes):
            self.node_states[input_node.id] = x[:, :, i]

        # Feed forward through layers
        inputs = [node.id for node in self.input_nodes]
        outputs = [node.id for node in self.output_nodes]
        
        #print(outputs)


        for layer in self.layers:
            for node_id in layer:
                # Gather inputs from incoming connections
                node_inputs = list(self.gather_inputs(node_id))
                # Sum inputs and apply activation function
                if len(node_inputs) > 0:
                    self.node_states[node_id] = self.nodes[node_id](torch.sum(torch.stack(node_inputs), dim=0))
                elif node_id not in self.node_states:
                    # TODO: shouldn't need to do this
                    self.node_states[node_id] = torch.zeros(x.shape[0:2], device=x.device, requires_grad=False)
        # Gather outputs
        outputs = [self.node_states[node_id] for node_id in outputs]
        outputs = torch.stack(outputs, dim=(0 if channel_first else -1))
        
        # outputs = torch.sigmoid(outputs)
        
        # normalize?
        outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
        
        # outputs = torch.nn.functional.relu(outputs)
        
        # outputs = torch.abs(outputs)
        
        outputs = torch.clamp(outputs, 0, 1)
        return outputs

    def mutate(self, config):
        """Mutates the CPPN based on the algorithm configuration."""
        add_node = config.prob_add_node
        add_connection = config.prob_add_connection
        remove_node = config.prob_remove_node
        disable_connection = config.prob_disable_connection
        mutate_weights = config.prob_mutate_weight
        mutate_bias = config.prob_mutate_bias
        mutate_activations = config.prob_mutate_activation
        mutate_sgd_lr_sigma = config.mutate_sgd_lr_sigma
        
        rng = lambda: torch.rand(1).item()
        for _ in range(config.mutation_iters):
            if config.single_structural_mutation:
                div = max(1.0, (add_node + remove_node +
                                add_connection + disable_connection))
                r = rng()
                if r < (add_node / div):
                    self.add_node(config)
                elif r < ((add_node + remove_node) / div):
                    self.remove_node(config)
                elif r < ((add_node + remove_node +
                            add_connection) / div):
                    self.add_connection(config)
                elif r < ((add_node + remove_node +
                            add_connection + disable_connection) / div):
                    self.disable_connection()
            else:
                # mutate each structural category separately
                if rng() < add_node:
                    self.add_node(config)
                if rng() < remove_node:
                    self.remove_node(config)
                if rng() < add_connection:
                    self.add_connection(config)
                if rng() < disable_connection:
                    self.disable_connection()
            
            self.mutate_activations(mutate_activations, config)
            self.mutate_weights(mutate_weights, config)
            self.mutate_bias(mutate_bias, config)
            self.mutate_lr(mutate_sgd_lr_sigma)
            self.update_layers()
            self.disable_invalid_connections(config)
            
            self.to(self.device) # TODO shouldn't need this
            
            self.node_states = None # reset the node states
            
            
    
    def disable_invalid_connections(self, config):
        """Disables connections that are not compatible with the current configuration."""
        return # TODO: test, but there should never be invalid connections
        invalid = []
        for key, connection in self.connections.items():
            if connection.enabled:
                if not is_valid_connection(self.nodes,
                                           [k.split(',') for k in self.connections.keys()],
                                           key.split(','),
                                           config,
                                           warn=True):
                    invalid.append(key)
        for key in invalid:
            key.enabled = False
            #del self.connections[key]


    def add_connection(self, config):
        """Adds a connection to the CPPN."""
        self.update_layers()
        
        for _ in range(20):  # try 20 times max
            [from_node, to_node] = random_choice(list(self.nodes.values()),
                                                 2, replace=False)
            if from_node.layer >= to_node.layer:
                continue  # don't allow recurrent connections
            # look to see if this connection already exists
            key = f"{from_node.id},{to_node.id}"
            if key in self.connections.keys():
                existing_cx = self.connections[key]
            else:
                existing_cx = None
            
            # if it does exist and it is disabled, there is a chance to reenable
            if existing_cx is not None:
                if not existing_cx.enabled:
                    if torch.rand(1)[0] < config.prob_reenable_connection:
                        existing_cx.enabled = True # re-enable the connection
                    break  # don't enable more than one connection
                continue # don't add more than one connection

            # else if it doesn't exist, check if it is valid
            if is_valid_connection(self.nodes,
                                           [k.split(',') for k in self.connections.keys()],
                                           key.split(','),
                                           config):
                # valid connection, add
                new_cx = Connection(self.rand_weight(config.weight_init_std))
                
                new_cx_key = f"{from_node.id},{to_node.id}"
                
                self.connections[new_cx_key] = new_cx
                self.update_layers()
                break # found a valid connection
            
            # else failed to find a valid connection, don't add and try again



    def add_node(self, config):
        """Adds a node to the CPPN.
            Looks for an eligible connection to split, add the node in the middle
            of the connection.
        """
        # only add nodes in the middle of non-recurrent connections (TODO)
        eligible_cxs = list(self.connections.keys())

        if len(eligible_cxs) == 0:
            return # there are no eligible connections, don't add a node

        # choose a random eligible connection
        old_cx_key = random_choice(eligible_cxs, 1, replace=False)

        # create the new node
        new_node = Node(random_choice(config.activations), type(self).get_new_node_id())
        
        assert new_node.id not in self.nodes.keys(),\
            "Node ID already exists: {}".format(new_node.id)
        
        self.nodes[new_node.id] =  new_node # add a new node between two nodes
        self.hidden_nodes.append(new_node)
        self.connections[old_cx_key].enabled = False  # disable old connection

        # The connection between the first node in the chain and the
        # new node is given a weight of one and the connection between
        # the new node and the last node in the chain
        # is given the same weight as the connection being split
        
        old_from, old_to = old_cx_key.split(',')
        new_cx_1_key = f"{old_from},{new_node.id}"
        new_cx_1 = Connection(torch.tensor([1.0], device=self.device))
        
        
        assert new_cx_1_key not in self.connections.keys()
        
        self.connections[new_cx_1_key] = new_cx_1

        new_cx_2_key = f"{new_node.id},{old_to}"
        new_cx_2 = Connection(self.connections[old_cx_key].weight)
        assert new_cx_2_key not in self.connections.keys()
        self.connections[new_cx_2_key] = new_cx_2

        self.update_layers() # update the layers of the nodes

        
    def remove_node(self, config):
        """Removes a node from the CPPN.
            Only hidden nodes are eligible to be removed.
        """

        hidden = self.hidden_nodes
        
        if len(hidden) == 0:
            return # no eligible nodes, don't remove a node

        # choose a random node
        node_id_to_remove = random_choice([n.id for n in hidden], 1, False)

        # delete all connections to and from the node
        for key, cx in list(self.connections.items())[::-1]:
            if node_id_to_remove in key.split(','):
                del self.connections[key]
        
        # delete the node
        for key, node in list(self.nodes.items())[::-1]:
            if key == node_id_to_remove:
                del self.nodes[key]
                break

        
        self.update_layers()
        self.disable_invalid_connections(config)


    
    def mutate_activations(self, prob, config):
        """Mutates the activation functions of the nodes."""
        if len(config.activations) == 1:
            return # no point in mutating if there is only one activation function

        eligible_nodes = self.hidden_nodes
        if config.output_activation is None:
            eligible_nodes.extend(self.output_nodes)
        if config.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes)
        for node in eligible_nodes:
            if torch.rand(1)[0] < prob:
                node.set_activation(random_choice(config.activations))
        self.outputs = None # reset the image


    def mutate_weights(self, prob, config):
        """
        Each connection weight is perturbed with a fixed probability by
        adding a floating point number chosen from a uniform distribution of
        positive and negative values """
        R_delta = torch.rand(len(self.connections.items()), device=self.device)
        R_reset = torch.rand(len(self.connections.items()), device=self.device)

        for i, connection in enumerate(self.connections.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.weight_mutation_std)
                connection.add_to_weight(delta)
                
            elif R_reset[i] < config.prob_weight_reinit:
                connection.weight = self.random_weight()

        # self.clamp_weights()
        self.outputs = None # reset the image


    def mutate_bias(self, prob, config):
        R_delta = torch.rand(len(self.nodes.items()), device=self.device)
        R_reset = torch.rand(len(self.nodes.items()), device=self.device)

        for i, node in enumerate(self.nodes.values()):
            if R_delta[i] < prob:
                delta = random_normal(None, 0, config.bias_mutation_std)
                node.bias = node.bias + delta
            elif R_reset[i] < config.prob_weight_reinit:
                node.bias = torch.zeros_like(node.bias)

        self.outputs = None # reset the image
        
        
    def mutate_lr(self, sigma):
        if not sigma:
            return # don't mutate
        self.sgd_lr = self.sgd_lr + random_normal(None, 0, sigma).item()
        self.sgd_lr = max(1e-8, self.sgd_lr)
        self.outputs = None



    def prune(self, config):
        if config.prune_threshold == 0 and config.min_pruned == 0:
            return
        removed = 0
        for cx in list(self.connections.values())[::-1]:
            if abs(cx.weight)< config.prune_threshold:
                del self.connections[cx.key]
                removed += 1
        for _ in range(config.min_pruned - removed):
            min_weight_key = min(self.connections, key=lambda k: self.connections[k].weight.item())
            removed += 1
            del self.connections[min_weight_key]
       
        # remove nodes with no connections
        all_keys = []
        all_keys.extend([cx.split(',')[0] for cx in self.connections])
        all_keys.extend([cx.split(',')[1] for cx in self.connections])
        all_keys.extend([node.id for node in self.input_nodes])
        all_keys.extend([node.id for node in self.output_nodes])

        for node in list(self.nodes.values())[::-1]:
            if node.id not in all_keys:
                del self.nodes[node]
        
        # print("Pruned {} connections".format(removed))
        
        self.update_layers()
        self.disable_invalid_connections(config)

    
    
    def disable_connection(self):
        """Disables a connection."""
        eligible_cxs = list(self.enabled_connections)
        if len(eligible_cxs) < 1:
            return
        cx = random_choice(eligible_cxs, 1, False)
        self.connections[cx].enabled = False

    
    
    
    
    def clear_data(self):
        raise NotImplementedError() # TODO: remove deprecated
        
    def discard_grads(self):
        raise NotImplementedError() # TODO: remove deprecated
        
    def rand_weight(self, std=1.0):
        return torch.randn(1) * std
        
    def clone(self, config, cpu=False, new_id=False):
        """Clones the CPPN, optionally on the CPU. If new_id is True, the new CPPN will have a new id."""
        
        # Create the child as an empty genome
        child = CPPN(config, do_init=False)
          
        # Copy the parent's genome
        for node in self.input_nodes:
            child.input_nodes.append(Node(type(node.activation), node.id, node.bias.item()))
        for node in self.output_nodes:
            child.output_nodes.append(Node(type(node.activation), node.id, node.bias.item()))
        for node in self.hidden_nodes:
            child.hidden_nodes.append(Node(type(node.activation), node.id, node.bias.item()))
        
        for node in child.input_nodes + child.output_nodes + child.hidden_nodes:
            # node.bias = nn.Parameter(node.bias.detach().clone())
            child.nodes[node.id] = node
        
        for conn_key, conn in self.connections.items():
            child.connections[conn_key] = Connection(conn.weight.detach().clone())
        
        child.update_layers()
        
        
        # Configure record keeping information
        if new_id:
            child.id = type(self).get_id()
            child.parents = (self.id, self.id)
            child.lineage = self.lineage + [self.id]
        else:
            child.id = self.id 
            child.parents = self.parents
            child.lineage = self.lineage
        if cpu:
            child.to(torch.device('cpu'))
        else:
            child.to(self.device)
            
        return child
    
    def crossover(self, other, config):
        child = self.clone(config, new_id=True)

        matching1, matching2 = get_matching_connections(
            self.connections, other.connections)
        
        # copy input and output nodes randomly
        input_node_ids = range(-1, -child.n_input - 1, -1)
        output_node_ids = range(-child.n_input - 1, -child.n_input - child.n_output - 1, -1)
        child.nodes = nn.ModuleDict()
        child.input_nodes = nn.ModuleList()
        child.output_nodes = nn.ModuleList()
        
        
        for node_id in input_node_ids:
            node_id = str(node_id)
            from_self = np.random.rand() < .5 
            n = self.nodes[node_id] if from_self else other.nodes[node_id]
            child.nodes[node_id] = Node(n.activation, n.id, n.bias.item())
            child.input_nodes.append(child.nodes[node_id])
                
                
        for node_id in output_node_ids:
            node_id = str(node_id)
            from_self = np.random.rand() < .5 
            n = self.nodes[node_id] if from_self else other.nodes[node_id]
            child.nodes[node_id] = Node(n.activation, n.id, n.bias.item())    
            child.output_nodes.append(child.nodes[node_id])
        
        for match_index in range(len(matching1)):
            # Matching genes are inherited randomly
            from_self = np.random.rand() < .5 
            
            
            if from_self:
                cx_key = matching1[match_index]
                copy_cx = self.connections[cx_key]
            else:
                cx_key = matching2[match_index]
                copy_cx = other.connections[cx_key]
            
            child.connections[cx_key] = Connection(copy_cx.weight.detach().clone(), copy_cx.enabled)
            
            # Disable the connection randomly if either parent has it disabled
            self_enabled = self.connections[cx_key].enabled
            other_enabled = other.connections[cx_key].enabled
                
            if(not self_enabled or not other_enabled):
                if(np.random.rand() < 0.75):  # from Stanley/Miikulainen 2007
                    child.connections[cx_key].enabled = False
            
        
        for cx_key in child.connections.keys():
            to_node, from_node = cx_key.split(',')
            for node in [to_node, from_node]:
                if node in child.nodes.keys():
                    continue
                in_both = node in self.nodes.keys() and node in other.nodes.keys()
                if in_both:
                    from_self = np.random.rand() < .5 
                else:
                    from_self = node in self.nodes.keys()
                n = self.nodes[node] if from_self else other.nodes[node]
                child.nodes[node] = Node(n.activation, n.id, n.bias.item())
                            
            
        
        child.hidden_nodes = nn.ModuleList()
        for node_id in child.nodes.keys():
            if int(node_id) not in input_node_ids and int(node_id) not in output_node_ids: 
                child.hidden_nodes.append(child.nodes[node_id])
        
        child.update_layers()
        child.disable_invalid_connections(config)
        
        return child

    def vis(self, x, fname='cppn_graph'):
        """Visualize the CPPN."""
        make_dot(self.forward(x), show_attrs=True, show_saved=True, params=dict(self.named_parameters())).render(fname, format="pdf")
        
        

if __name__== "__main__":
    from cppn.fourier_features import add_fourier_features
    from torchviz import make_dot
    
    size = (256, 256)
    
    # coordinates
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = initialize_inputs(size[0], size[1],
                               True, False,
                               3, device,
                               coord_range=(-1,1))
    
    inputs = add_fourier_features(
            inputs,
            4,
            .65,
            dims=2,
            include_original=True,
            )
    
    # cppn
    cppn = CPPN(inputs.shape[-1], 3, (32,16), .99).to(device)    
    print(f"Number of parameters: {get_n_params(cppn)}")
    
    # forward pass
    output = cppn(inputs)
    
    import imageio.v2 as imageio
    import cv2
    
    
    target = imageio.imread('../data/sunrise.png', pilmode='RGB')
    # resize
    target = cv2.resize(target, size) / 255.0
    target = torch.tensor(target, dtype=torch.float32, device=device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(cppn.parameters(), lr=1e-3)
    
    from tqdm import trange
    pbar = trange(100000)
    images = []
    try:
        for step in pbar:
            output = cppn(inputs)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            if step % 1 == 0:
                images.append(output.detach().cpu().numpy())
            
    except KeyboardInterrupt:
        pass  
    
      
    import matplotlib.pyplot as plt
    plt.imshow(output.detach().cpu().numpy(), cmap='gray')
    plt.savefig(f'test.png')

    # make gif
    import numpy as np
    imageio.mimsave('test.gif', [np.array(img) for img in images], fps=60)

    make_dot(output, params=dict(cppn.named_parameters())).render("attached", format="png")
