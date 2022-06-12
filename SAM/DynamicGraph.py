from torch import Tensor, tensor, unique, transpose, cat, arange, sort, ones, float64
from torch.nn import Module, Parameter
from torch.nn.init import xavier_uniform_
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from typing import Optional, Union, Tuple, List
        
class edges():
    def __init__(self):
        self.connections = tensor([])
        
class nodes():
    def __init__(self, keys, values, strengths):
        self.keys = keys
        self.values = values
        self.strengths = strengths

class DynamicGraph(Module):
    
    def __init__(self, 
                 input_length: int,
                 output_length: int,
                 node_num: Optional[int] = 2,
                 node_strengths: Optional[int] = 1.
                ):
        super(DynamicGraph, self).__init__()
        self.node = nodes(keys = Parameter(Tensor(node_num, input_length), requires_grad = False), 
                          values = Parameter(Tensor(node_num, output_length)), 
                          strengths = ones(node_num, dtype = float64, requires_grad = False) * node_strengths
                         )
       
        self.edge = edges()
        xavier_uniform_(self.node.keys)
        xavier_uniform_(self.node.values)
        
        
    def get_nodes_num(self):
        return self.node.keys.shape[0]
    
    def get_edges(self):
        U = self.edge.connections[0,:]
        V = self.edge.connections[1,:]
        return U, V
    
    def add_nodes(self, 
                  new_keys: Tensor, 
                  new_values: Tensor, 
                  new_strengths: Union[Tensor, int]
                 ):
        assert type(new_keys) == Tensor, "invalid type for new_keys"
        assert type(new_values) == Tensor, "invalid type for new_values"
        if len(new_keys.size()) == 1:
            new_keys = new_keys.unsqueeze(0)
        if len(new_values.size()) == 1:
            new_keys = new_keys.unsqueeze(0)
        assert new_keys.size(0) == new_values.size(0), "first dimension must be the same"
        self.node.keys = Parameter(data = cat((self.node.keys, new_keys), dim=0), requires_grad = False)
        self.node.values = Parameter(data = cat((self.node.values, new_values), dim=0))
        if type(new_strengths) == int:
            new_strengths = ones(new_keys.size(0)) * new_strengths
        self.node.strengths = cat((self.node.strengths, new_strengths), dim=0)
    
    def add_edges(self, 
                  U: Tensor, 
                  V: Tensor
                 ):
        assert (U.max() <= len(self.node.keys)-1) and (V.max() <= len(self.node.keys)-1), "edge index exceeds node count"
        new_e = sort(cat((U.unsqueeze(0),V.unsqueeze(0)), dim=0), dim=0).values
        if self.edge.connections.shape[0] == 0:
            m = int(new_e[0].max()) + 1
            n = int(new_e[1].max()) + 1
            self.edge.connections, _ = coalesce(new_e, None, m=m, n=n)
        else:
            concatinated_e = cat((self.edge.connections, new_e),dim=1)
            m = int(concatinated_e[0].max()) + 1
            n = int(concatinated_e[1].max()) + 1
            self.edge.connections, _ = coalesce(concatinated_e, None, m=m, n=n)
    
    def remove_edges(self, 
                     U: Tensor, 
                     V: Tensor
                    ):
        N = self.get_nodes_num()
        E = self.edge.connections.T
        Erem = sort(cat((U.unsqueeze(0), V.unsqueeze(0)), dim=0), dim=0).values.T
        mask = E.unsqueeze(1) == Erem
        mask = mask.all(-1)
        non_repeat_mask = ~mask.any(-1)
        new_e = self.edge.connections[:,non_repeat_mask]
        m = int(new_e[0].max()) + 1
        n = int(new_e[1].max()) + 1
        self.edge.connections, _ = coalesce(new_e, None, m=m, n=n)
        
    def remove_nodes(self, 
                     node_indices: Union[Tensor, List]
                    ):
        if type(node_indices) == list:
            node_indices = tensor(node_indices)
        node_mask = (arange(self.node.keys.size(0)).unsqueeze(1) == node_indices).sum(1) == 0
        self.node.keys = Parameter(self.node.keys[node_mask,:], requires_grad = False)
        self.node.values = Parameter(self.node.values[node_mask,:])
        self.node.strengths = self.node.strengths[node_mask]
        if self.edge.connections.shape[0] != 0:
            edge_mask = self.edge.connections.unsqueeze(2) == node_indices
            edge_mask = transpose(edge_mask, 1, 2).sum(1).sum(0) < 1
            edge_updates = self.edge.connections.unsqueeze(2) > node_indices
            edge_updates = transpose(edge_updates, 1,2).sum(1)
            self.edge.connections = self.edge.connections[:,edge_mask]
            self.edge.values = self.edge.values[edge_mask]
            edge_updates = edge_updates[:,edge_mask]
            self.edge.connections -= edge_updates
    
    def update_node_strengths(self, 
                              node_indices: Tensor, 
                              node_strengths: Tensor, 
                              operation: str
                             ):
        reduced_strengths = scatter_mean(node_strengths, node_indices)
        if operation == "add":
            self.node.strengths += reduced_strengths
        elif operation == "sub":
            self.node.strengths -= reduced_strengths
        else:
            raise ValueError('operation not supported. Must be either add or sub')
