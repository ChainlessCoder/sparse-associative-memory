from SAM.DynamicGraph import DynamicGraph
import torch

def test_createGraphDefault():
    dg = DynamicGraph(input_length = 10, output_length = 15)
    assert list(dg.node.keys.shape) == [2,10]
    assert list(dg.node.strengths.shape) == [2]
    assert dg.edge.connections.sum() == 0
    
def test_createGraphCustom():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    assert list(dg.node.keys.shape) == [7,10]
    assert list(dg.node.strengths.shape) == [7]
    assert dg.edge.connections.sum() == 0
    
def test_add_nodes():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    dg.add_nodes(new_keys= torch.rand(4,10), 
                 new_values= torch.rand(4,15),
                 new_strengths = 0
                )
    assert list(dg.node.keys.shape) == [11,10]
    assert list(dg.node.values.shape) == [11,15]
    assert list(dg.node.strengths.shape) == [11]
    assert dg.edge.connections.sum() == 0
    
def test_remove_nodes():
    node_num = 7
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = node_num)
    indices2remove = [1,2,5]
    left_indices = sorted(list(set(list(range(node_num))) - set(indices2remove)))
    cloned_keys = dg.node.keys[left_indices,:].detach().clone()
    cloned_values = dg.node.values[left_indices,:].detach().clone()
    dg.remove_nodes(node_indices = torch.tensor(indices2remove))
    assert torch.all(torch.eq(cloned_keys, dg.node.keys)).item()
    assert torch.all(torch.eq(cloned_values, dg.node.values)).item()
    assert list(dg.node.strengths.shape) == [4]
    assert dg.edge.connections.sum() == 0
    
def test_empty_graph():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 0)
    assert list(dg.node.keys.shape) == [0,10]
    assert list(dg.node.values.shape) == [0,15]

def test_add_then_remove_nodes():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 0)
    node_num = 7
    dg.add_nodes(new_keys= torch.rand(node_num,10), 
                 new_values= torch.rand(node_num,15),
                 new_strengths = 0
                )
    indices2remove = [1,2,5,6]
    left_indices = sorted(list(set(list(range(node_num))) - set(indices2remove)))
    cloned_keys = dg.node.keys[left_indices,:].detach().clone()
    cloned_values = dg.node.values[left_indices,:].detach().clone()
    dg.remove_nodes(node_indices = torch.tensor(indices2remove))
    assert torch.all(torch.eq(cloned_keys, dg.node.keys)).item()
    assert torch.all(torch.eq(cloned_values, dg.node.values)).item()
    assert list(dg.node.strengths.shape) == [3]
    assert dg.edge.connections.sum() == 0
    
def test_get_nodes_num():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 0)
    num = dg.get_nodes_num()
    assert num == 0
    node_num = 500
    dg.add_nodes(new_keys= torch.rand(node_num,10), 
                 new_values= torch.rand(node_num,15),
                 new_strengths = 0
                )
    num = dg.get_nodes_num()
    assert num == node_num

def test_add_edges():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5])
                )
    res = torch.tensor([[0, 0, 1, 2, 3, 5],
                        [0, 1, 1, 4, 4, 6]])
    assert torch.all(torch.eq(dg.edge.connections, res)).item()
    

def test_remove_edges():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    dg.add_edges(U = torch.tensor([1,1,0,1,1,3,4,6]), 
                 V = torch.tensor([0,1,0,0,1,4,2,5])
                )
    dg.remove_edges(U = torch.tensor([0,2,5]), 
                    V = torch.tensor([0,4,6])
                   )
    res = torch.tensor([[0,1,3],[1,1,4]])
    assert torch.all(torch.eq(dg.edge.connections, res)).item()
    
    
def test_add_to_node_strengths():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    dg.update_node_strengths(node_indices = torch.tensor([2,3,6,6]), 
                             node_strengths = torch.tensor([1.,2.,6.,3.]),
                             operation = 'add'
                          )
    res = torch.tensor([1.,1.,2.,3.,1.,1.,5.5])
    assert torch.all(torch.eq(dg.node.strengths, res)).item()
    
def test_reduce_from_node_strengths():
    dg = DynamicGraph(input_length = 10, output_length = 15, node_num = 7)
    dg.update_node_strengths(node_indices = torch.tensor([2,3,6,6]), 
                             node_strengths = torch.tensor([1,2,6,3]),
                             operation = 'sub'
                          )
    res = torch.tensor([1.,1.,0.,-1.,1.,1.,-3.])
    assert torch.all(torch.eq(dg.node.strengths, res)).item()
