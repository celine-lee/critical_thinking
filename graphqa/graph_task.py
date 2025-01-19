# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The graph tasks to be tried with LLMs."""

import random

import networkx as nx
import numpy as np

import graph_text_encoder

# Modified from the original code to leave flexible for COT, model's choice. 
# Also requests change in model answer formatting. Makes some tasks easier (e.g. no alphabetical)

class GraphTask:
  """The parent class for all the graph tasks."""

  def __init__(self):
    self.name = 'default'
    self.maximum_nnodes_cot_graph = 10

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    raise NotImplementedError()

  def create_few_shot_example(
      self, graph, encoding_method, cot
  ):
    assert False, "No few-shot in this experiment."
    # raise NotImplementedError()


class CycleCheck(GraphTask):
  """The graph task to check if there is at least one cycle or not."""

  def __init__(self):
    super().__init__()
    self.name = 'cycle_check'
    self._task_description = 'Q: Is there a cycle in this graph? Provide your final answer as True or False, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = (
          graph_text_encoder.encode_graph(graph, encoding_method)
          + self._task_description
      )
      try:
        cycle = nx.find_cycle(graph)
        # answer = 'Yes, there is a cycle.'
        answer = 'True'
      except nx.NetworkXNoCycle:
        # answer = 'No, there is no cycle.'
        answer = 'False'
        cycle = None
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'cycle': cycle,
          'lrun': len(cycle) if cycle else 0,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': self._task_description,
          # # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

class EdgeExistence(GraphTask):
  """The graph task to check if an edge exist in a graph or not."""

  def __init__(self):
    super().__init__()
    self.name = 'edge_existence'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is node %s connected to node %s? Provide your final answer as True or False, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]' % (
          name_dict[source],
          name_dict[target],
      )
      question += task_description
      if ((source, target) in graph.edges()) or (
          (target, source) in graph.edges()
      ):
        answer = 'True'
      else:
        answer = 'False'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': 1,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict



class NodeCount(GraphTask):
  """The graph task for finding number of nodes in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_count'
    self._task_description = 'Q: How many nodes are in this graph? Provide your final answer as an integer, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = len(graph.nodes())
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': 1,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': self._task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def get_nodes_string(self, name_dict, nnodes):
    node_string = ''
    for i in range(nnodes - 1):
      node_string += name_dict[i] + ', '
    node_string += 'and ' + name_dict[nnodes - 1]
    return node_string


class NodeDegree(GraphTask):
  """The graph task for finding degree of a node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_degree'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: What is the degree of node %s? Provide your final answer as an integer, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]' % name_dict[source_node]
      )
      question += task_description
      answer = graph.degree[source_node]
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': answer,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_edge_string(
      self, name_dict, graph, source_node
  ):
    """Gets a string identifying the edges a given node is connected to."""
    edge_string = ''
    target_edges = graph.edges(source_node)
    target_nodes = []
    for edge in target_edges:
      target_nodes.append(edge[1])
    if target_nodes:
      for i in range(len(target_nodes) - 1):
        edge_string += name_dict[target_nodes[i]] + ', '
      edge_string += 'and ' + name_dict[target_nodes[-1]]
    else:
      edge_string = 'no nodes'
    return edge_string


class EdgeCount(GraphTask):
  """The graph task for finding number of edges in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'edge_count'
    self._task_description = 'Q: How many edges are in this graph? Provide your final answer as an integer, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      question += self._task_description
      answer = len(graph.edges())
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': answer,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': self._task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict

  def get_edges_string(
      self, name_dict, edges
  ):
    edges_string = ''
    for edge in edges:
      edges_string += (
          '(' + name_dict[edge[0]] + ', ' + name_dict[edge[1]] + '), '
      )
    return edges_string.strip()[:-1]


class ConnectedNodes(GraphTask):
  """The graph task for finding connected nodes to a given node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'connected_nodes'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: List all the nodes connected to %s. Provide your final answer as a list, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]'
          % name_dict[source_node]
      )
      question += task_description
      outgoing_edges = list(graph.edges(source_node))
      answer = []
      if outgoing_edges:
        answer = self.get_connected_nodes(outgoing_edges, name_dict)
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': len(answer),
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_connected_nodes(
      self, edges, name_dict
  ):
    """Gets a list including all the nodes that are connected to source."""
    connected_nodes = []
    for edge in edges:
      connected_nodes.append(name_dict[edge[1]])
    return connected_nodes
    


class DisconnectedNodes(GraphTask):
  """The task for finding disconnected nodes for a given node in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'disconnected_nodes'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      source_node = random.sample(list(graph.nodes()), k=1)[0]
      task_description = (
          'Q: List all the nodes that are not connected to %s. Provide your final answer as a list, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]' % name_dict[source_node]
      )
      question += task_description
      outgoing_edges = list(graph.edges(source_node))
      answer, lrun = self.get_disconnected_nodes(
          source_node, outgoing_edges, name_dict, list(graph.nodes())
      )
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': lrun,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source_node],
      }
    return examples_dict

  def get_disconnected_nodes(
      self,
      source,
      edges,
      name_dict,
      all_nodes,
  ):
    """Gets a list with all the nodes that are not connected to source."""
    lrun = 0
    for edge in edges:
      if edge[1] in all_nodes:
        lrun += 1
        all_nodes.remove(edge[1])
    if source in all_nodes:
      lrun += 1
      all_nodes.remove(source)
    all_nodes_names = []
    for node in all_nodes:
      all_nodes_names.append(name_dict[node])
    return all_nodes_names, lrun


class Reachability(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'reachability'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = 'Q: Is there a path from node %s to node %s? Provide your final answer as True or False, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]' % (
          name_dict[source],
          name_dict[target],
      )
      question += task_description
      shortest_path = []
      answer = 'False'
      if nx.has_path(graph, source, target):
        shortest_path = nx.shortest_path(graph, source, target)
        answer = 'True'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': len(shortest_path),
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict


class ShortestPath(GraphTask):
  """The graph task to check if there is a path from a source to target."""

  def __init__(self):
    super().__init__()
    self.name = 'shortest_path'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
          'Q: What is the number of edges on the shortest path from node %s to node'
          ' %s? Provide your final answer as an integer or None, following this template: [ANSWER]\nanswer = YOUR ANSWER\n[/ANSWER]'
          % (
              name_dict[source],
              name_dict[target],
          )
      )
      question += task_description
      try:
        path = nx.shortest_path(graph, source, target)
        answer = len(path) - 1
      except nx.NetworkXNoPath:
        answer = None
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'lrun': len(path),
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict
    
# these ones are kinda hard lets ignore them.

class TriangleCounting(GraphTask):
  """The graph task to count the number of triangles in a graph."""

  def __init__(self):
    super().__init__()
    self.name = 'triangle_counting'
    self._task_description = 'Q: How many triangles are in this graph?\nA: '

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    for ind, graph in enumerate(graphs):
      question = (
          graph_text_encoder.encode_graph(graph, encoding_method)
          + self._task_description
      )
      ntriangles = int(np.sum(list(nx.triangles(graph).values())) / 3)

      answer = '%i.' % ntriangles
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': self._task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [],
      }
    return examples_dict


class MaximumFlow(GraphTask):
  """The graph task to compute the maximum flow from a source to a target."""

  def __init__(self):
    super().__init__()
    self.name = 'maximum_flow'

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]

    for ind, graph in enumerate(graphs):
      graph = add_edge_weight(graph)
      source, target = random.sample(list(graph.nodes()), k=2)
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      task_description = (
          'Q: What is the maximum capacity of the flow from node %s to node'
          ' %s?\nA: ' % (name_dict[source], name_dict[target])
      )
      question += task_description
      maximum_flow_value = nx.maximum_flow(
          graph, source, target, capacity='weight'
      )[0]
      answer = str(maximum_flow_value) + '.'
      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': len(graph.nodes()),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          'node_ids': [source, target],
      }
    return examples_dict


def has_edge_weights(graph):
  for _, _, data in graph.edges(data=True):
    if 'weight' not in data:
      return False
  return True


def add_edge_weight(graph):
  if has_edge_weights(graph):
    return graph
  else:
    for edge in graph.edges():
      graph[edge[0]][edge[1]]['weight'] = random.randint(1, 10)
    return graph


class NodeClassification(GraphTask):
  """The graph task to classify a given node in the graph."""

  def __init__(self):
    super().__init__()
    self.name = 'node_classification'
    self.classes = [
        'soccer',
        'baseball',
        'tennis',
        'golf',
        'football',
        'surfing',
    ]

  def prepare_examples_dict(
      self,
      graphs,
      generator_algorithms,
      encoding_method,
  ):
    classes = random.sample(list(self.classes), k=2)
    examples_dict = {}
    name_dict = graph_text_encoder.TEXT_ENCODER_DICT[encoding_method]
    for ind, graph in enumerate(graphs):
      question = graph_text_encoder.encode_graph(graph, encoding_method)
      nnodes = len(graph.nodes())
      # Sampling nnodes // 2 + 1 nodes.
      sampled_nodes = random.sample(
          list(graph.nodes(data=True)), k=nnodes // 2 + 1
      )
      # Adding the class of half of the nodes.
      for node_data in sampled_nodes[:-1]:
        node_class = classes[node_data[1]['block']]
        question += (
            'Node ' + name_dict[node_data[0]] + ' likes ' + node_class + '.\n'
        )
      # Reserving the last sampled node for the question.
      task_description = 'Q: Does node %s like %s or %s?\nA: ' % (
          name_dict[sampled_nodes[-1][0]],
          classes[0],
          classes[1],
      )
      question += task_description
      answer = classes[sampled_nodes[-1][1]['block']]

      examples_dict[ind] = {
          'question': question,
          'answer': answer,
          'nnodes': str(nnodes),
          'nedges': len(graph.edges()),
          'task_description': task_description,
          # 'graph': graph,
          'algorithm': generator_algorithms[ind],
          # id of the last samples node
          'node_ids': [sampled_nodes[-1][0]],
      }

    return examples_dict
