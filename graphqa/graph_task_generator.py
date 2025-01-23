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

r"""The graph tasks to be tried with LLMs."""

from collections.abc import Sequence
import os
import random
import glob

from absl import app
from absl import flags
import networkx as nx
import numpy as np

import graph_task
import graph_task_utils as utils

_TASK = flags.DEFINE_enum(
    'task',
    None,
    [
        'edge_existence',
        'node_degree',
        'node_count',
        'edge_count',
        'connected_nodes',
        'cycle_check',
        'disconnected_nodes',
        'reachability',
        'shortest_path',
        'maximum_flow',
        'triangle_counting',
        'node_classification',
    ],
    'The task to generate datapoints.',
    required=True,
)
_ALGORITHM = flags.DEFINE_enum(
    'algorithm',
    None,
    ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path', 'all'],
    'The graph generator algorithm to generate datapoints.',
    required=True,
)
_TASK_DIR = flags.DEFINE_string(
    'task_dir', None, 'The directory to write tasks.', required=True
)
_GRAPHS_DIR = flags.DEFINE_string(
    'graphs_dir', None, 'The directory containing the graphs.', required=True
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed to use for task generation.',
    required=True,
)


TASK_CLASS = {
    'edge_existence': graph_task.EdgeExistence,
    'node_degree': graph_task.NodeDegree,
    'node_count': graph_task.NodeCount,
    'edge_count': graph_task.EdgeCount,
    'connected_nodes': graph_task.ConnectedNodes,
    'cycle_check': graph_task.CycleCheck,
    'disconnected_nodes': graph_task.DisconnectedNodes,
    'reachability': graph_task.Reachability,
    'shortest_path': graph_task.ShortestPath,
    'maximum_flow': graph_task.MaximumFlow,
    'triangle_counting': graph_task.TriangleCounting,
    'node_classification': graph_task.NodeClassification,
}


def zero_shot(
    task,
    graphs,
    algorithms,
    text_encoders,
    cot,
    subfolder_name,
):
  """Creating zero-shot or zero-cot examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
  """
  assert not cot, "Not doing COT in this experiment"
  if not os.path.exists(_TASK_DIR.value):
    os.makedirs(_TASK_DIR.value)
  for text_encoder in text_encoders:
    zero_shot_examples = utils.create_zero_shot_task(
        task, graphs, algorithms, [text_encoder], cot=cot
    )
    file_name = task.name + f'_{text_encoder}' + '.json'
    utils.write_examples(
        zero_shot_examples,
        os.path.join(_TASK_DIR.value, subfolder_name, file_name),
    )


def generate_random_sbm_graph(random_state):
  # Sampling a small number as the probability of the two nodes in different
  # communities being connected.
  small_number = random.uniform(0, 0.05)
  # Sampling a large number as probability of the nodes in one community
  # being connected.
  large_number = random.uniform(0.6, 0.8)
  number_of_nodes = random.choice(np.arange(5, 20))
  sizes = [number_of_nodes // 2, number_of_nodes // 2]
  probs = [[large_number, small_number], [small_number, large_number]]
  return nx.stochastic_block_model(sizes, probs, seed=random_state)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _ALGORITHM.value == 'all':
    algorithms = ['er', 'ba', 'sbm', 'sfn', 'complete', 'star', 'path']
  else:
    algorithms = [_ALGORITHM.value]

  text_encoders = [
      'adjacency',
      'incident',
      'coauthorship',
      'friendship',
      'south_park',
      'got',
      'social_network',
      'politician',
      'expert',
  ]

  # Defining a task on the graphs
  task = TASK_CLASS[_TASK.value]()

  # Loading the graphs.
  # graphs = []
  # generator_algorithms = []
  for algorithm in algorithms:
    for foldername in glob.glob(os.path.join(
        _GRAPHS_DIR.value,
        algorithm,
        "*"
    )):
      graphs = utils.load_graphs(foldername)
      # graphs += loaded_graphs
      # generator_algorithms += [algorithm] * len(loaded_graphs)
      generator_algorithms = [algorithm] * len(graphs)
      subfolder_name=os.path.basename(foldername.rstrip('/ '))
      if not os.path.exists(os.path.join(_TASK_DIR.value, subfolder_name)):
        os.makedirs(os.path.join(_TASK_DIR.value, subfolder_name), exist_ok=True)
      zero_shot(
          task,
          graphs,
          generator_algorithms,
          text_encoders,
          cot=False,
          subfolder_name=subfolder_name
      )


if __name__ == '__main__':
  app.run(main)
