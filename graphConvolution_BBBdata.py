#!/usr/bin/env python
"""BBBPredictionEngine

 Predict blood-brain barrier permeation
 from molecular positions
 							                                    
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.tensor_graph import TensorGraph

def loadBBB(filename, featurizer='GraphConv', split='index', reload=True, K=4):
  """Load BBB data"""
  import os

  bbb_tasks = ['BBB']

  data_dir = 'bbbdata'
  #data_dir = dc.utils.get_data_dir()
  if reload:
    #save_dir = os.path.join(data_dir, "bbb/" + featurizer + "/" + split)
    save_dir = os.path.join(data_dir, filename)
    loaded, all_dataset, transformers = dc.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return bbb_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, filename)
  #dataset_file = os.path.join(data_dir, "bbb.csv.gz")

  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = dc.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = dc.feat.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = dc.feat.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)

  loader = dc.data.CSVLoader(
      tasks=bbb_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': dc.splits.IndexSplitter(),
      'random': dc.splits.RandomSplitter(),
      'scaffold': dc.splits.ScaffoldSplitter(),
      'butina': dc.splits.ButinaSplitter(),
      'task': dc.splits.TaskSplitter()
  }

  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
  else:
    train, valid, test = splitter.train_valid_test_split(dataset)
    all_dataset = (train, valid, test)
    if reload:
      dc.utils.save.save_dataset_to_disk(data_dir, train, valid, test,
                                               transformers)
      #dc.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
  return bbb_tasks, all_dataset, transformers

tg = TensorGraph(use_queue=False)

# Example
# Load dataset and define features
from deepchem.models.tensorgraph.layers import Feature
filename = 'finaldata.csv'
tox21_tasks, tox21_datasets, transformers = loadBBB(
        filename, featurizer='GraphConv', split='index', reload=True, K=4)

#atom_features = Feature(shape=(None, 75))
#degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
#membership = Feature(shape=(None,), dtype=tf.int32)
#
#deg_adjs = []
#for i in range(0, 10 + 1):
#    deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
#    deg_adjs.append(deg_adj)
#
## define the graph convolution network
#from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
#from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
#
#batch_size = 50
#
#gc1 = GraphConv(
#    64,
#    activation_fn=tf.nn.relu,
#    in_layers=[atom_features, degree_slice, membership] + deg_adjs)
#batch_norm1 = BatchNorm(in_layers=[gc1])
#gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)
#gc2 = GraphConv(
#    64,
#    activation_fn=tf.nn.relu,
#    in_layers=[gp1, degree_slice, membership] + deg_adjs)
#batch_norm2 = BatchNorm(in_layers=[gc2])
#gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)
#dense = Dense(out_channels=128, activation_fn=tf.nn.relu, in_layers=[gp2])
#batch_norm3 = BatchNorm(in_layers=[dense])
#readout = GraphGather(
#    batch_size=batch_size,
#    activation_fn=tf.nn.tanh,
#    in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)
#
## train
## Epochs set to 1 to render tutorials online.
## Set epochs=10 for better results.
#tg.fit_generator(data_generator(train_dataset, epochs=1))
#
## Set nb_epoch=10 for better results.
#metric = dc.metrics.Metric(
#    dc.metrics.roc_auc_score, np.mean, mode="classification")
#
## evaluate
#def reshape_y_pred(y_true, y_pred):
#    """
#    TensorGraph.Predict returns a list of arrays, one for each output
#    We also have to remove the padding on the last batch
#    Metrics taks results of shape (samples, n_task, prob_of_class)
#    """
#    n_samples = len(y_true)
#    retval = np.stack(y_pred, axis=1)
#    return retval[:n_samples]
#
#
#print("Evaluating model")
#train_predictions = tg.predict_on_generator(data_generator(train_dataset, predict=True))
#train_predictions = reshape_y_pred(train_dataset.y, train_predictions)
#train_scores = metric.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
#print("Training ROC-AUC Score: %f" % train_scores)
#
#valid_predictions = tg.predict_on_generator(data_generator(valid_dataset, predict=True))
#valid_predictions = reshape_y_pred(valid_dataset.y, valid_predictions)
#valid_scores = metric.compute_metric(valid_dataset.y, valid_predictions, valid_dataset.w)
#print("Valid ROC-AUC Score: %f" % valid_scores)

