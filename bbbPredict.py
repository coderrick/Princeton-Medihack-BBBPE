#!/usr/bin/env python
"""bbbPredict

 Predict blood-brain barrier permeation
 from molecular positions
 							                                    
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, BatchNorm, GraphConv
from deepchem.models.tensorgraph.models.graph_models import GraphConvTensorGraph
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, SoftMax, \
    SoftMaxCrossEntropy, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
  
class bbbPredict(object):

  def loadBBBpredict(self, filename, featurizer='GraphConv', K=4, reload=False):
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
  
    return bbb_tasks, dataset, transformers

  def getPrediction(self):
  
    filename = 'onepredictdata.csv'
    #fpredict = open('bbbdata/'+filename, 'w')
    #fpredict.write('smiles,BBB\n')
    #fpredict.write('CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0\n')
    #fpredict.write('O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1I,0')
    
    #fpredict.close()
    bbb_tasks, bbb_datasets, transformers = self.loadBBBpredict(
            filename, featurizer='GraphConv', K=4, reload=False)
    
    model = GraphConvModel(
        len(bbb_tasks), batch_size=50, mode='classification',model_dir='models')
    
    model = model.load_from_dir('models')
    
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    
    predicted_val = model.predict(bbb_datasets, transformers)
  
    return predicted_val

if __name__ == '__main__':
  sys.exit(main())
