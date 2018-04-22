#!/usr/bin/env python
"""BBBPredictionEngine

 Predict blood-brain barrier permeation
 from molecular structures
 							                                    
 Largely from the deepchem 
 'Graph Convolutions For Tox21' Tutorial

 usage example:
     'python graphConvolution_BBBdata.py --filename finaldata.csv
                                         --split_method index
                                         --training_fraction 0.6 
                                         --testing_fraction 0.2
                                         --validation_fraction 0.2
                                         --confusion_matrix
                                         --bbbp_split 0.5 --generate'

"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, argparse
import numpy as np
import tensorflow as tf
import deepchem as dc

# Load deepchem modules
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel,\
                                                            GraphConvTensorGraph
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, \
     Dense, BatchNorm, GraphConv, GraphPool, GraphGather, \
     SoftMax, SoftMaxCrossEntropy, Label, Stack, \
     Weights, WeightedError
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

class graphConvolution_BBBdata(object):
  """Callable graph convolution for bbb data
  """

  def addParser(self, parser):
    """
    Get relevant values from an argparse parser
    """
    if (parser.parse_args().filename):
      self.filename = parser.parse_args().filename
    else:
      self.filename = 'finaldata.cs'

    if (parser.parse_args().split_method):
      self.split_method = parser.parse_args().split_method
    else:
      self.split_method = 'scaffold'

    self.training_fraction = parser.parse_args().training_fraction
    self.testing_fraction = parser.parse_args().testing_fraction
    self.validation_fraction = parser.parse_args().validation_fraction

    if (parser.parse_args().confusion_matrix):
      self.confusion_matrix = parser.parse_args().confusion_matrix
    else:
      self.confusion_matrix = True

    self.calc_confusion_matrix = parser.parse_args().confusion_matrix

    bbbp_split_input = parser.parse_args().bbbp_split
    self.bbbp_split = [0.5, 0.5]
    if 0 < bbbp_split_input < 0.5: 
      self.bbbp_split = [1.-bbbp_split,_input, bbbp_split_input]
    elif 0.5 < bbbp_split_input < 1:
      self.bbbp_split = [bbbp_split,_input, 1.-bbbp_split_input]
    elif bbbp_split_input == 0.5: 
      self.bbbp_split = [0.5, 0.5]
    else:
      print('bbbp_split value must be between 1 and 0')
      self.bbbp_split = [0.5, 0.5]

  # evaluate
  def reshape_y_pred(self, y_true, y_pred):
    """
    TensorGraph.Predict returns a list of arrays, one for each output
    We also have to remove the padding on the last batch
    Metrics taks results of shape (samples, n_task, prob_of_class)
    """
    n_samples = len(y_true)
    return y_pred[:n_samples]
    #retval = np.stack(y_pred, axis=1)
    #return retval[:n_samples]

  def loadBBB(self, filename, featurizer='GraphConv', split='index', reload=True, K=4,
              ftrain=0.8, fvalid=0.1, ftest=0.1):
    """Load BBB data"""
    import os
  
    # BBB is a integer, that is either 
    #        1, for passing the blood-brain barrier (bbb)
    #     or 2 for NOT passing the bbb
    bbb_tasks = ['BBB']
  
    data_dir = 'bbbdata'
    if reload:
      # for future implementation into deepchem
      #save_dir = os.path.join(data_dir, "bbb/" + featurizer + "/" + split)
      save_dir = os.path.join(data_dir, filename)
      loaded, all_dataset, transformers = dc.utils.save.load_dataset_from_disk(
          save_dir)
      if loaded:
        return bbb_tasks, all_dataset, transformers
  
    dataset_file = os.path.join(data_dir, filename)
  
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
      train, valid, test = splitter.train_valid_test_split(dataset, 
                          frac_train=ftrain, frac_valid=fvalid, frac_test=ftest)
      all_dataset = (train, valid, test)
      if reload:
        print('reloading')
        dc.utils.save.save_dataset_to_disk(data_dir, train, valid, test,
                                                 transformers)
    return bbb_tasks, all_dataset, transformers
  
  def runGraphConv(self, filename='finaldata.csv'):
    """Read the bbb penetration data
    """
    # Example
    # Load dataset and define features
    filename = 'finaldata.csv'
    # splitter options: index random scaffold butina task
    # featurizer options: ECFP GraphConv Weave Raw AdjacencyConv
    # reload feature not quite working
    # not sure what the K value is
    bbb_tasks, bbb_datasets, transformers = self.loadBBB(
            filename, featurizer='GraphConv', split=self.split_method, reload=False, K=4,
                ftrain=self.training_fraction, fvalid=self.validation_fraction, 
                ftest=self.testing_fraction)
    
    train_dataset, valid_dataset, test_dataset = bbb_datasets
    # print( "mean bbb value (like % passing) in the: training, validation and test data sets" )
    #print(np.mean(train_dataset.y), np.mean(valid_dataset.y), np.mean(test_dataset.y))
    
    model = GraphConvModel(
        len(bbb_tasks), batch_size=50, mode='classification',model_dir='models')
    # Set nb_epoch=10 for better results.
    model.fit(train_dataset, nb_epoch=1)
    
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    
    print("Evaluating model")
    train_scores = model.evaluate(train_dataset, [metric], transformers)
    print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
    valid_scores = model.evaluate(valid_dataset, [metric], transformers)
    print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])
    
    tg = TensorGraph(use_queue=False)
    
    atom_features = Feature(shape=(None, 75))
    degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    membership = Feature(shape=(None,), dtype=tf.int32)
    
    deg_adjs = []
    for i in range(0, 10 + 1):
        deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
        deg_adjs.append(deg_adj)
    
    # define the graph convolution network
    batch_size = 50
    
    gc1 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[atom_features, degree_slice, membership] + deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)
    gc2 = GraphConv(
        64,
        activation_fn=tf.nn.relu,
        in_layers=[gp1, degree_slice, membership] + deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])
    gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)
    dense = Dense(out_channels=128, activation_fn=tf.nn.relu, in_layers=[gp2])
    batch_norm3 = BatchNorm(in_layers=[dense])
    readout = GraphGather(
        batch_size=batch_size,
        activation_fn=tf.nn.tanh,
        in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)
    
    costs = []
    labels = []
    for task in range(len(bbb_tasks)):
        classification = Dense(
            out_channels=2, activation_fn=None, in_layers=[readout])
    
        softmax = SoftMax(in_layers=[classification])
        tg.add_output(softmax)
    
        label = Label(shape=(None, 2))
        labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
    
    all_cost = Stack(in_layers=costs, axis=1)
    weights = Weights(shape=(None, len(bbb_tasks)))
    loss = WeightedError(in_layers=[all_cost, weights])
    tg.set_loss(loss)
    # train
    # Epochs set to 1 to render tutorials online.
    # Set epochs=10 for better results.
    def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
      for epoch in range(epochs):
        if not predict:
            print('Starting epoch %i' % epoch)
        for ind, (X_b, y_b, w_b, ids_b) in enumerate(
            dataset.iterbatches(
                batch_size, pad_batches=pad_batches, deterministic=True)):
          d = {}
          for index, label in enumerate(labels):
            d[label] = to_one_hot(y_b[:, index])
          d[weights] = w_b
          multiConvMol = ConvMol.agglomerate_mols(X_b)
          d[atom_features] = multiConvMol.get_atom_features()
          d[degree_slice] = multiConvMol.deg_slice
          d[membership] = multiConvMol.membership
          for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
          yield d
    
    tg.fit_generator(data_generator(train_dataset, epochs=1))
    
    # Set nb_epoch=10 for better results.
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    
    
    print("Evaluating model")
    train_predictions = tg.predict_on_generator(data_generator(train_dataset, predict=True))
    train_predictions = self.reshape_y_pred(train_dataset.y, train_predictions)
    train_scores = metric.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
    print("Training ROC-AUC Score: %f" % train_scores)
    
    valid_predictions = tg.predict_on_generator(data_generator(valid_dataset, predict=True))
    valid_predictions = self.reshape_y_pred(valid_dataset.y, valid_predictions)
    valid_scores = metric.compute_metric(valid_dataset.y, valid_predictions, valid_dataset.w)
    print("Valid ROC-AUC Score: %f" % valid_scores)
    
    # save the model
    model.save()
  
    #self.writePredictions(train_dataset.y, train_predictions)
    self.writePredictions(valid_dataset.y, valid_predictions)
    # calculate the confusion matrices
    # and output the predictions

  def writePredictions(self, data, predictions):
    """
    writes predictions, expects data and predictions to be n x 2 numpy arrays
    """
    
    # true_positive false_positive
    # false_negative true_negative 
    if self.calc_confusion_matrix:
      confusion_matrix = np.array([ [0, 0], [0, 0]])

    f_mol_predict = open('prediction.dat', 'w')
    f_mol_predict.write("# molecule_id data frac_predict predict(%f,%f)" %
                                                   (self.bbbp_split[0], self.bbbp_split[1]))
    for i in range(len(data)):
      f_mol_predict.write("%d %f %f" % (i, data[i,0], predictions[i,1]))

      if self.calc_confusion_matrix:
        if predictions[i][1] > self.bbbp_split[0]:    
            bbb_prediction = 1
            if data[i] == 1:
                confusion_matrix[0,0] += 1
            else:
                confusion_matrix[0,1] += 1
        if predictions[i][1] < self.bbbp_split[1]:    
            bbb_prediction = 0
            if data[i] == 1:
                confusion_matrix[1,0] += 1
            else:
                confusion_matrix[1,1] += 1

        f_mol_predict.write("%d" % bbb_prediction)

      f_mol_predict.write("%s" % '\n')
    
    f_mol_predict.close()

    if self.calc_confusion_matrix:
      print( 'confusion matrix' )
      print( confusion_matrix)
  
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
  
    filename = 'shortexhibit.csv'
    #fpredict = open('bbbdata/'+filename, 'w')
    #fpredict.write('smiles,BBB\n')
    #fpredict.write('CCOc1ccc2nc(S(N)(=O)=O)sc2c1,0\n')
    #fpredict.write('O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1I,0')
    
    #fpredict.close()
    bbb_tasks, bbb_datasets, transformers = self.loadBBB(
            filename, featurizer='GraphConv', split=self.split_method, reload=False, K=4,
                ftrain=1., fvalid=0., ftest=0.)
    
    #bbb_tasks, bbb_datasets, transformers = self.loadBBBpredict(
    #        filename, featurizer='GraphConv', K=4, reload=False)
    
    model = GraphConvModel(
        len(bbb_tasks), batch_size=10, mode='classification',model_dir='models')
    
    model = model.load_from_dir('models')
    
    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")
    
    predicted_val = model.predict(bbb_datasets[0], transformers)
  
    self.writePredictions(bbb_datasets[0].y, predicted_val)

def main(argv=None):
  # Parse in command-line arguments, and create the user help instructions
  parser = argparse.ArgumentParser(description='Predict blood-brain barrier permeation '
                                               'from molecular structure.')
  parser.add_argument('-f', "--filename", type=str,
                 help='csv filename, which must be included in the bbbdata directory.')
  parser.add_argument("--split_method", type=str, choices=['index', 'random',
                                                           'scaffold', 'butina', 'task'],
                 help='Method for splitting the data into '
                      'training, testing and validation sets')
  parser.add_argument("--training_fraction", type=float, default=0.8,
                 help='fraction of total data to be split into training data, default 0.8')
  parser.add_argument("--testing_fraction", type=float, default=0.1,
                 help='fraction of total data to be split into testing data, default 0.1')
  parser.add_argument("--validation_fraction", type=float, default=0.1,
                 help='fraction of total data to be split into validation data, default 0.1')
  parser.add_argument("--confusion_matrix", action="store_true", default=True,
                 help='Calculate the confusion matrix')
  parser.add_argument("--bbbp_split", type=float, default=0.5,
                 help='Acceptable value (between 0 and 1) for the bbb prediction, for the '
                      'confusion matrix.  Default is 0.5.  If "--bbbp_split 0.6" then a '
                      'bbb prediction > 0.6 would be a positive, and a'
                      'bbb prediction < 0.4 would be a negative')
  parser.add_argument("--generate", action="store_true",
                 help='Generate and save the model')
  parser.add_argument("--predict", action="store_true",
                 help='Predict with the model')

  # Initialize the graphConvolution_BBBdata class
  bbb_gc = graphConvolution_BBBdata()
  bbb_gc.addParser(parser)

  # Tell the class everything specified in files and command line
  if (parser.parse_args().generate):
    bbb_gc.runGraphConv()

  if (parser.parse_args().predict):
    bbb_gc.getPrediction()

if __name__ == '__main__':
  sys.exit(main())

