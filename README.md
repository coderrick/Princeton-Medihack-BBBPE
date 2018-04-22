# Princeton-Medihack-BBBPE
![alt text](bbbpe.png)
BBBPE (Blood Brain Barrier Penetration Engine) is an application that leverages modern computational methods for drug discovery. Our platform utilizes DeepChem and PyTorch libraries to determine each small molecule's ability to permeate the blood brain barrier. 

The python module can be used via command-line arguments.
This requires installing deepchem, and all it's dependencies

calling 'python graphConvolution_BBBdata.py -h' will give information
about the command-line arguments
example python call:
     'python graphConvolution_BBBdata.py --filename 'finaldata.csv' 
                                         --split_method='index'
                                         --training_fraction 0.6 
                                         --testing_fraction 0.2
                                         --validation_fraction 0.2
                                         --confusion_matrix
                                         --bbbp_split 0.5 --generate'

Note: any data for training or predicting must be located in the 'bbbdata' directory.
Note: data must be in the csv format, and include collumn headers: 'smiles' and 'bbb' 
