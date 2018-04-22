#!/usr/bin/env python
"""test_bbbPredict

 Predict blood-brain barrier permeation
 from molecular positions
 							                                    
"""
from bbbPredict import bbbPredict

predict = bbbPredict()
print("predicted value: ", predict.getPrediction())
