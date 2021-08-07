import read_dataset
import numpy as np
import training_models
import predicting_models
import evaluating_models
import plotting
import sys
import os
'''Reading data'''
features_train, labels_train, labels_base = read_dataset.read_from_folder(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset\\trainset')))

features_test, labels_test, labels_base = read_dataset.read_from_folder(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset\\testset')))


'''Training and predicting'''
models = training_models.train(features_train, labels_train) ### chose model in trainng_models.py

results, results_prob = predicting_models.predict_models(features_test, models)


'''Evaluation'''
print(evaluating_models.evaluation_all(labels_test, results)) # evaluate accuracy of each step and complet pipeline, for each subset and complete dataset



'''Plot results'''

# plotting.plot_results(94,results,labels_test, results_prob)
#
# for i in range(results.shape[0]//6 + 1):
#     start = i*6
#     try:
#         plotting.plot_results(start,results,labels_test, results_prob)
#     except IndexError:
#         print ("printed 100 examples")
