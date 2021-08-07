import read_dataset
import numpy as np
import training_models
import predicting_models
import evaluating_models
import os

features, labels, labels_base = read_dataset.read_from_folder(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset\\trainset')))

np.random.seed(1)
idx = np.random.permutation(features.shape[0])
features, labels = features[idx], labels[:, idx, :]
k_fold = []
fold_num = 5
length = len(features)
partition_len = length//fold_num
for it in range(fold_num):
    ### shuffle
    start = it*partition_len
    end = start + partition_len if start + partition_len<length else length
    ### Training
    train_idx =  list(range(0,start)) + list(range(end,length))
    training_labels = labels[:, train_idx, :]
    training_features = features[train_idx]
    models = training_models.train(training_features, training_labels)     ### choose model in training_models
    ### Testing
    test_features = features[start:end]
    test_labels = labels[:, start:end, :]

    results,_ = predicting_models.predict_models_idependently(test_labels,test_features, models)
    k_fold.append(evaluating_models.evaluation(test_labels, results))

k_fold = np.asarray(k_fold)
print(k_fold.mean(axis=0))


