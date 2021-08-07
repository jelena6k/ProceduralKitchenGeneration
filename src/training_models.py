import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

idx = 0
def plot_feature_importances(model,training_features):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = range(training_features.shape[1])
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(training_features.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(training_features.shape[1]), indices, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-1, training_features.shape[1]])
    plt.ylim(0, 1)
    global idx
    idx = idx + 1
    plt.savefig("g" + str(idx))
    plt.close()
    print(indices)


def subtype_train(training_features, training_labels):

    '''Uncomment desired model'''

    # model = MLPClassifier(max_iter = 5000,hidden_layer_sizes=(100,100,100), random_state = 7, activation = 'tanh')
    model = RandomForestClassifier(n_estimators= 500, max_features= 'auto', max_depth= 6)
    # model = SVC(C= 1, gamma= 0.1, kernel= 'linear')
    #model = LogisticRegression(random_state=0)
    # model = GaussianNB()
    # model = AdaBoostClassifier(n_estimators=100, random_state=0)
    # model = tree.DecisionTreeClassifier(random_state=0)
    # model = ExtraTreesClassifier(n_estimators=100,
    #                               random_state=0)

    model.fit(training_features, training_labels)

    return model


def position_train(training_features, training_labels):
    '''Uncomment desired model'''

    model = RandomForestClassifier(n_estimators=500, max_depth=13,random_state = 7,max_features='sqrt')
    # model = MLPClassifier(max_iter = 5000,hidden_layer_sizes=(100,100,100), random_state = 7, activation = 'tanh')
    # model = SVC(C= 1, gamma= 0.1, kernel= 'linear')
    #model = LogisticRegression(random_state=0, solver='liblinear')
    # model = GaussianNB()
    # model = AdaBoostClassifier(n_estimators=100, random_state=0)

    # model = ExtraTreesClassifier(n_estimators=100, random_state=7)
    # model = tree.DecisionTreeClassifier(random_state=0)

    model.fit(training_features, training_labels)
    #plot feature importance for random forest
    #plot_feature_importances(model, training_features)
    return model


'''
Training subtype and position models
'''
def subtype_position_train(training_features, training_labels, storage=False):
    # zato sto storage ima dva tipa iste  duzine, pa onda ne mogu da uzmem duzinu kao labelu klase
    if storage:
        model_subtype = subtype_train(training_features, training_labels[:, 2])
    else:
        model_subtype = subtype_train(training_features, training_labels[:, 0])

    add = training_labels[:, 0].reshape(-1, 1)
    training_features = np.concatenate([training_features, add], axis=1)
    if storage:
        mask = training_labels[:, 0] != 0
        training_features, training_labels = training_features[mask], training_labels[:, 1][mask]
        model_position = position_train(training_features, training_labels)
    else:
        model_position = position_train(training_features, training_labels[:, 1])

    return model_subtype, model_position


'''
Training pipeline
'''
def train(training_features, training_labels):
    model_subtype_washing, model_position_washing = subtype_position_train(training_features, training_labels[0])
    training_features = np.concatenate([training_features, training_labels[0, :, :2]], axis=1)
    model_subtype_cooking, model_position_cooking = subtype_position_train(training_features, training_labels[1])

    training_features = np.concatenate([training_features, training_labels[1, :, :2]], axis=1)
    model_subtype_storage, model_position_storage = subtype_position_train(training_features, training_labels[2], True)

    return [model_subtype_washing, model_position_washing, model_subtype_cooking, \
            model_position_cooking, model_subtype_storage, model_position_storage]

