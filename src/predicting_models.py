import numpy as np

'''
Predicting layout
'''

### predicts pipeline outputs
def predict_models(test_features, models):
    lista = []
    ### Washing
    subtype = models[0].predict(test_features)
    subtype = subtype.reshape(-1, 1)
    test_features = np.concatenate([test_features, subtype], axis=1)
    position = models[1].predict(test_features)
    lista.append(models[1].predict_proba(test_features))

    ### Cooking
    position = position.reshape(-1, 1)
    test_features = np.concatenate([test_features, position], axis=1)
    subtype = models[2].predict(test_features)
    subtype = subtype.reshape(-1, 1)
    test_features = np.concatenate([test_features, subtype], axis=1)
    position = models[3].predict(test_features)
    lista.append(models[3].predict_proba(test_features))

    ### Storage
    position = position.reshape(-1, 1)
    test_features = np.concatenate([test_features, position], axis=1)
    subtype = models[4].predict(test_features)
    storage_length = np.asarray([0, 60, 60, 90, 120])
    subtype1 = subtype
    subtype = np.asarray([storage_length[idx] for idx in subtype]).reshape(-1, 1)
    test_features = np.concatenate([test_features, subtype], axis=1)
    position = models[5].predict(test_features)
    lista.append(models[5].predict_proba(test_features))
    test_features[:, 10] = subtype1
    ### Adding storage prediction
    position = position.reshape(-1, 1)
    test_features = np.concatenate([test_features, position], axis=1)

    return test_features, lista

### predicts pipeline stpes independently
def predict_models_idependently(labels,test_features, models):
    lista = []
    ### Washing
    subtype = models[0].predict(test_features)
    subtype = subtype.reshape(-1, 1)
    test_features1 = np.concatenate([test_features, labels[0,:,0].reshape(-1, 1)], axis=1)
    test_features = np.concatenate([test_features, subtype], axis=1)
    position = models[1].predict(test_features1)

    ### Cooking
    test_features1 = np.concatenate([test_features, labels[0,:,1].reshape(-1, 1)], axis=1)
    position = position.reshape(-1, 1)
    test_features = np.concatenate([test_features, position], axis=1)
    subtype = models[2].predict(test_features1)
    subtype = subtype.reshape(-1, 1)
    test_features1 = np.concatenate([test_features1, labels[1,:,0].reshape(-1, 1)], axis=1)
    test_features = np.concatenate([test_features, subtype], axis=1)

    ### Storage
    position = models[3].predict(test_features1)
    position = position.reshape(-1, 1)
    test_features1 = np.concatenate([test_features1, labels[1,:,1].reshape(-1, 1)], axis=1)
    test_features = np.concatenate([test_features, position], axis=1)
    subtype = models[4].predict(test_features1)
    storage_length = np.asarray([0, 60, 60, 90, 120])
    subtype1 = subtype
    subtype = np.asarray([storage_length[idx] for idx in subtype]).reshape(-1, 1)
    test_features1 = np.concatenate([test_features1, labels[2,:,0].reshape(-1, 1)], axis=1)
    test_features = np.concatenate([test_features, subtype], axis=1)
    position = models[5].predict(test_features1)
    test_features[:, 10] = subtype1
    ### Adding storage prediction
    position = position.reshape(-1, 1)
    test_features = np.concatenate([test_features, position], axis=1)
    # print(lista)
    return test_features, lista

