from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error
import numpy as np


def start(bar, size): #convert  position class to position from beggining in cm
    bar = np.copy(bar)
    b_less = bar < 20
    bar[b_less] = bar[b_less] * 30
    bar[~b_less] = (size[~b_less]-1 - (bar[~b_less] - 20) * 30)
    return bar

# evaluate pipeline steps
def evaluation(test_labels, results):

    w_subtype = accuracy_score(test_labels[0, :, 0], results[:, 6])
    w_start = start(results[:, 7], results[:, 0])
    w_start_labels = start(test_labels[0, :, 1], results[:, 0])
    w_pos = accuracy_score(w_start, w_start_labels)

    c_subtype = accuracy_score(test_labels[1, :, 0], results[:, 8])
    c_start = start(results[:, 9], results[:, 0])
    c_start_labels = start(test_labels[1, :, 1], results[:, 0])
    c_pos = accuracy_score(c_start, c_start_labels)

    s_subtype = accuracy_score(test_labels[2, :, 2], results[:, 10])
    s_start = start(results[:, 11][test_labels[2, :, 0] != 0], results[:, 0][test_labels[2, :, 0] != 0])
    s_start_labels = start(test_labels[2, :, 1][test_labels[2, :, 0] != 0], results[:, 0][test_labels[2, :, 0] != 0])
    s_pos = accuracy_score(s_start, s_start_labels)

    return ([w_subtype, w_pos, c_subtype, c_pos, s_subtype, s_pos])

# evaluate pipeline steps and whole pipeline
def evaluation_all(test_labels, results):

    # washing accuracy
    w_subtype = accuracy_score(test_labels[0, :, 0], results[:, 6])
    w_start = start(results[:, 7], results[:, 0])
    w_start_labels = start(test_labels[0, :, 1], results[:, 0])
    w_pos = accuracy_score(w_start, w_start_labels)
    washing_pos = w_start== w_start_labels
    washing_subt =  test_labels[0, :, 0]== results[:, 6]
    washing =np.logical_and(washing_pos,washing_subt)

    #cooking accuracy
    c_subtype = accuracy_score(test_labels[1, :, 0], results[:, 8])
    c_start = start(results[:, 9], results[:, 0])
    c_start_labels = start(test_labels[1, :, 1], results[:, 0])
    c_pos = accuracy_score(c_start, c_start_labels)
    cooking_pos = c_start ==  c_start_labels
    cooking_sub = test_labels[1, :, 0] == results[:, 8]
    cooking = np.logical_and(cooking_pos,cooking_sub)

    #storage accuracy
    s_subtype = accuracy_score(test_labels[2, :, 2], results[:, 10])
    s_start = start(results[:, 11][test_labels[2, :, 0] != 0], results[:, 0][test_labels[2, :, 0] != 0])
    s_start_labels = start(test_labels[2, :, 1][test_labels[2, :, 0] != 0], results[:, 0][test_labels[2, :, 0] != 0])
    s_pos = accuracy_score(s_start, s_start_labels)
    storage_pos =   start(results[:, 11] , results[:, 0] ) == start(test_labels[2, :, 1] , results[:, 0] )
    storage_pos[ results[:, 10] == 0] = True
    storage_sub = test_labels[2, :, 2]== results[:, 10]
    storage = np.logical_and(storage_sub,storage_pos)

    wc = np.logical_and(washing,cooking)
    ecs = np.logical_and(wc,storage)
    k =sum(ecs)

    ## subsets accuracy

    c_subtype_0 = accuracy_score(test_labels[0, :10, 0], results[:10, 6])
    c_pos_0 = accuracy_score(w_start[:10], w_start_labels[:10])
    c_subtype_1 = accuracy_score(test_labels[0, 10:30, 0], results[10:30, 6])
    c_pos_1 = accuracy_score(w_start[10:30], w_start_labels[10:30])
    c_subtype_2 = accuracy_score(test_labels[0, 30:60, 0], results[30:60, 6])
    c_pos_2 = accuracy_score(w_start[30:60], w_start_labels[30:60])
    c_subtype_3 = accuracy_score(test_labels[0, 60:, 0], results[60:, 6])
    c_pos_3 = accuracy_score(w_start[60:], w_start_labels[60:])
    c_gourps = [c_subtype_0,c_pos_0, c_subtype_1, c_pos_1,c_subtype_2,c_pos_2,c_subtype_3,c_pos_3]
    print("washing accuracy for each subset")
    print(c_gourps)

    c_subtype_0 = accuracy_score(test_labels[1, :10, 0], results[:10, 8])
    c_pos_0 = accuracy_score(c_start[:10], c_start_labels[:10])
    c_subtype_1 = accuracy_score(test_labels[1, 10:30, 0], results[10:30, 8])
    c_pos_1 = accuracy_score(c_start[10:30], c_start_labels[10:30])
    c_subtype_2 = accuracy_score(test_labels[1, 10:30, 0], results[10:30, 8])
    c_pos_2 = accuracy_score(c_start[30:60], c_start_labels[30:60])
    c_subtype_3 = accuracy_score(test_labels[1, 60:, 0], results[60:, 8])
    c_pos_3 = accuracy_score(c_start[60:], c_start_labels[60:])
    c_gourps = [c_subtype_0,c_pos_0, c_subtype_1, c_pos_1,c_subtype_2,c_pos_2,c_subtype_3,c_pos_3]
    print("cooking accuracy for each subset")
    print(c_gourps)

    c_subtype_0 = accuracy_score(test_labels[2, :10, 2], results[:10, 10])
    c_pos_0 = accuracy_score(s_start[:10], s_start_labels[:10])
    c_subtype_1 = accuracy_score(test_labels[2, 10:30, 2], results[10:30, 10])
    c_pos_1 = accuracy_score(s_start[10:30], s_start_labels[10:30])
    c_subtype_2 = accuracy_score(test_labels[2, 30:60, 2], results[30:60, 10])
    c_pos_2 = accuracy_score(s_start[30:60], s_start_labels[30:60])
    c_subtype_3 = accuracy_score(test_labels[2, 60:, 2], results[60:, 10])
    c_pos_3 = accuracy_score(s_start[60:], s_start_labels[60:])
    c_gourps = [c_subtype_0,c_pos_0, c_subtype_1, c_pos_1,c_subtype_2,c_pos_2,c_subtype_3,c_pos_3]
    print("storage accuracy for each subset")
    print(c_gourps)

    ### joint accuracy of subsets
    print("joint accuracy for each subset")
    print(sum(ecs[:10]) / 10, sum(ecs[10:30]) / 20, sum(ecs[30:60]) / 30, sum(ecs[60:]) / 40)
    ### independent accuracy
    print("accuracy for each step")
    print([w_subtype, w_pos, c_subtype, c_pos, s_subtype, s_pos])
    print("joint accuracy",k)
    return (k, [w_subtype, w_pos, c_subtype, c_pos, s_subtype, s_pos])

