import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# matplotlib.rcParams.update({'font.size': 8})
def plot_grid(elem, ax, elem_labels=None, res_prob=None, i=None):
    plt.subplot(ax)
    plt.plot([0, elem[0]], [0, 0], 'k', linewidth=3)
    if (elem[1] != -1):
        plt.scatter(elem[1], 0, s=100, marker="v", c="k")  # ventilacija
    if (elem[2] != -1):
        plt.scatter(elem[2], 0, s=100, marker="*", c="k")  # kanalizacija
    if (elem[3] != -1):
        plt.scatter(elem[3], 0, s=100, marker="p", c="k")  # prozor
    if (elem[4] != -1):
        plt.scatter(elem[4], 0, s=100, marker="<", c="k")
    if (elem[5] != -1):
        plt.scatter(elem[5], 0, s=100, marker=">", c="k")
    # washing plavo, cooking crveno, storage zeleno
    washing = [0, 30, 60, 120]
    cooking = [0, 30, 60, 90]
    storage = [0, 60, 60, 90, 120]

    def start(bar, size):
        if bar < 20:

            return bar * 30
        else:
            return (size - (bar - 20) * 30)

    ### ovo utepuvanje ovde je zato sto moze da se desi da su npr 3 i 22 klasa zapravo isti pocetak, npr kad je duzina kuhine 150
    w_start = start(elem[7], elem[0])
    c_start = start(elem[9], elem[0])
    s_start = start(elem[11], elem[0])
    w_end = w_start + elem[6]
    c_end = c_start + elem[8]
    s_end = s_start + storage[elem[10]]
    plt.plot([w_start, w_end], [0, 0], 'b', linewidth=3)
    plt.plot([c_start, c_end], [0, 0], 'r', linewidth=3)
    if elem[10] != 0:
        plt.plot([s_start, s_end], [0, 0], 'g', linewidth=3)
    if elem_labels is not None:  # ove labele iscrtavam samo za predikciju, kad crtam iz trening skupa onda nema ovih labela
        w_start_labels = start(elem_labels[0, 1], elem[0])
        c_start_labels = start(elem_labels[1, 1], elem[0])
        s_start_labels = start(elem_labels[2, 1], elem[0])
        plt.title(  # str(elem[0])  + '|' + \
            str(w_start) + '-' +\
            str(w_end) + \
            ("T" if elem_labels[0, 0] == elem[6] else 'F') + ("T" if w_start == w_start_labels else 'F') + \
                # str(int(max(res_prob[0][i]) * 100)) + \
            '|' + str(c_start) + '-' + str(c_end) + \
            ("T" if elem_labels[1, 0] == elem[8] else 'F') + ("T" if c_start == c_start_labels else 'F') + \
                # str(int(max(res_prob[1][i]) * 100)) + \
            '|' + str(s_start) + '-' + str(s_end) + \
            ("T" if elem_labels[2, 2] == elem[10] else 'F') + (
                "F" if (s_start != s_start_labels) & (elem[10] != 0) else 'T'), fontsize=19) # + str(int(max(res_prob[2][i]) * 100)) \

    else:
        plt.title(str(elem[0]) + \
                  '|' + str(w_start) + '-' + str(w_end) + \
                  '|' + str(c_start) + '-' + str(c_end) + \
                  '|' + str(s_start) + '-' + str(s_end) \
                  , fontsize=19)


def plot_results(start, results, test_labels, results_prob):
    plt.clf()
    plt.figure(figsize=(10, 10))
    idd = 0
    for i in range(start, start + 6):
        axx = idd + 321
        plot_grid(results[i] - [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], axx, test_labels[:, i, :], results_prob, i)
        idd += 1

    plt.savefig(str(start))
    plt.close()

def plot_examples(start, features, labels):
    plt.clf()
    plt.figure(figsize=(10, 10))
    idd = 0
    features = np.concatenate([features, labels[0, :, :2], labels[1, :, :2], labels[2, :, 1:][:, [1, 0]]], axis=1)
    for i in range(start, start + 6):
        axx = idd + 321
        plot_grid(features[i] - [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], axx)
        idd += 1
    plt.show()
    plt.close()

