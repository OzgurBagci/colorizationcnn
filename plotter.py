import numpy as np
import matplotlib.pyplot as plt
import sys
from evaluate import main
from io import StringIO


def plot_tests(filename, outname):
    """
    Works if number of epochs are 25. Change it if you want more or less.

    7 and 12 is for float with 5 decimal points, change it to 4 and 9 if you are using float with 2 decimal points in
    'evaluate.py'. And change 25 to batch size. (While you are reading this, I probably had changed it but written it
    just in case.)



    :param filename: str
    :param outname: str
    :return: None
    """
    with open(filename) as log:
        lines = log.readlines()
    nlines = list(map(lambda x: x[:-1], lines))
    nlines = list(filter(lambda x: x[7:12] == '/1.00', nlines))     # 7 and 12 may change according to output decimals.
    i = 0
    datas = []
    data = []
    while True:
        if len(nlines) == i + 1:
            break
        while True:
            i += 1
            if (i + 1) // 25 == (i + 1) / 25:   # Change 25 to Batch Size
                datas.append(np.array(data))
                data = []
                break
            data.append(float(nlines[i][:7]))   # 7 may change according to output decimals.
    llines = list(map(lambda x: x[:-1], lines))
    llines = list(filter(lambda x: x[:27] == 'In directory: ./outputs_25_', llines))    # 25 in str is batch size.
    label_lr = list(map(lambda x: 'Learning Rate: ' + x[27:x.index('_', 27)], llines))
    label_wd = list(map(lambda x: 'Weight Decay: ' + x[x.index('_', 27) + 1:
        x.index('_', x.index('_', x.index('_', 27) + 1))], llines))
    label_mo = list(map(lambda x: 'Momentum: ' + x[x.index('_', x.index('_', x.index('_', 27) + 1)) + 1:-1], llines))
    labels = list(zip(label_lr, label_wd, label_mo))
    labels = list(map(lambda x: x[0] + ', ' + x[1] + ', ' + x[2], labels))
    for i in range(len(datas)):
        plt.plot(datas[i])
    lgd = plt.legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('Epoch No')
    plt.ylabel('Success Rate of Validation Set')
    plt.savefig(outname, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_final(dirr, outname):
    out = StringIO()
    sys.stdout = out
    li = []
    for i in range(40):     # Change 75 if there are different number of epochs.
        main((dirr + 'output_train_' + str(i) + '.npy', 'train.txt'))   # Change names for train and valid.
    for line in out.getvalue().splitlines():
        li.append(line[:8])
    plt.plot(np.array(li))
    plt.xlabel('Epoch No')
    plt.ylabel('Success Rate of Validation Set')
    plt.locator_params(nbins=8)
    plt.savefig(outname)
    plt.close()


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3:
        print('Insufficient or more than usual'
              ' arguments. Args: Operation Type (test, final), Filename/Directory with Path, Output Image Filename')
        exit()
    op = args[0]
    if op == 'test':
        plot_tests(args[1], args[2])
    elif op == 'final':
        plot_final(args[1], args[2])
    else:
        print('Give a proper operation type!')
