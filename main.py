import sys
import warnings
import itertools
from train import train_eval
from evaluate import main
import numpy as np


def find_params(start, stop, step, step_type, adjust, others):
    """
    For working on model and finding better parameters.

    :param start: float
    :param stop: float
    :param step: float
    :param step_type: str
    :param adjust: str
    :param others: tuple(float, float)
    :return: None
    """
    numb_epochs = 25    # Change the epoch number for more or less epochs.
    batch_size = 25  # Change the value if training set size smaller.
    if step_type == 'lin':
        multiplier1 = 1 / start
        multiplier2 = 1 * multiplier1 / stop
        nstart = int(start * multiplier1 * multiplier2)
        nstop = int(stop * multiplier1 * multiplier2)
        nstep = int(step * multiplier1 * multiplier2)
        divider = multiplier1 * multiplier2
        numbers = range(nstart, nstop, nstep)
        numbers = np.array(numbers).astype(float)
        numbers /= divider
        numbers = list(numbers)
    elif step_type == 'exp':
        numbers = []
        acc = start
        while True:
            numbers.append(acc)
            acc /= step
            if acc <= stop:
                break
    else:
        raise Exception('Give a proper Step Type!')
    if adjust == 'lr':
        combs = itertools.product(numbers, [others[0]], [others[1]])
    elif adjust == 'wd':
        combs = itertools.product([others[0]], numbers, [others[1]])
    elif adjust == 'mom':
        combs = itertools.product([others[0]], [others[1]], numbers)
    else:
        raise Exception('Give a proper str for Adjust Type!')
    for comb in combs:
        train_eval(
            (
                './gray/',
                './color_64/',
                'train.txt',
                'valid.txt',
                numb_epochs,
                batch_size,
                comb[0],
                comb[1],
                comb[2]
            )
        )
        directory = './outputs_' + str(batch_size) + '_' + str(comb[0]) + '_' + str(comb[1]) + '_' + \
                    str(comb[2]) + '/'
        print('\n---------------------')
        print('---------------------')
        print('---------------------\n')
        print('In directory: ' + directory)
        print('Loss for ' +
              str(numb_epochs) +
              ' epochs in order from first epoch to last.'
              # ' epochs in order of 1 for train sample and 1 for validation from first epoch to last.'
              )
        for i in range(numb_epochs):
            # main((directory + 'output_train_' + str(i) + '.npy', 'train.txt'))
            main((directory + 'output_valid_' + str(i) + '.npy', 'valid.txt'))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 7:
        warnings.warn(
            'Arguments needed: Starting Rate, Ending Rate, Step, Step Type ("exp", "lin"), '
            'Rate to be Adjusted("lr, "wd", "mom"), Values of Others(in order of ["lr, "lin", "mom"] - Adjusted Rate.'
                     )
        warnings.warn(
            'If step is 10 for example and the Step Type is lin, Rates are range(start, end, (end - start) / step=10).'
        )
        warnings.warn(
            'Example: "python main.py 1e-1 1e-5 10 exp lr 1e-2 1e-3"'
        )
        exit()
    find_params(float(args[0]), float(args[1]), float(args[2]), args[3], args[4], (float(args[5]), float(args[6])))
