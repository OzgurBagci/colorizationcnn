from models import *
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from imgops import *
import sys


def train_eval(params):
    """
    For params see first line of the code.

    All paths should end with '/' char.

    train_file and valid_file in which there lays down the names of the images must be in current directory.

    :param params: tuple(str, str, str, str, int, int, float, float, float)
    :return: None
    """

    gray_path, coloured_path, train_file, valid_file, num_epoch, batch_size, lr, reg, mom \
        = params

    is_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if is_cuda else 'cpu')

    cpu = torch.device('cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    with open(train_file, 'r') as tfile:
        train_li = tfile.readlines()

    train_list = []
    for train in train_li:
        train_list.append(train[:-1])

    with open(valid_file, 'r') as vfile:
        valid_li = vfile.readlines()

    valid_list = []
    for valid in valid_li:
        valid_list.append(valid[:-1])

    torch.manual_seed(7)

    cnn = ColorfulCNN().to(device)

    optimizer = torch.optim.RMSprop(cnn.parameters(), lr=lr, weight_decay=reg, momentum=mom)

    loss_func = nn.MSELoss()

    train_data = ColorfulCNN.prepare_images(read_imgs(gray_path, train_list, True)).float()
    train_expect = ColorfulCNN.prepare_images(read_imgs(coloured_path, train_list, False, True)).float()

    train_dataset = ColorfulDataset(train_data, train_expect)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    valid_data = ColorfulCNN.prepare_images(read_imgs('./test_gray/', valid_list, True)).float()

    for epoch in range(num_epoch):
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = \
                Variable(data, requires_grad=True).to(device), \
                Variable(target).to(device)
            output = cnn(data)
            loss = loss_func(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # train_out = torch.Tensor([]).to(cpu)
        # for i in range(len(train_data) // batch_size):
        #     train_o = cnn(Variable(train_data[i * batch_size:(i + 1) * batch_size]).to(device)).to(cpu).detach()
        #     train_out = torch.cat((train_out, train_o))
        # new_trains = create_rgb(gray_path, train_file, train_out, False)
        # write_npy(
        #     './outputs_' + str(batch_size) + '_' + str(lr) + '_' + str(reg) + '_' + str(mom) + '/',
        #     'output_train_' + str(epoch) + '.npy',
        #     new_trains.numpy()
        # )
    valid_out = torch.Tensor([]).to(cpu)
    for i in range(len(valid_data) // batch_size):
        valid_o = cnn(Variable(valid_data[i * batch_size:(i + 1) * batch_size]).to(device)).to(cpu).detach()
        valid_out = torch.cat((valid_out, valid_o))
    new_valids = create_rgb('./test_gray/', valid_file, valid_out, False)
    write_npy(
        './output/', 'output_test.npy', new_valids.numpy()
    )


if __name__ == '__main__':
    args = sys.argv[1:]

    if len(args) != 9:
        print('Inputs: Gray Images Path, Colored 64x64 Images Path, Train Filenames File, Validation Filenames Name, '
              'Number of Epochs at Max, Mini-batch Size, Learning Rate, L2 Regression Rate, Momentum')
        print('Try again...')
        exit()

    train_eval(
        (
            args[0],
            args[1],
            args[2],
            args[3],
            int(args[4]),
            int(args[5]),
            float(args[6]),
            float(args[7]),
            float(args[8])
        )
    )
