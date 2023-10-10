import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

from torch.autograd import Variable
import math
from torch.utils.data import DataLoader
import os
import matlab
import matlab.engine

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ROI = 5
sub = 50
subLen = 200


def _FirstOderLR(x):
    x = np.mat(x)
    x = x.T
    N = x.shape[0]
    T = x.shape[1]
    temp1 = np.mat(np.ones((1, T)))
    temp2 = (np.mean(x.T, axis=0)).T
    X = x - temp2 * temp1
    eps = 0.0000000000000002220446049250313
    temp3 = (np.std(X.T, axis=0, ddof=1) + eps).T * temp1
    X = np.divide(X, temp3)
    Cov = np.cov(X)
    temp4 = (X * np.tanh(X.T) - np.tanh(X) * X.T)
    LR = np.multiply(Cov, temp4) / T
    print(LR, "\n")
    return LR


def _h(W):
    d = ROI
    M = np.eye(d) + W * W / d
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d
    G_h = E.T * W * 2
    return h, G_h


class RNN(nn.Module):
    def __init__(self, inputsize, hiddensize, outsize):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=inputsize,
            hidden_size=hiddensize,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hiddensize, outsize)

    def forward(self, x):
        r_out, h_state = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self, inputsize):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=inputsize,
                out_channels=16,
                kernel_size=101,
                stride=1,
                padding=50,
            ),
            nn.ReLU(),
            nn.MaxPoold(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Convd(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, sizes, **kwargs):
        super(Discriminator, self).__init__()
        self.sht = kwargs.get('shortcut', False)
        activation_function = kwargs.get('activation_function', nn.Sigmoid)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if dropout != 0.:
                layers.append(nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        return self.layers(x)


class FilterGenerator(nn.Module):

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Initialize a generator."""
        super(FilterGenerator, self).__init__()
        gpu = kwargs.get('gpu', True)
        gpu_no = kwargs.get('gpu_no', 0)

        activation_function = kwargs.get('activation_function', nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        self.lstm = nn.LSTM(sizes[0], sizes[0], 2)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

        self._filter = th.ones(1, sizes[0])
        for i in zero_components:
            self._filter[:, i].zero_()

        self._filter = Variable(self._filter, requires_grad=False)
        self.fs_filter = nn.Parameter(self._filter.data)

        if gpu:
            self._filter = self._filter.cuda(gpu_no)

    def forward(self, x):
        rows, cols = x.size()
        x = x.view(1, rows, cols)
        h = th.randn(2, rows, cols)
        c = th.randn(2, rows, cols)
        x = x.cuda()
        h = h.cuda()
        c = c.cuda()
        x, h = self.lstm(x, (h, c))
        x = x.view(rows, cols)
        return self.layers(x * (self._filter * self.fs_filter).expand_as(x))


class Generators(nn.Module):

    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        super(Generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]

        gpu = kwargs.get('gpu', True)
        gpu_no = kwargs.get('gpu_no', 0)

        rows, self.cols = data_shape
        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        if gpu:
            self.noise = [i.cuda(gpu_no) for i in self.noise]

        self.blocks = th.nn.ModuleList()
        for i in range(self.cols):
            self.blocks.append(FilterGenerator(
                [self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        """Feed-forward the model."""
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables


def run_GANEC(df_data, skeleton=None, **kwargs):
    gpu = kwargs.get('gpu', True)
    gpu_no = kwargs.get('gpu_no', 0)

    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)

    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    rou = kwargs.get('rou', 1)
    maxp = kwargs.get('rou', 2)
    dnh = kwargs.get('dnh', None)

    d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {}"
    try:
        list_nodes = list(df_data.columns)
        df_data = (df_data[list_nodes]).as_matrix()
    except AttributeError:
        list_nodes = list(range(df_data.shape[1]))
    data = df_data.astype('float32')
    data = th.from_numpy(data)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()

    if skeleton is not None:
        zero_components = [[] for i in range(cols)]
        for i, j in zip(*((1 - skeleton).nonzero())):
            zero_components[j].append(i)
    else:
        zero_components = [[i] for i in range(cols)]
    sam = Generators((rows, cols), zero_components, batch_norm=True, **kwargs)

    activation_function = kwargs.get('activation_function', th.nn.Tanh)

    try:
        del kwargs["activation_function"]
    except KeyError:
        pass
    discriminator_sam = Discriminator(
        [cols, dnh, dnh, 1], batch_norm=True,
        activation_function=th.nn.Sigmoid,
        activation_argument=None, **kwargs)
    kwargs["activation_function"] = activation_function

    if gpu:
        sam = sam.cuda(gpu_no)
        discriminator_sam = discriminator_sam.cuda(gpu_no)
        data = data.cuda(gpu_no)

    criterion = th.nn.BCEWithLogitsLoss()
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = th.optim.Adam(discriminator_sam.parameters(), lr=lr_disc)

    true_variable = Variable(th.ones(batch_size, 1), requires_grad=False)
    false_variable = Variable(th.zeros(batch_size, 1), requires_grad=False)
    causal_filters = th.zeros(data.shape[1], data.shape[1])

    if gpu:
        true_variable = true_variable.cuda(gpu_no)
        false_variable = false_variable.cuda(gpu_no)
        causal_filters = causal_filters.cuda(gpu_no)

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(train_epochs + test_epochs):
        for i_batch, batch in enumerate(data_iterator):
            batch = Variable(batch)
            batch_vectors = [batch[:, [i]] for i in range(cols)]

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            generated_variables = sam(batch)
            disc_losses = []
            gen_losses = []

            for i in range(cols):
                generator_output = th.cat([v for c in [batch_vectors[: i], [
                    generated_variables[i]],
                    batch_vectors[i + 1:]] for v in c], 1)

                disc_output_detached = discriminator_sam(
                    generator_output.detach())
                disc_output = discriminator_sam(generator_output)
                disc_losses.append(
                    criterion(disc_output_detached, false_variable))

                gen_losses.append(criterion(disc_output, true_variable))

            true_output = discriminator_sam(batch)
            adv_loss = sum(disc_losses) / cols + \
                       criterion(true_output, true_variable)
            gen_loss = sum(gen_losses)

            adv_loss.backward()
            d_optimizer.step()

            filters = th.stack(
                [i.fs_filter[0, :-1].abs() for i in sam.blocks], 1)

            l1_reg = 0.5 * regul_param * filters.sum() * math.log(l)
            A = filters.data
            W = A.numpy()
            h, G_h = _h(W)
            dag = 0.5 * rou * (h * h)
            loss = gen_loss + l1_reg + dag

            if verbose and epoch % 200 == 0 and i_batch == 0:
                string_txt = str(i) + " " + d_str.format(epoch,
                                                         adv_loss.cpu().item(),
                                                         gen_loss.cpu(
                                                         ).item() / cols,
                                                         l1_reg.cpu().item())
                print(string_txt)
                print(causal_filters)
                print(type(causal_filters))
            loss.backward()

            if epoch > train_epochs:
                causal_filters.add_(filters.data)
            g_optimizer.step()

    tp = causal_filters.div_(10).cpu().numpy()

    return causal_filters.div_(10).cpu().numpy()


class GANEC(object):
    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1):
        super(GANEC, self).__init__()
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize

    def Train_RUN(self, data, graph=None):
        list_out = [run_GANEC(data, skeleton=graph,
                              lr_gen=self.lr, lr_disc=self.dlr,
                              regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                              train_epochs=self.train,
                              test_epochs=self.test, batch_size=self.batchsize)]

        return list_out

    def Test_RUN(self, data):
        list_out = [run_GANEC(data, skeleton=None,
                              lr_gen=self.lr, lr_disc=self.dlr,
                              regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                              train_epochs=self.train,
                              test_epochs=self.test, batch_size=self.batchsize)]
        return list_out


if __name__ == '__main__':
    GAN = GANEC(lr=0.01, dlr=0.01, l1=0.1, nh=100, dnh=100, train_epochs=400, test_epochs=1600)
    eng = matlab.engine.start_matlab()
    [NC, TP] = eng.nPSE()
    p_txt = "/sim1.txt"
    sim = np.array(pd.read_csv(p_txt, sep="\t", header=None))
    start = 0
    end = TP[0]
    l = end-start
    data = np.zeros((sub*l, ROI))
    for i in range(0, sub):
        data[i*l:(i+1)*l, :] = sim[i*subLen+start:i*subLen+end, :]
    GAN.Test_RUN(data)
