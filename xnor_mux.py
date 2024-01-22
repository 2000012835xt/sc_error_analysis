# this file is to simulate the error distribution of stochastic computing

"""Assume weight follows normal distribution with sigma_w, and activation follows uniform distribution
   bipolar MUL with XNOR and ADD with MUX
"""

import numpy as np
import matplotlib as plt
import math
import xlrd
import xlwt
import pdb
import torch

wb = xlwt.Workbook()
ws = wb.add_sheet('0117_SC_MAC')

data=torch.load("hook_data_fp.pt")

weight=torch.clamp(data['weight'],-1,1)
activation=torch.clamp(data['input'],0,1)
output=data['output']


weight_1d=weight.flatten()
activation_1d=activation.flatten()


repeat = 200
BSL = [128, 256, 512, 1024, 2048, 4096]
CHL = [16, 32, 64]
# sgw = [0.1, 0.2, 0.4]
row = 0
precision = 16
for j in range(len(BSL)):
    for k in range(len(CHL)):
        # for l in range(len(sgw)):

        bitstream_length = BSL[j]
        channel = CHL[k]
        accumulation_width = 3 * 3 * channel

        print(row)
        print("BSL", bitstream_length)
        print("NUM", accumulation_width)

        output_value_vector = np.zeros(repeat)
        output_value_from_sc_vector = np.zeros(repeat)
        for i in range(repeat):
            if i % 20 == 0:
                print(i)
            # value sampling
            weight_indices=torch.randperm(weight_1d.size(0))
            activation_indices=torch.randperm(activation_1d.size(0))

            weights_value = np.round(precision * np.clip(weight_1d[weight_indices[0:accumulation_width]].cpu().detach().numpy(), -1, 1)) / precision
            activation_value = np.round(precision * activation_1d[activation_indices[0:accumulation_width]].cpu().numpy()) / precision
            # bitstream generation

            weights_prob = (weights_value + 1) / 2
            activation_prob = (activation_value + 1) / 2
            weights_sequence = np.array(
                [np.random.choice([0, 1], size=bitstream_length, replace=True, p=[1 - prob, prob]) for prob in
                    weights_prob])
            activation_sequence = np.array(
                [np.random.choice([0, 1], size=bitstream_length, replace=True, p=[1 - prob, prob]) for prob in
                    activation_prob])

            # output result
            output_value = np.dot(weights_value, activation_value) / accumulation_width
            output_value_vector[i] = output_value

            # output bitstream
            product = (weights_sequence == activation_sequence).astype(
                int)  # XNOR to multiply weight with activation
            select = np.random.randint(0, accumulation_width, size=bitstream_length)
            output_bitstream = product[select, np.arange(bitstream_length)]

            # output_bitstream=(np.sum(product,axis=0)>accumulation_width/2).astype(int)  #apc
            # pdb.set_trace()

            output_prob = np.sum(output_bitstream) / bitstream_length
            output_value_from_sc = output_prob * 2 - 1
            output_value_from_sc_vector[i] = output_value_from_sc

        MSE = np.mean(np.square(output_value_vector - output_value_from_sc_vector))
        MAE = np.mean(np.abs(output_value_vector - output_value_from_sc_vector))
        RMSE= np.mean(np.square(np.abs(output_value_from_sc_vector-output_value_vector)/(output_value_vector+0.0001)))
        row = row + 1
        # print(output_value_vector)
        # print(output_value_from_sc_vector)
        print("MSE", MSE)
        print("MAE", MAE)
        print("RMSE",RMSE)
        AVG = np.mean(np.abs(output_value_vector))
        print("AVG", AVG)
        ws.write(row, 2, bitstream_length)
        ws.write(row, 3, accumulation_width)
        ws.write(row, 4, MSE)
        ws.write(row, 5, MAE)
        ws.write(row, 6, AVG)
        ws.write(row, 7, RMSE)
        print()
        wb.save('0117_SC_MAC.xls')
        print('\n')

