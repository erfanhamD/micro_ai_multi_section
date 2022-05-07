import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

class Lift_base_network(nn.Module):
    def __init__(self):
        super(Lift_base_network, self).__init__()
        self.block = nn.Sequential(
        nn.Linear(5, 20),
        nn.Tanh(),
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 2)
        )
    def forward(self, x):
        x = self.block(x)
        return x

def preprocess(Data):
    # Data = pd.read_csv(Data_addr, header = None, delimiter=',')
    Data[4] = Data[4]/Data[0]
    scaler = pickle.load(open('/Users/venus/AI_lift/multi_section/model/x_mm_scaler.pkl', 'rb'))
    Data = scaler.transform(Data)
    return Data

def inference(Data, model):
    Data_torch = torch.from_numpy(Data)
    Cl = model(Data_torch.float())
    scaler = pickle.load(open('/Users/venus/AI_lift/multi_section/model/y_mm_scaler.pkl', 'rb'))
    Cl = Cl.cpu().detach().numpy()
    Cl_map = scaler.inverse_transform(Cl)
    return Cl_map

def main():
    input_data_addr = '/Users/venus/AI_lift/multi_section/data/z-3-50-30.csv'
    model_address = '/Users/venus/AI_lift/multi_section/model/model_state_dict_3Apr_mm'
    model = Lift_base_network()
    model.load_state_dict(torch.load(model_address))
    Data = preprocess(input_data_addr)
    Cl_map = inference(Data, model)
    return Cl_map