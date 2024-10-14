import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn

import joblib

from model_architecture import IDACModelV0
from utils import predict_IDAC



if __name__ == '__main__':
    # scaler
    scaler = joblib.load('./models/IDACModelScaler.pkl')

    # model
    INPUT_SHAPE = 13
    HIDDEN_UNITS = 40
    OUTPUT_SHAPE = 1
    ACTIVATION = 'sigmoid'

    model = IDACModelV0(INPUT_SHAPE, HIDDEN_UNITS, OUTPUT_SHAPE, ACTIVATION)
    model.load_state_dict(torch.load('./models/IDACModel.pth', weights_only=True))

    # Example usage
    input_vector = [561.6, 5226120, 0.000230947,
                    0.268, 2.94, 98.95916, 652.5,
                    2777000, 0.000497, 0.5963,
                    1.649998201, 130.22792, 298.2]
    
    target = 0.985816795  # expreimental value of ln(gamma)

    prediction = predict_IDAC(model, scaler, input_vector)

    print('-' * 50)
    print(f'Experimental value for ln(inf dilution activity coeff): {target: 8.4f}')
    print(f'Predicted value for ln(inf dilution activity coeff):    {prediction: 8.4f}')
    print('-' * 50)
