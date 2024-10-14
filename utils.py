import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def prepare_input_vector(input_vector, scaler):
    """
    Scales the input vector and converts it to a PyTorch tensor.
    """
    # Convert the input vector to a NumPy array if it isn't already
    input_array = np.array(input_vector).reshape(1, -1)

    # Scale the input vector
    scaled_vector = scaler.transform(input_array)

    # Convert the scaled vector to a PyTorch tensor
    tensor_input = torch.tensor(scaled_vector, dtype=torch.float32)

    return tensor_input


def predict_IDAC(model, scaler, input):
    tensor_input = prepare_input_vector(input, scaler)

    model.eval()
    with torch.inference_mode():
        prediction = model(tensor_input)

    return prediction.item()