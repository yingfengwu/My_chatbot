import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
hidden_size = 256

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10




