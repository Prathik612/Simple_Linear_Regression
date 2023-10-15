import torch
from torch import nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt

#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using: {device}")

salary_data = pd.read_csv('AI\pyTorch_Course\linear_regression\Salary_dataset.csv')
salary_data = salary_data.drop(columns=salary_data.columns[0], axis=1)

x_data = torch.tensor(salary_data["YearsExperience"])
y_data = torch.tensor(salary_data["Salary"])

#salary_tensor = torch.tensor(salary_data)

print(y_data)


class salaryPredictionv0(nn.Module):

    def __init__(self):
        super().__init__()
