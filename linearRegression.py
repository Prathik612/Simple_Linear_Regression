import torch
from torch import nn
from torch import optim
import pandas as pd
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}\n")

salary_data = pd.read_csv('Salary_dataset.csv')
salary_data = salary_data.drop(columns=salary_data.columns[0], axis=1)

X_data = torch.tensor(salary_data["YearsExperience"])
X_data = X_data.unsqueeze(dim=1).float()
y_data = torch.tensor(salary_data["Salary"])
y_data = y_data.unsqueeze(dim=1).float()

train_split = int(0.8 * len(X_data))

X_train, y_train = X_data[:train_split], y_data[:train_split]
X_test, y_test = X_data[train_split:], y_data[train_split:]

def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):

    plt.scatter(train_data, train_labels, s=5, c='b', label="Training Data")
    plt.scatter(test_data, test_labels, s=5, c='g', label="Testing Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, s=5, c='r', label="Predicted Data")        

    plt.legend()
    plt.show()

class salaryPredictionv0(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)
    
model = salaryPredictionv0().to(device)

loss_fn = nn.MSELoss()
optmizer = optim.SGD(model.parameters(), lr=0.01)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

torch.manual_seed(42)
epochs = 2000 

for epoch in range(epochs):
    model.train()

    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch%100 == 0:
       print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")


model.eval()
with torch.inference_mode():
    y_preds = model(X_test)

plot_predictions(predictions= y_preds.cpu())