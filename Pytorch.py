import torch


x_data = torch.Tensor([1.0,2.0,3.0,4.0])
y_data = torch.Tensor([2.0,4.0,6.0,8.0])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(4,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("weights", model.linear.weight)
print("biases", model.linear.bias)
x_test_data = torch.Tensor([5.0,6.0,7.0,8.0])
y_test_data = torch.Tensor(x_test_data)
print(x_test_data, y_test_data)