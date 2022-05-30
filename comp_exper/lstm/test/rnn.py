import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import time


since = time.time()
num_time_steps = 50  # 点的数量
input_size = 1
hidden_size = 16
output_size = 1
lr = 0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev


device = 'cpu'
# if torch.cuda.is_available():
#     device = torch.device('cuda')

model = Net().to(device)
critirion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)
hidden_prev = torch.zeros(1, 1, hidden_size).to(device)


for iter in range(6000):
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start, start+10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps-1, 1).to(device)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps-1, 1).to(device)

    output, hidden_prev = model(x, hidden_prev)
    hidden_prev = hidden_prev.detach()

    loss = critirion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Iteration: {}, Loss: {}'.format(iter, loss.item()))

start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start+10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps-1, 1).to(device)
y = torch.tensor(data[1:]).float().view(1, num_time_steps-1, 1).to(device)

predictions = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    (pred, hidden_prev) = model(input, hidden_prev)
    input = pred
    predictions.append(pred.cpu().detach().numpy().ravel()[0])

x = x.cpu().data.numpy().ravel()
y = y.cpu().data.numpy()
plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

# cpu 18.958863735198975 s
# gpu 41.58114838600159 s
print(f'{device} consume time: {time.time()-since}')

plt.scatter(time_steps[1:], predictions)
plt.show()