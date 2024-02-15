from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv
from tqdm import tqdm, trange

# seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.lin1 = torch.nn.Linear(1, dataset.num_node_features * hidden_channels)
        self.lin2 = torch.nn.Linear(1, hidden_channels * dataset.num_classes)
        self.conv1 = NNConv(dataset.num_node_features, hidden_channels, self.lin1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, dataset.num_classes, self.lin2, aggr='mean')
    
    def forward(self, data):
        # x here corresponds to node labels, with feature size 1323
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.view(-1, 1)
        # normalize edge_attr
        # edge_attr = (edge_attr - edge_attr.mean()) / edge_attr.std()
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        # aggregate the node features
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

# download and process
dataset = TUDataset(root='./data/', name='TWITTER-Real-Graph-Partial', use_edge_attr=True)

# total of 144033 graphs, 2 classes, but with 1323 classes for node labels

dataset = dataset.shuffle()
len_dataset = len(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split dataset
train_dataset = dataset[:int(len_dataset * 0.8)]
val_dataset = dataset[int(len_dataset * 0.8):int(len_dataset * 0.9)]
test_dataset = dataset[int(len_dataset * 0.9):]

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = GCN(256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data).max(1)[1]
        correct += int(out.eq(data.y).sum().item())
    return correct / len(loader.dataset)

for epoch in trange(1, 101):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Train accuracy: {evaluate(train_loader)}, Val accuracy: {evaluate(val_loader)}')
    print(f'Test accuracy: {evaluate(test_loader)}')

# evaluate
model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data).max(1)[1]
    correct += int(out.eq(data.y).sum().item())

print(f'Accuracy: {correct / len(test_dataset)}')