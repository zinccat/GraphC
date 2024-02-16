from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, NNConv, GATConv, SAGEConv, GINConv, global_max_pool, global_add_pool
from tqdm import tqdm, trange
from torcheval.metrics.aggregation.auc import AUC
import wandb
import numpy as np

# seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def load_embeddings():
    embeddings = np.load('./data/embeddings.npy')
    return embeddings

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels* hidden_channels)
        )
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * hidden_channels)
        )
        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * dataset.num_classes)
        )
        self.conv1 = NNConv(hidden_channels, hidden_channels, self.lin1, aggr='mean')
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.lin2, aggr='mean')
        self.conv3 = NNConv(hidden_channels, hidden_channels, self.lin3, aggr='mean')
        self.embedding = torch.nn.Embedding(dataset.num_node_features, hidden_channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr.view(-1, 1)
        # normalize edge_attr
        # edge_attr = (edge_attr - edge_attr.mean()) / edge_attr.std()
        # turn x from one-hot to continuous
        x = x.argmax(dim=1)
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        embed_dim = 50
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, embed_dim)
            self.conv1 = GATConv(embed_dim, hidden_channels, normalize=normalize)
            embed = load_embeddings()
            self.embedding.weight.data.copy_(torch.from_numpy(embed))
        else:
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = GATConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv3 = GATConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GAT_layer2(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, hidden_channels)
            self.conv1 = GATConv(hidden_channels, hidden_channels, normalize=normalize)
        else:
            self.conv1 = GATConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = GATConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        embed_dim = 50
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, embed_dim)
            self.conv1 = GCNConv(embed_dim, hidden_channels, normalize=normalize)
            embed = load_embeddings()
            self.embedding.weight.data.copy_(torch.from_numpy(embed))
            # self.embedding.weight.requires_grad = False
        else:
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index, edge_weight=data.edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GCN_l4(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        embed_dim = 50
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, embed_dim)
            self.conv1 = GCNConv(embed_dim, hidden_channels, normalize=normalize)
            embed = load_embeddings()
            self.embedding.weight.data.copy_(torch.from_numpy(embed))
            # self.embedding.weight.requires_grad = False
        else:
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv4(x, edge_index, edge_weight=data.edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GCN_l5(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        embed_dim = 50
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, embed_dim)
            self.conv1 = GCNConv(embed_dim, hidden_channels, normalize=normalize)
            embed = load_embeddings()
            self.embedding.weight.data.copy_(torch.from_numpy(embed))
            # self.embedding.weight.requires_grad = False
        else:
            self.conv1 = GCNConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv4 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv5 = GCNConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv4(x, edge_index, edge_weight=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv5(x, edge_index, edge_weight=data.edge_attr)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        embed_dim = 50
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, embed_dim)
            self.conv1 = GINConv(torch.nn.Sequential(
        torch.nn.Linear(embed_dim, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ))
            embed = load_embeddings()
            self.embedding.weight.data.copy_(torch.from_numpy(embed))
            # self.embedding.weight.requires_grad = False
        else:
            self.conv1 = GINConv(torch.nn.Sequential(
                torch.nn.Linear(dataset.num_node_features, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv3 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels=16, dropout=0.5, use_embedding=True, normalize=True):
        super().__init__()
        if use_embedding:
            self.embedding = torch.nn.Embedding(dataset.num_node_features, hidden_channels)
            self.conv1 = SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
        else:
            self.conv1 = SAGEConv(dataset.num_node_features, hidden_channels, normalize=normalize)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
        self.dropout = dropout
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, dataset.num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(self, 'embedding'):
            x = x.argmax(dim=1)
            x = self.embedding(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.mlp(x)
        return x

def evaluate(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data).max(1)[1]
        correct += int(out.eq(data.y).sum().item())
    return correct / len(loader.dataset)

def evaluate_auc(loader):
    model.eval()
    auc = AUC(n_tasks=1)
    for data in loader:
        data = data.to(device)
        out = model(data)
        auc.update(out[:, 1], data.y)
    return auc.compute().item()

def train(model, optimizer, loader):
    model.train()
    losses = []
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

if __name__ == '__main__':
    # download and process
    dataset = TUDataset(root='./data/', name='TWITTER-Real-Graph-Partial', use_edge_attr=True)

    # total of 144033 graphs, 2 classes, but with 1643 classes for node labels

    dataset = dataset.shuffle()
    len_dataset = len(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # split dataset
    train_dataset = dataset[:int(len_dataset * 0.8)]
    val_dataset = dataset[int(len_dataset * 0.8):int(len_dataset * 0.9)]
    test_dataset = dataset[int(len_dataset * 0.9):]

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    hidden_channels = 32 #64 #256
    dropout = 0.1
    embed = False #True
    normalize = True
    model = GIN(hidden_channels=hidden_channels, dropout=dropout, use_embedding=embed, normalize=normalize).to(device)
    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    wandb.init(project='TwitterG', name=f"{model.__class__.__name__} lr={lr} dim={hidden_channels} p={dropout} embed={embed}, norm={normalize}")
    wandb.watch(model)

    for epoch in trange(1, 101):
        model.train()
        losses = train(model, optimizer, train_loader)
        train_loss = sum(losses) / len(losses)
        train_acc = evaluate(train_loader)
        val_acc = evaluate(val_loader)
        test_acc = evaluate(test_loader)
        print(f'Epoch: {epoch}, Loss: {sum(losses) / len(losses)}')
        print(f'Epoch: {epoch}, Train accuracy: {train_acc}, Val accuracy: {val_acc}, Test accuracy: {test_acc}')
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})

    # evaluate
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data).max(1)[1]
        correct += int(out.eq(data.y).sum().item())

    print(f'Accuracy: {correct / len(test_dataset)}')