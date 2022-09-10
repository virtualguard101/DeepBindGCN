import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.data import InMemoryDataset, DataLoader

TRAIN_BATCH_SIZE = 500

# GCN based model
class GCNNet(torch.nn.Module):
    #def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=21, output_dim=128, dropout=0.2):
    def __init__(self, n_output=1, n_filters=64, embed_dim=1280,num_features_xd=78, num_features_xt=30, output_dim=1280, dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        print (str(num_features_xd)+'xxxxxx')
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.conv4 = GCNConv(num_features_xd*4, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.conv1_xt = GCNConv(num_features_xt, num_features_xt)
        self.conv2_xt = GCNConv(num_features_xt, num_features_xt*2)
        self.conv3_xt = GCNConv(num_features_xt*2, num_features_xt * 4)
        self.conv4_xt = GCNConv(num_features_xt*4, num_features_xt * 4)
        self.fc_g1_xt = torch.nn.Linear(num_features_xt*4, 1024)
        self.fc_g2_xt = torch.nn.Linear(1024, output_dim)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)


        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data, TRAIN_BATCH_SIZE, device):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print (data.batch)
        #print (x)
        #print ('yyyyyyyyyyyyyyyyyyyyyyyyyyy')
        # get protein input
        target = data.target
        #print (data.name)
        #print (target)

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = self.conv4(x, edge_index)
        x = self.relu(x)



        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        #x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        ##embedded_xt = self.embedding_xt(target)
        ##conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        ##xt = conv_xt.view(-1, 32 * 121)
        ##xt = self.fc1_xt(xt)
        #pocket graphic
        
        train_loader2 = DataLoader(target,batch_size=TRAIN_BATCH_SIZE)
        for xx in train_loader2:
                
                xx[0].to(device)
                # (xx[0].x)
                xt, edge_index_t,batch_t=xx[0].x, xx[0].edge_index,xx[0].batch

        xt = self.conv1_xt(xt, edge_index_t)
        xt = self.relu(xt)

        xt = self.conv2_xt(xt, edge_index_t)
        xt = self.relu(xt)

        xt = self.conv3_xt(xt, edge_index_t)
        xt = self.relu(xt)

        xt = self.conv4_xt(xt, edge_index_t)
        xt = self.relu(xt)



        xt = gmp(xt, batch_t)       # global max pooling

        # flatten
        xt = self.relu(self.fc_g1_xt(xt))
        #xt = self.dropout(xt)
        xt = self.fc_g2_xt(xt)
        xt = self.dropout(xt)

        #  combination
        # concat
        xc = torch.cat((x, xt), 1)

        # add some dense layers
        #xc = self.fc1(xc)
        xc=self.fc1(xc)
        xc = self.relu(xc)
        #xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out



