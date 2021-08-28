import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, dilation):
        super(ConvBlock, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.conv = torch.nn.Conv1d(in_channels, n_filters, kernel_size=kernel_size, dilation=dilation, padding = kernel_size//2 + dilation - 1)
        self.gelu = torch.nn.GELU()
        self.batchnorm = torch.nn.BatchNorm1d(n_filters)
    
    def forward(self, X):
        X = self.gelu(X)
        X = self.conv(X)
        X = self.batchnorm(X)
        
        return X
    

class RConvBlock(torch.nn.Module):
    def __init__(self, in_channels, n_filters, kernel_size, dilation):
        super(RConvBlock, self).__init__()
        self.conv = ConvBlock(in_channels, n_filters, kernel_size=kernel_size, dilation=dilation)
    
    def forward(self, X):
        X_conv = self.conv(X)
        X = torch.add(X_conv, X)
        
        return X
    

class DilatedStem(torch.nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DilatedStem, self).__init__()
        self.conv = ConvBlock(in_channels, n_filters, kernel_size = 21, dilation = 1)
        self.rconv = RConvBlock(n_filters, n_filters, kernel_size = 1, dilation = 1)
    
    def forward(self, X):
        X = self.rconv(self.conv(X))
        
        return X

class MHABlock(torch.nn.Module):
    def __init__(self, n_filters, n_keys, n_heads):
        super(MHABlock, self).__init__()
        self.layernorm = torch.nn.LayerNorm(n_filters)
        self.MHA = torch.nn.MultiheadAttention(embed_dim=n_filters, num_heads=n_heads)
        self.dropout = torch.nn.Dropout(p=0.4)
        
    def forward(self, X):
        X_mha = self.MHA(self.layernorm(X))
        X = torch.add(X_mha,X)
        
        return X

    
class MLP(torch.nn.Module):
    def __init__(self, n_filters):
        super(MLP, self).__init__()
        self.layernorm = torch.nn.LayerNorm(n_filters)
        self.conv1 = torch.nn.Conv1d(n_filters, 2*n_filters, kernel_size=1)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(2*n_filters, n_filters, kernel_size=1)
    
    def forward(self, X):
        X_m = self.layernorm(X)
        X_m = self.conv1(X_m)
        X_m = self.dropout(X_m)
        X_m = self.relu(X_m)
        X_m = self.conv2(X_m)
        X_m = self.dropout(X_m)
        
        X = torch.add(X_m,X)
        
        return X
    
class ConvTower(torch.nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ConvTower, self).__init__()
        self.conv = ConvBlock(in_channels, n_filters, kernel_size = 5, dilation = 1)
        self.rconv = RConvBlock(n_filters, n_filters, kernel_size = 1, dilation = 1)

    def forward(self, X):
        X = self.rconv(self.conv(X))

        
        return X

    
class DilatedConvolution(torch.nn.Module):
    def __init__(self, in_channels, n_filters, dilation):
        super(DilatedConvolution, self).__init__()
        self.conv_1 = ConvBlock(in_channels, n_filters, kernel_size = 3, dilation = dilation)
        self.conv_2 = ConvBlock(in_channels, n_filters, kernel_size = 1, dilation = 1)
        self.dropout = torch.nn.Dropout(p=0.3)
    
    def forward(self, X):
        X_m = self.dropout(self.conv_2(self.conv_1(X)))
        X = torch.add(X_m,X)
        
        return X

class Dilated(torch.nn.Module):
    def __init__(self, in_channels, n_filters, n_dilation_layer, trimming, n_celltypes, n_assays):
        super(Dilated, self).__init__()
        self.trimming = trimming
        self.n_filters = n_filters
        self.n_layers = n_dilation_layer
        self.n_celltypes = n_celltypes
        self.n_assays = n_assays
        self.stem = DilatedStem(in_channels, n_filters//2)
        self.ci_list = [int(n_filters * (1.2225 ** i)) for i in range(1, 6)]
        self.convtower = torch.nn.Sequential(ConvTower(n_filters//2, self.ci_list[0]),
                                                  ConvTower(self.ci_list[0], self.ci_list[1]),
                                                  ConvTower(self.ci_list[1], self.ci_list[2]),
                                                  ConvTower(self.ci_list[2], self.ci_list[3]),
                                                  ConvTower(self.ci_list[3], self.ci_list[4]),
                                                  ConvTower(self.ci_list[4], n_filters)
                                                 )
        self.di_list = [int(numpy.around(1.5 ** i, decimals = 1)) for i in range(n_dilation_layer)]
        self.dilatedconv = torch.nn.ModuleList([DilatedConvolution(n_filters, n_filters, self.di_list[i]) for i in range(n_dilation_layer)])
        self.conv = ConvBlock(n_filters, 2*n_filters, kernel_size = 1, dilation = 1)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.gelu = torch.nn.GELU()
        
        self.assay_convs = torch.nn.ModuleList([
            torch.nn.Conv1d(2*n_filters, n_filters, kernel_size=75) for i in range(n_assays)
        ])

        self.celltype_convs = torch.nn.ModuleList([
            torch.nn.Conv1d(2*n_filters, n_filters, kernel_size=75) for i in range(n_celltypes)
        ])
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.conv_2 = torch.nn.Conv1d(n_filters, 1, kernel_size = 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, X, celltype_idxs = torch.tensor([0]), assay_idxs = torch.tensor([0])):
        start, end = self.trimming, X.shape[2] - self.trimming
        X = self.convtower(self.stem(X))
        for layer in self.dilatedconv:
            X = layer(X)
        
        
        X = X[:, :, start:end]
        X = self.gelu(self.dropout(self.conv(X)))
        
        X_celltype, X_assay = [], []
        for i, (celltype_idx, assay_idx) in enumerate(zip(celltype_idxs, assay_idxs)):
            Xc = self.celltype_convs[celltype_idx](X[i:i+1])
            X_celltype.append(Xc)

            Xa = self.assay_convs[assay_idx](X[i:i+1])
            X_assay.append(Xa)

        X_celltype = torch.cat(X_celltype)
        X_assay = torch.cat(X_assay)

        y_profile = self.relu(torch.mul(X_celltype, X_assay))
        y_profile = self.conv_2(y_profile).squeeze()
        y_profile = self.logsoftmax(y_profile)
        
        return y_profile
    
    def predict(self, X, celltype_idxs, assay_idxs, batch_size=64):
        with torch.no_grad():
            starts = numpy.arange(0, X.shape[0], batch_size)
            ends = starts + batch_size

            y_hat = []
            for start, end in zip(starts, ends):
                y_hat_ = self(X[start:end], celltype_idxs[start:end],
                    assay_idxs[start:end]).cpu().detach().numpy()
                y_hat.append(y_hat_)

            y_hat = numpy.concatenate(y_hat)
            return y_hat

    def fit_generator(self, training_data, optimizer, X_valid=None, 
        celltype_idxs_valid=None, assay_idxs_valid=None, y_valid=None, 
        max_epochs=100, batch_size=64, validation_iter=100, verbose=True):

        if X_valid is not None: 
            X_valid = X_valid.cuda()
            celltype_idxs_valid = celltype_idxs_valid.cuda()
            assay_idxs_valid = assay_idxs_valid.cuda()
        
        y_valid_ = y_valid.detach().numpy()

        if verbose:
            print("Epoch\tIteration\tTraining Time\tValidation Time\tTraining MLL\tValidation MLLL\tValidation Correlation")

        start = time.time()
        iteration = 0
        best_corr = 0

        for epoch in range(max_epochs):
            tic = time.time()

            for X, celltype_idxs, assay_idxs, y in training_data:
                X = X.cuda()
                celltype_idxs = celltype_idxs.cuda()
                assay_idxs = assay_idxs.cuda()
                y = y.cuda()

                optimizer.zero_grad()
                self.train()

                y_profile = self(X, celltype_idxs, assay_idxs)

                loss = MLLLoss(y_profile, y)
                train_loss = loss.item()
                loss.backward()

                optimizer.step()

                if verbose and iteration % validation_iter == 0:
                    self.eval()

                    train_time = time.time() - start
                    tic = time.time()

                    y_profile = self.predict(X_valid, celltype_idxs_valid, assay_idxs_valid, batch_size=batch_size)
                    valid_loss = MLLLoss(y_profile, y_valid).item()

                    y_profile = numpy.exp(y_profile)
                    valid_corrs = numpy.mean(numpy.nan_to_num(pearson_corr(y_profile, y_valid_)))
                    valid_time = time.time() - tic

                    print("{}\t{}\t{:4.4}\t{:4.4}\t{:6.6}\t{:6.6}\t{:4.4}".format(
                    epoch, iteration, train_time, valid_time, train_loss, valid_loss, 
                        valid_corrs))
                    start = time.time()

                    if valid_corrs > best_corr:
                        best_corr = valid_corrs

                        self = self.cpu()
                        torch.save(self, "/mnt/data/imputation_yangyuan/models/batchnorm.{}.{}.torch".format(self.n_filters, self.n_layers))
                        self = self.cuda()
                        
                iteration += 1