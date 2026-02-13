### nn architecture, plus train and test loops

import itertools
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings


class ld_layer(nn.Module):
    """
    Layer capturing LD-related features from SNPs.

    Inputs:
        geno_input: (bsz, n, num_snps)
        pos_input:  (bsz, 1, num_snps)

    Output:
        (bsz, output_size)
    """
    def __init__(self, n, output_size, l):
        super(ld_layer, self).__init__()

        ### params
        self.pool_size = 10
        self.f=64  # hidden layer size
        kernel_size=2
        self.rbf_bins = int(np.ceil(np.log(l)))
        
        ### RBF parameters
        low=0.0
        high=1.0
        nbins = int(np.ceil(np.log(l)))
        self.rbf_epsilon = 1/l
        log_low = torch.log(torch.tensor(low + self.rbf_epsilon))
        log_high = torch.log(torch.tensor(high + self.rbf_epsilon))
        rbf_centers = torch.linspace(log_low, log_high, nbins)
        rbf_width = (rbf_centers[1] - rbf_centers[0])
        rbf_centers = rbf_centers.view(1, nbins, 1)
        self.register_buffer('rbf_centers', rbf_centers)  # non-learnable tensor
        self.register_buffer('rbf_width', rbf_width)

        # layers for W-generating function
        self.W_dense_0 = nn.Conv1d(in_channels=self.rbf_bins,
                                    out_channels=self.f,
                                    kernel_size=1,
                                    stride=1,
                                    )
        self.pos_conv1 = nn.Conv1d(in_channels=self.f,
                               out_channels=self.f,
                               kernel_size=1,
                               stride=1,
        )

        ### LD genotype convs
        self.geno_conv_0 = nn.Conv1d(in_channels=n,
                               out_channels=self.f,
                               kernel_size=1,
                               stride=1,
        )
        self.geno_conv_1 = nn.Conv1d(in_channels=self.f*2,
                               out_channels=self.f,
                               kernel_size=1,
                               stride=1,
        )
        self.geno_conv_2 = nn.Conv1d(in_channels=self.f,
                               out_channels=self.f,
                               kernel_size=1,
                               stride=1,
        )
        self.geno_conv_3 = nn.Conv1d(in_channels=self.f,
                               out_channels=self.f,
                               kernel_size=1,
                               stride=1,
                               )

        ### additional layers after the convolutions
        warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
        self.final_dense_0 = nn.LazyLinear(128)
        self.final_dense_1 = nn.Linear(128, 128)
        self.final_dense_2 = nn.Linear(128, 128)
        self.final_dense_3 = nn.Linear(128, 128)
        self.final_dense_4 = nn.Linear(128, output_size)

    def logRBFExpansion(self, d):
        log_d = torch.log(d + self.rbf_epsilon)
        diff = log_d - self.rbf_centers
        rbf = torch.exp(-(diff ** 2) / (self.rbf_width ** 2))

        return rbf
    
    def partner(self, geno_input, loops):
        M = geno_input.shape[-1]
        idx = torch.arange(M, device=geno_input.device)
        idx = idx.repeat(loops)
        partners = torch.rand(M*loops, device=geno_input.device)  # use same partner indices across batch elements
        partners = torch.floor(torch.exp(partners * math.log(M)))  # log-U

        if torch.rand(1) < 0.5:  # move positively half the time
            partners = idx + partners
            mask = partners < (M - 1)
            pair_idx = torch.stack([idx[mask], partners[mask]], dim=-1)  # (snp_pairs, 2)
            
        else:  # move in negative direction other times
            partners = idx - partners
            mask = partners >= 0
            pair_idx = torch.stack([partners[mask], idx[mask]], dim=-1)  # (snp_pairs, 2)
        
        return pair_idx.type(torch.int64)
        
    def positional_mapping(self, pos):

        # compute distances                                                      
        dist = torch.abs(pos[:, :, 0] - pos[:, :, 1])  # (bsz, snp_pairs)           

        # rbf                                                                    
        dist = dist.unsqueeze(1)  # (bsz, 1, snp_pairs)                           
        rbf_features = self.logRBFExpansion(dist)  # (bsz, nbins, snp_pairs)                  

        # learn W
        W = self.W_dense_0(rbf_features)  # (bsz, f, snp_pairs)
        W = F.relu(W)
        W = self.pos_conv1(W)  # (bsz, f, snp_pairs)                              
        W = F.relu(W)

        return W
        
    def geno_conv(self, h, W=None):
        B, n, snp_pairs, _ = h.shape  # bsz, n, snp_pairs, 2                             

        # position-wise
        h = h.reshape(B, n, snp_pairs*2)  # temporarily stack pairs for position-wise operation
        h = self.geno_conv_0(h)  # to filter-size torch.Size([bsz, f, snp_pairs*2])
        h = F.relu(h)

        # pairwise                                                       
        h = h.reshape(B, self.f, snp_pairs, 2)
        h = h.permute(0, 1, 3, 2)  # (bsz, f, 2, snp_pairs)               
        h = h.reshape(B, self.f*2, snp_pairs)  # (bsz, f*2, snp_pairs)            
        h = self.geno_conv_1(h)
        h = F.relu(h)  # (relu here or after W multiplication are equivilent (if W has relu))
        h = self.geno_conv_2(h)  # (bsz, f, snp_pairs)
        h = F.relu(h)
        
        # rescale by positional coefficients                             
        if W is not None:
            h = h * W

        # extra layers                                                   
        h = self.geno_conv_3(h)
        h = F.relu(h)

        # average
        h = torch.mean(h, dim=-1)  # (bsz, f)                                                                                               
        
        return h
    
    def forward(self, geno_input, pos_input):

        pos_input = pos_input.squeeze(1)  # (bsz, m)  (keep the "(1)" to avoid oversqueezing size=1 batches)        
        pair_idx = self.partner(geno_input, loops=10)  # generate pair indices
        geno_h = geno_input[:, :, pair_idx]  # (bsz, n, snp_pairs, 2)  Currently more efficient to take pairs first-usually ~1600 out of 5000 snps-BEFORE position-wise operations. Also consistent with mapping fxn.
        pos_h = pos_input[:, pair_idx]  # (bsz, snp_pairs, 2)   
        W = self.positional_mapping(pos_h)  # (bsz, f, snp_pairs)        
        h = self.geno_conv(geno_h, W)

        # dense
        h = self.final_dense_0(h)
        h = F.relu(h)
        h = self.final_dense_1(h)
        h = F.relu(h)
        h = self.final_dense_2(h)
        h = F.relu(h)
        h = self.final_dense_3(h)
        h = F.relu(h)
        output = self.final_dense_4(h)

        return output        


    
class pairwise_cnn(nn.Module):
    def __init__(self, args):
        super(pairwise_cnn, self).__init__()

        # update conv+pool iterations based on number of SNPs
        self.pool_size = 10
        self.num_conv_iterations = int(np.floor(np.log10(args.num_snps)) - 1)  # (specific to poolsize=10)
        if self.num_conv_iterations < 0:
            self.num_conv_iterations = 0
        self.kernel_size = 2
        self.pairs = int(args.pairs)  # used in forward()
        self.use_locs = bool(args.use_locs)
        
        # organize pairs
        all_indices = list(range(int(args.n*(args.n-1)/2)))
        self.pair_indices = random.sample(all_indices, self.pairs)  # sampling fewer pairs (also shuffles)
        combinations_all = list(itertools.combinations(range(args.n), 2))  # total theoretical pairs
        self.combinations = [combinations_all[i] for i in self.pair_indices]
        all_indices_2 = list(range(self.pairs))
        self.pair_indices_encode = random.sample(all_indices_2, args.pairs_encode)  # even fewer pairs used for training 
        self.encoder_mask = torch.zeros(self.pairs, dtype=torch.bool)
        self.encoder_mask[self.pair_indices_encode] = True
        
        # initialize lists of shared layers                       
        self.GENO_CONVS = nn.ModuleList()

        # initialize a number of convolutions
        previous_size_geno = 2  # (two individuals)
        for conv in range(self.num_conv_iterations):
            filter_size = 20 + 44 * (conv + 1)
            
            # geno
            self.GENO_CONVS.append(
                nn.Conv1d(in_channels=previous_size_geno,
                          out_channels=filter_size,
                          kernel_size=self.kernel_size,
                          stride=self.kernel_size
                          )
            )
            previous_size_geno = int(filter_size)

        # additional layers after the convolutions
        warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
        self.dense_shared_geno = nn.LazyLinear(128)
        
        # single dense layer after pairwise operations
        self.final_dense = nn.Linear(128*self.pairs, args.output_size)

    def forward(self, geno_input, pos_input):

        pair_indices = torch.tensor(self.combinations, device=geno_input.device)  # (pairs, 2)
        geno_h = geno_input[:, pair_indices, :]  # repeat data for pairs  (bsz, pairs, 2, snps) ((checked 10/23/25))

        B, _, M = geno_input.shape
        geno_h = geno_h.reshape(B * self.pairs, 2, M)  # collapse pairs and batches- for now (bsz*pairs, 2, snps)
        
        # separate out some pairs from training convs
        mask = self.encoder_mask.repeat(B).to(geno_input.device)
        geno_h_encode = geno_h[mask]  # (bsz*pairs_encode, 2, snps)   
        geno_h_skip = geno_h[~mask]

        # first loop with gradient
        for conv_number in range(self.num_conv_iterations):
            geno_h_encode = self.GENO_CONVS[conv_number](geno_h_encode)  # (bsz*pairs_encode, features, snps)
            geno_h_encode = F.relu(geno_h_encode)
            geno_h_encode = F.avg_pool1d(geno_h_encode, kernel_size=self.pool_size)
        
        # second loop without gradient
        with torch.no_grad():
            for conv_number in range(self.num_conv_iterations):
                geno_h_skip = self.GENO_CONVS[conv_number](geno_h_skip)
                geno_h_skip = F.relu(geno_h_skip)
                geno_h_skip = F.avg_pool1d(geno_h_skip, kernel_size=self.pool_size)
        
        # merge back
        _, feat_dim, M = geno_h_encode.shape
        geno_h = torch.zeros((B*self.pairs, feat_dim, M), device=geno_input.device)
        geno_h[mask] = geno_h_encode
        geno_h[~mask] = geno_h_skip    
                
        # flatten
        geno_h = torch.flatten(geno_h, start_dim=1)  # torch.Size([bsz*pairs, 2052])

        # concatenate locs
        if self.use_locs:
            locs_h = loc_input[:, pair_indices, :]  # (bsz, pairs, 2, 2)
            dist = torch.norm(locs_h[:,:,0,:] - locs_h[:,:,1,:], dim=-1)  # (bsz, pairs)
            dist = dist.reshape(B * self.pairs, 1) 
            geno_h = torch.cat([geno_h, dist], dim = -1)

        # dense
        geno_h = self.dense_shared_geno(geno_h)  # torch.Size([bsz*pairs, 128])
        geno_h = F.relu(geno_h)
        
        # separate out batches again
        geno_h = geno_h.reshape(B, self.pairs, -1)  # torch.Size([bsz, pairs, 128])
        
        # final dense
        geno_h = geno_h.flatten(start_dim=1)  # torch.Size([bsz, pairs*128])
        output = self.final_dense(geno_h)
        
        return output


    
class cnn(nn.Module): 
    def __init__(self, args):
        super(cnn, self).__init__()
        
        # params                            
        self.pool_size = 10
        self.num_conv_iterations = int(np.floor(np.log10(args.num_snps)) - 1)  # (specific to poolsize=10)                                                                                 
        if self.num_conv_iterations < 0:
            self.num_conv_iterations = 0
        self.kernel_size= 2  # p. 228  
        
        # initialize lists of shared layers
        self.GENO_CONVS = nn.ModuleList()
            
        # initialize a number of convolutions
        previous_size_geno = int(args.n)
        for conv in range(self.num_conv_iterations):
            filter_size = 20 + 44 * (conv + 1)
            self.GENO_CONVS.append(  # regular conv
                nn.Conv1d(in_channels=previous_size_geno,
                          out_channels=filter_size,
                          kernel_size=self.kernel_size,
                          stride=self.kernel_size
                          ))
            previous_size_geno = int(filter_size)

        
        # additional layers after the convolutions
        warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
        self.final_dense_0 = nn.LazyLinear(128)
        self.final_dense_1 = nn.Linear(128, 128)
        self.final_dense_2 = nn.Linear(128, 128)
        self.final_dense_3 = nn.Linear(128, args.output_size)
        
    def forward(self, geno_input, pos_input):
        geno_h = geno_input  # torch.Size([bsz, n, num_snps])
        for conv_number in range(self.num_conv_iterations):
            geno_h = self.GENO_CONVS[conv_number](geno_h)
            geno_h = F.relu(geno_h)
            geno_h = F.avg_pool1d(geno_h, kernel_size=self.pool_size)

        geno_h = torch.flatten(geno_h, start_dim=1)
        geno_h = self.final_dense_0(geno_h)  
        geno_h = F.relu(geno_h)
        geno_h = self.final_dense_1(geno_h)
        geno_h = F.relu(geno_h)
        geno_h = self.final_dense_2(geno_h)
        geno_h = F.relu(geno_h)
        output = self.final_dense_3(geno_h)
            
        return output
