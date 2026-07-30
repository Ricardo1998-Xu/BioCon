'''
importing libraries
'''
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import scipy
import statistics


class scale_module_mean(nn.Module):
    def __init__(self, inputDim, Nnodes):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, 1)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z)
        
        return z
    

    
class scale_module_std(nn.Module):
    def __init__(self, inputDim, Nnodes):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, 1)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        
        return z

class latent_module_mean(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z)
        
        return z

class latent_module_std(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        self.latent_dim = latent_dim
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        
        return z
    
class get_gamma_params(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))   
        z = torch.relu(self.fc3(z))   
        z = torch.nn.functional.softmax(self.fc4(z), dim = 1)
        
        return z


class get_gamma_theta(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = torch.relu(self.fc4(z))
        
        return z
    
    
    
class bernoulli_sample(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, inputDim)
        
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.sigmoid(self.fc4(z))
        z = torch.clamp(z, max = 0.99, min = 0.01)
        return z

    
    
class expand_time(nn.Module):
    def __init__(self, Nnodes, latent_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(1, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, latent_dim)
        
        self.latent_dim = latent_dim
        self.Nnodes = Nnodes
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z)
        
        return z 
    
    
class predict_time_from_latent_space(nn.Module):
    def __init__(self, inputDim, Nnodes):
        super().__init__()
        
        self.fc1 = nn.Linear(inputDim, Nnodes)
        self.fc2 = nn.Linear(Nnodes, Nnodes)
        self.fc3 = nn.Linear(Nnodes, Nnodes)
        self.fc4 = nn.Linear(Nnodes, 1)
        
        self.inputDim = inputDim
        self.Nnodes = Nnodes
        
    def forward(self,z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        z = self.fc4(z)
        
        return z 

    
'''
basic SNOW object
'''
class SNOW_v1(nn.Module):
    def __init__(self, inputDim, Nnodes, latent_dim, time_dim = 32, fixStd = False, fixVal = 0.5):
        super().__init__()


        self.latentMean = latent_module_mean(inputDim, Nnodes, latent_dim)
        self.latentStd = latent_module_std(inputDim, Nnodes, latent_dim)
        self.getGamma = get_gamma_params(inputDim, Nnodes, latent_dim + time_dim - 1)              
        self.getGammaS = get_gamma_theta(inputDim, Nnodes, latent_dim + time_dim - 1)              
        self.bernoulli = bernoulli_sample(inputDim, Nnodes, latent_dim + time_dim - 1)             
        self.expandTime = expand_time(Nnodes, time_dim)
        self.inputDim = inputDim
        self.latentDim = latent_dim
        self.time_dim = time_dim
        self.fixStd = fixStd
        self.fixVal = fixVal
       
    def forward(self, z):
        
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self.latentDim - 1), torch.eye(self.latentDim - 1))
        
        '''
        get parameters of latent space for gene expression and library size
        '''
        latent_mean = self.latentMean(z)
        latentStd = self.latentStd(z)
        
        estimated_time = latent_mean[:,-1].view(z.shape[0],1)
        time_representation = self.expandTime(estimated_time)
        
        if self.fixStd == False:
            latent_state = torch.hstack((latent_mean[:,0:-1] + mvn.sample([1]) * latentStd[:,0:-1],
                                         time_representation
                                        )
                                       )
        
        if self.fixStd == True:
            latent_state = torch.hstack((latent_mean[:,0:-1] + mvn.sample([1]) * self.fixVal,
                                         time_representation
                                        )
                                       )
        
        latent_mean_state = torch.hstack((latent_mean[:,0:-1],
                                     time_representation
                                    )
                                   )
        

        
        '''
        for each cell, get count fraction (rho)
        '''
        rho = self.getGamma(latent_state)
        
        '''
        and the gene specific, cell-specific inverse dispersion
        '''
        theta = self.getGammaS(latent_state) + 1e-4
        gamma_dist = torch.distributions.gamma.Gamma(theta, theta/rho)
        w = gamma_dist.sample([1])
        w = w[0]
        
        '''
        use actual library size as l_sample
        '''
        l_sample = z.sum(dim = 1).view(z.shape[0],1)
        
        '''
        sample count from a poisson and obtain zero probability
        '''
        y = torch.poisson(l_sample * torch.clamp(w, min=1e-30, max = 1))
        h = torch.bernoulli(self.bernoulli(latent_state
                                          )
                           )

        x = y*h

        
      
        '''
        computing logprob of each cell 
        '''
        p0 = torch.log(((1-self.bernoulli(latent_state)) + (self.bernoulli(latent_state)) * (theta/(theta + l_sample * rho))**theta))
        p1 = torch.log((self.bernoulli(latent_state))) + torch.lgamma(z + theta) - torch.lgamma(z + 1.0) - torch.lgamma(theta) + theta * torch.log(theta/(theta +  l_sample*rho)) + z * torch.log( l_sample * rho/(theta +  l_sample*rho))
        logprob = p0*(z==0) +  p1*(z!=0)
        
        '''
        mean logprob for each cell
        '''
        N = z.shape[0] 
        mean_logprob = (p0[z==0].sum() +  p1[z!=0].sum()) / N
        

        
        # latent dim parameters
        self.latent_mean_all = latent_mean
        self.latent_std_all = latentStd
        self.latent_std = latent_mean.std(dim = 0)
        self.latent_mean = latent_mean.mean(dim = 0)
        
        # various sampling
        self.rho = self.getGamma(latent_mean_state)
        self.l_sample = l_sample
        self.sampledX = x

        
        # last step
        self.h = h
        self.y = y
        self.logprob = logprob
        self.meanlogprob = mean_logprob
        self.libSize = actual_size
        self.z = z
        self.estimated_time = estimated_time
        self.latentState = latent_state
        
        self.KL_loss = 0.5 *  ((latent_mean[:, 0:-1]**2).sum(dim = 1)  - torch.clamp(torch.log(latentStd[:, 0:-1]**2), min = -100).sum(dim = 1) - (self.latentDim - 1) + (latentStd[:, 0:-1]**2).sum(dim = 1) )
        
        return x
    

def sample_wasserstein(distribution, sample, dirNum):
    Ndim_samp = sample.shape[1]
    Nsamp = sample.shape[0]
    
    sampled_directions = torch.randn(dirNum, Ndim_samp)
    sampled_directions = sampled_directions / torch.linalg.norm(sampled_directions, dim = 1).view(sampled_directions.shape[0],1)
    dist_samples = distribution.sample([Nsamp])
    
    dist_projected = torch.matmul(dist_samples, sampled_directions.t())
    test_projected = torch.matmul(sample, sampled_directions.t())

    dist_projected_sort = torch.sort(dist_projected, dim = 0).values
    test_projected_sort = torch.sort(test_projected, dim = 0).values

    #return ((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).mean()
    #print(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1).shape)
    return torch.sqrt(((dist_projected_sort - test_projected_sort)**2).sum(dim = 1)).mean() / dirNum

def two_sample_wasserstein(sample1, sample2, dirNum):
    Ndim_samp = sample1.shape[1]
    Nsamp = np.min((sample1.shape[0], sample2.shape[0]))
    sampled_directions = torch.randn(dirNum, Ndim_samp)
    sampled_directions = sampled_directions / torch.linalg.norm(sampled_directions, dim = 1).view(sampled_directions.shape[0],1)
    
    samp1_index = range(sample1.shape[0])
    samp2_index = range(sample2.shape[0])
    
    sampled_1 = sample1[random.sample(samp1_index, Nsamp),:]
    sampled_2 = sample2[random.sample(samp2_index, Nsamp),:]
    
    samp1_projected = torch.matmul(sampled_1, sampled_directions.t())
    samp2_projected = torch.matmul(sampled_2, sampled_directions.t())
    
    samp1_projected_sort = torch.sort(samp1_projected, dim = 0).values
    samp2_projected_sort = torch.sort(samp2_projected, dim = 0).values
    
    return torch.sqrt(((samp1_projected_sort - samp2_projected_sort)**2).sum(dim = 1)).mean() / dirNum
    
def sample_wasserstein_individual(distribution, sample, dirNum):
    Ndim_samp = sample.shape[1]
    Nsamp = sample.shape[0]
    
    W = 0
    for i in range(Ndim_samp):
        
        dist_samples = distribution.sample([Nsamp])

        dist_projected_sort = torch.sort(dist_samples, dim = 0).values
        test_projected_sort = torch.sort(sample[:,i], dim = 0).values
        
        W_new = torch.sqrt(((dist_projected_sort - test_projected_sort)**2).sum()) / dirNum
        W += W_new
        
    return W/Ndim_samp
    
    


'''
training the network
'''
def train_net(input_anndata, 
               model, 
               time_field, 
              model_name,
               itr = 30000,
              Ndir = 50,
              max_time = 48, 
              batchSize = 300,
              smoothPower = 20):
    
    time_dim = 1
    all_cell_index = range(input_anndata.X.shape[0])
    
    optimizer_AE = torch.optim.Adam(model.parameters(), lr = 0.0005, betas = (0.8, 0.9), weight_decay=0.0001) 
    Ndim = model.latentDim
    
    mvn = torch.distributions.MultivariateNormal(torch.zeros(Ndim - time_dim), torch.eye(Ndim - time_dim))
    normalD = torch.distributions.Normal(0,1)
    
    counter = 0
    for epoch in range(itr):
        random_sample = random.sample(all_cell_index,batchSize)
        thisCell = torch.tensor(input_anndata.X[tuple(random_sample),:].toarray()).type(torch.float32)
        cellTime = torch.tensor(input_anndata.obs[time_field]).type(torch.float32)[random_sample]


        optimizer_AE.zero_grad()


        info = model(thisCell)
        thisCell_hat = model.l_sample * model.rho
        random_push = torch.rand([batchSize]) * max_time
        time_representation = model.expandTime(random_push.view(batchSize, 1))
        original_embedding = model.latent_mean_all.clone()

        latent_state = torch.hstack((original_embedding[:,0:-time_dim], 
                                     time_representation
                                    )
                                    )
        estimated_gene_exp = model.getGamma(latent_state) * model.libSize
        estimated_time = model.latentMean(estimated_gene_exp)[:,-time_dim]
        new_embedding = model.latentMean(estimated_gene_exp)[:,0:-time_dim]    
        W_dist_pushed = sample_wasserstein(mvn, new_embedding, Ndir)

        '''
        compute Wasserstein by time
        '''
        sample_class = input_anndata.obs[time_field][random_sample]
        unique_class = list(set(sample_class.values))

        W_dist_perT = torch.zeros(len(unique_class))
        for Nclass in range(len(unique_class)):
            className = unique_class[Nclass]
            W_dist_perT[Nclass] = sample_wasserstein(mvn, model.latent_mean_all[sample_class == className, 0:-time_dim], Ndir)

        W_dist = sample_wasserstein(mvn, model.latent_mean_all[:,0:-time_dim], Ndir)

        '''
        producing a time series for each cell
        '''
        t_sample = torch.linspace(0,max_time,200) 
        t_sample = t_sample + torch.randn(t_sample.shape)
        t_sample.requires_grad = True
        cell_embedding = model.latent_mean_all.detach().clone()[:,0:-time_dim]                  
        time_representation = model.expandTime(t_sample.view(t_sample.shape[0], 1)) 
        cell_embedding_time = cell_embedding.repeat(t_sample.shape[0],1,1) # this is time point x Ncell x ndim

        cell_timeSeries = torch.cat((cell_embedding_time, time_representation.unsqueeze(1).repeat(1,batchSize,1)), dim = 2)
        reshape_tensor = cell_timeSeries.view(cell_timeSeries.shape[0]*cell_timeSeries.shape[1], cell_timeSeries.shape[2])
        rho_timeSeries = model.getGamma(reshape_tensor)
        rho_timeSeries = rho_timeSeries.view(cell_timeSeries.shape[0], cell_timeSeries.shape[1], NvarGenes)

        random_gene = int(torch.rand([1]) * NvarGenes)
        gg = torch.autograd.grad(outputs=rho_timeSeries[:,0,random_gene].sum(), inputs=t_sample, retain_graph=True, create_graph=True)[0] 
        o = t_sample.argsort()
        twoD = torch.diff(gg[o] / rho_timeSeries[:,0,random_gene].detach().mean())
        loss_smoothness2 = torch.clamp(twoD.abs().max(), max = 1)



        '''
        losses
        '''
        loss_latent_space = 0.5 *  ((model.latent_mean[0:-time_dim]**2).sum()  - torch.log(model.latent_std[0:-time_dim].prod() + 1e-10) - Ndim + model.latent_std[0:-time_dim].sum() )
        loss_reconstruction = ((thisCell - thisCell_hat)**2).sum(dim = 1).mean()
        loss_reconstruction = loss_reconstruction.sqrt() / NvarGenes
        loss_time =((cellTime - model.estimated_time[:,0])**2).mean()
        loss_push_time = ((estimated_time - random_push)**2).mean()


        loss_final =  -model.meanlogprob   + 10 * loss_time + 5e4 * (W_dist  + W_dist_perT.mean()) + 100 * W_dist_pushed + smoothPower * loss_smoothness2 + 1 * loss_push_time 
        
        loss_final.backward()
        optimizer_AE.step()


        if np.mod(counter, 100) == 0:
            torch.save(model, model_name)

        if np.mod(counter, 50) == 0:
            print('Iter: {}, loss_re: {:.4f}, loss_log: {:.4f}, loss KL: {:.4f}, time prediction: {:.4f}, wasserstein:{:.4f}'
                  .format(counter, loss_reconstruction, -model.meanlogprob.detach().numpy() / NvarGenes, model.KL_loss.mean(), loss_time, W_dist.mean()))    

        counter += 1

    print('done training')
    return model
    
