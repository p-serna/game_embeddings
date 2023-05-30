import torch
import torch.nn as nn
import torch.nn.functional as F

def createblocklist(nu, dropout = None):
    '''This function return a list of modules to be implemented as a sequential layer block
    Input variables:
    nu: array with number of filters (or strides for pooling layers) for different layers of the network
    
    Output:
    list of pytorch layers
    '''
    # Initial conv+relu that takes previous number of filters to the new one
    if dropout is None:
        modlist = [[nn.Linear(nu[i],nu[i+1]),nn.ReLU()] for i in range(len(nu)-1)]
    else:
        modlist = [[nn.Linear(nu[i],nu[i+1]),nn.ReLU(), dropout] 
                   for i in range(len(nu)-1)]
    
    modlist = [x for sublist in modlist for x in sublist]
    return modlist



class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, nu = None, nconv = 3, dropout = None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        if nu is None:
            nu = [32, 64, 128, 512]
        self.nu = nu
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
		

        self.FC0 = nn.Linear(512,nu[0])
        self.FClist = nn.Sequential(*createblocklist(nu, self.dropout))
        self.FCf = nn.Linear(nu[-1], action_size)        
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax()

        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        
        x = self.FC0(x)
        x = self.relu(x)
        x = self.FClist(x)
        x = self.FCf(x)
        x = self.Softmax(x)

        return x


class QNetwork_Conv(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, nu = None, nconv = 3, dropout = None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_Conv, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        if nu is None:
            nu = [32, 64, 128, 512]
        self.nu = nu
        
        kernels = [3]*nconv

        modlist = [nn.Conv2d(2,nu[0]
                      ,kernels[0], padding = kernels[0]//2),
                   nn.ReLU(),
                   nn.MaxPool2d(kernels[0],
                        stride = 2)]
        extendlist = [[nn.Conv2d(nu[i-1],nu[i]
                      ,kernels[i], padding = kernels[i]//2),
                   nn.ReLU(),
                   nn.MaxPool2d(kernels[i],
                        stride = 2)] for i in range(1,nconv)]
        extendlist = [x for sublist in extendlist for x in sublist]
        modlist.extend(extendlist)
        
        self.convblock0 = nn.Sequential(*modlist)
        
        self.FC0 = nn.Linear(nu[nconv-1],nu[nconv])
        self.FClist = nn.Sequential(*createblocklist(nu[(nconv):]))
        self.FCf = nn.Linear(nu[-1], action_size)        
        self.FCv0 = nn.Linear(nu[nconv], 32)        
        self.FCv1 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax()
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.convblock0(x)
        x = torch.mean(x, 2)
        x = torch.mean(x, 2)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        
        x = x.view(-1, shape)
        
        x = self.FC0(x)
        x = self.relu(x)
        adv = self.relu( self.FCv0(x))
        adv = self.relu( self.FCv1(adv))
        advmean = adv.mean(1,keepdim=True)

        x = self.FClist(x)
        x = self.FCf(x)
        x = x+adv-advmean
        x = self.Softmax(x)
    
        return x
    
class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, 
                    nu = None, nu_state = None, nu_adv = None, dropout = None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            nu : list of number of units of a sequential list of FC layers 
            nu_state : similar list for the part of the state value Sv 
            nu_adv : similar list for the Advantage
            nu : number of units 
        """
        super(DuelQNetwork, self).__init__() 
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        if nu is None:
            nu = [32, 64, 128, 512]
        self.nu = nu
        if nu_state is None:
            nu_state = [128]
        self.nu_state = nu_state
        if nu_adv is None:
            nu_adv = [128]
        self.nu_adv = nu_adv

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
		

        self.FC0 = nn.Linear(state_size,nu[0])
        self.FClist = nn.Sequential(*createblocklist(nu, self.dropout))

        nut = [nu[-1], *nu_state]
        self.FClstate = nn.Sequential(*createblocklist(nut, self.dropout))
        self.FCfstate = nn.Linear(nut[-1], 1)
        nut = [nu[-1], *nu_adv]
        self.FCladv = nn.Sequential(*createblocklist(nut, self.dropout))
        self.FCfadv = nn.Linear(nut[-1], action_size)

        self.relu = nn.ReLU()
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.FC0(state)
        x = self.relu(x)
        x = self.FClist(x)
        
        state = self.FClstate(x)
        state = self.FCfstate(state)
        
        adv = self.FCladv(x)
        adv = self.FCfadv(adv)
        
        # Removing ambiguous constant
        advmean = adv.mean(1,keepdim=True)
        return state+adv-advmean
        
