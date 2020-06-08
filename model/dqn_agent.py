import numpy as np
import numpy.random as random
import random as random0

from collections import namedtuple, deque

from .model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''Interacts with and learns from the environment.'''

    def __init__(self, state_size, action_size, seed, embedding_size, 
                    neighbours,
                    nu = None, dropout = None, model = QNetwork ):
        '''Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.embedding_size = embedding_size
        self.neighbours = neighbours
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_online = model(state_size, action_size, 
                                       seed, nu, dropout).to(device)
        self.qnetwork_target = model(state_size, action_size,
                                        seed, nu, dropout).to(device)
        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=LR)

        self.criterion = F.mse_loss
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(np.expand_dims(state,0), action, reward, np.expand_dims(next_state,0), done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        '''Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_online.eval()
        with torch.no_grad():
            action_values = self.qnetwork_online(state)
        self.qnetwork_online.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return self.get_actions(action_values)
            #np.argmax(action_values.cpu().data.numpy())
        else:
            return self.random_choice()

    def get_actions(self,action_values):
        idx = np.argmax(action_values.cpu().data.numpy())
        move = idx % 3
        spin_1 = (idx//3) % self.embedding_size
        spin_2 = (idx//3)//self.embedding_size
        return (move, spin_1, spin_2)

    def random_choice(self):
        move = random.randint(3)
        spin_1 = random.randint(self.embedding_size)
        spin_2 = random.randint(self.neighbours)
        return (move, spin_1, spin_2)
        
    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        yhat = rewards
        
        qtargetnext, _ = torch.max(self.qnetwork_target(next_states).detach(),1)
        qtargetnext = qtargetnext.unsqueeze(1)
        
        qtarget = rewards + gamma*qtargetnext*(1-dones)
        
        qexpect = self.qnetwork_online(states).gather(1,actions)
        
        loss = self.criterion(qexpect, qtarget)
        
        # Update parameters and gradients to zero
        # Compute the gradients
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU)                     

    def soft_update(self, online_model, target_model, tau):
        '''Soft update model parameters.
        network_target = tau*network_online + (1 - target)*network_target

        Params
        ======
            online_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        '''
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)

class AgentDoubleQ(Agent):
    '''Interacts with and learns from the environment.'''

    def __init__(self, state_size, action_size, 
                    seed, nu = None, dropout = None, model = QNetwork):
        '''Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            e: parameter for priority pi = |tderror| + e
            a: parameter for sampling priority pi^a/sum pk^a 
        '''    
        super().__init__(state_size, action_size, seed, nu, dropout, model)
        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), 
                                    lr=.5e-4)
        self.e = e
        self.a = a

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        
        qonlinenext, argnext = torch.max(self.qnetwork_online(next_states).detach(),1)
        # This line introduces the double Q-learning, another way would be using another
        # network
        qtargetnext =  self.qnetwork_target(next_states).detach().gather(1,argnext.unsqueeze(1))
        
        qtarget = rewards + gamma*qtargetnext*(1-dones)
        
        qexpect = self.qnetwork_online(states).gather(1,actions)
        
        loss = self.criterion(qexpect, qtarget)
        
        # Update parameters and gradients to zero
        # Compute the gradients
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
                
                
class AgentRainbow(Agent):
    '''Interacts with and learns from the environment.'''

    def __init__(self, state_size, action_size, 
                    seed, nu = None, dropout = None, model = QNetwork,
                    e = 1e-1, alpha = 0.5, beta = 1.0 ):
        '''Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        '''    
        super().__init__(state_size, action_size, seed, nu, dropout, model)
        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), 
                                    lr=.5e-4)
        
        self.memory = ReplayBufferPrioritized(action_size, 
                                              BUFFER_SIZE, BATCH_SIZE, seed)
        self.alpha = alpha
        self.beta = beta
        self.e = e

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        '''
        states, actions, rewards, next_states, dones, indices, weights = experiences

        yhat = rewards
        
        qtargetnext, _ = torch.max(self.qnetwork_target(next_states).detach(),1)
        qtargetnext = qtargetnext.unsqueeze(1)
        
        qtarget = rewards + gamma*qtargetnext*(1-dones)
        
        qexpect = self.qnetwork_online(states).gather(1,actions)

        #loss = self.criterion(qexpect, qtarget)
        loss = (qexpect - qtarget).pow(2)*weights
        
        priorities = loss.detach().numpy()*1.0 # With or without weights?
        
        loss = loss.mean()
        
        # Update parameters and gradients to zero
        # Compute the gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(indices, priorities)
        
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_online, self.qnetwork_target, TAU) 
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample_prioritized(self.e, 
                                                             self.alpha,self.beta)
                self.learn(experiences, GAMMA)
                
                



class ReplayBuffer:
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                            field_names=["state", "action", "reward", 
                                        "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        experiences = random0.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)

    
    
class ReplayBufferPrioritized(ReplayBuffer):
    '''Fixed-size buffer to store experience tuples with prioritized replay.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer with Prioritized replay object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        '''
        super().__init__(action_size, buffer_size, batch_size, seed)

        self.experience = namedtuple("Experience", 
                            field_names=["state", "action", "reward", 
                                        "next_state", "done"])
        self.priorities = deque(maxlen=buffer_size) 
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(self.max_priority)
    
    def update_priorities(self, indices, tderrors):
        '''Update priorities array with new tderrors'''
        self.max_priority = np.max([self.max_priority,tderrors.max()])
        for idx, tde in zip(indices,tderrors):
            self.priorities[idx] = tde
        
    def sample_prioritized(self, e, alpha, beta):
        '''Randomly sample a batch of experiences from memory with priority.'''
        
        # Calculating probabilities for priorities
        tderrors = np.asarray(self.priorities, dtype = np.float32).flatten()
        pis = (np.abs(tderrors)+e)**alpha
        pis = pis/pis.sum() 
        
        # Random indices with probabilities pis
        indices = np.random.choice(len(self.memory), size=self.batch_size, p = pis)
        
        # Selecting episodes from memory
        states = torch.from_numpy(np.vstack([self.memory[idx].state for idx in indices])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx].action for idx in indices])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx].reward for idx in indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx].next_state for idx in indices])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx].done for idx in indices]).astype(np.uint8)).float().to(device)
        
        # Importance sampling 
        weights = 1.0/(len(tderrors)*pis[indices])**(beta)
        # Reshape is needed because the flattening at the beginning screw dimensions
        weights = (weights/weights.max()).reshape(weights.shape[0],1)
        weights = torch.from_numpy(weights).float().to(device)
        
        
        return (states, actions, rewards, next_states, dones, indices, weights)


        