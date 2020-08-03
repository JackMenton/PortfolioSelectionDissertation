# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:50:06 2020

@author: jackm

This notebook has solves the problem with a utility function and Nueral Networks
for the value function and a input feature approximation for the policy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set seeds for reproducability
np.random.seed(0)
torch.manual_seed(0)

dt = 1/52 #  time increments
T = 2*dt # time horizon
x_0 = 1.0 # initial wealth
N = math.floor(T/dt) # number of time steps
EPOCHS = 10000 # number of training episodes

# Some Parameters
mu = 0.1 # drift of stock
sigma = 0.1 # volatility of stock
r =  0.02 # risk free rate
rho = (mu - r)/sigma # sharpe ratio
lam = 0.1 # temperature parameter for entropy
z = 1.1 # desired rate of return
gamma = 0.5 # gamma for power utility function
            
class ValueFuncNetwork(nn.Module):
    ''' Neural Network for critic, estimating the value function '''
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_returns):
        super(ValueFuncNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_returns = n_returns
        self.lr = lr
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # input is wealth and time to maturity 
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,n_returns) # output is value of the state
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        
    def forward(self, observation):
        state = torch.Tensor(observation).float()
        x = F.leaky_relu(self.fc1(state), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01) # leaky relu to help with derivs
        return x     
    
class Agent(object):
    def __init__(self, alpha, beta, starting_mean_w, starting_sd_w, input_dims, gamma = 1, l1_size = 256, l2_size = 256):
        self.gamma = gamma
        self.reward_memory = [] #  to store episodes rewards
        self.deriv_log_mu_memory = np.zeros(2) # to store episodes actions
        self.deriv_log_sd_memory = np.zeros(2) # to store episodes actions
        self.value_memory = [] # to store value of being in each state
        self.mean_memory = []
        self.sd_memory = []
        self.value = ValueFuncNetwork(beta, input_dims, l1_size, l2_size, n_returns = 1)
        self.mean_weight = starting_mean_w
        self.sd_weight = starting_sd_w
        self.alpha = alpha
        
    def choose_action(self, current_wealth, t):
        '''sample action, given state'''       
        self.mean, self.sd = mean_sample(self.mean_weight, current_wealth), sd_sample(self.sd_weight, T-t)
        action_dist = torch.distributions.normal.Normal(self.mean,self.sd) # define distribution
        action = action_dist.sample() # sample from distribution     
        self.deriv_log_mu_memory = np.vstack([self.deriv_log_mu_memory, deriv_log_mean(action.item(), current_wealth, self.mean_weight, self.sd_weight)]) # store log probs
        self.deriv_log_sd_memory = np.vstack([self.deriv_log_sd_memory, deriv_log_sd(action.item(), current_wealth, self.mean_weight, self.sd_weight)]) # store log probs        
        self.reward = lam*action_dist.entropy().item() # store running reward
        
        return action.item()
    
    def get_value(self, current_wealth, t):
        ''' get value of state'''
        state = [current_wealth, T-t]
        value_ = self.value.forward(state)
        return value_
    
    def store_value(self, value):
        '''store values'''
        self.value_memory.append(value)
    
    def store_rewards(self, reward):
        '''store rewards'''
        self.reward_memory.append(reward)
    
    def store_means(self, mean):
        '''store rewards'''
        self.mean_memory.append(mean)

    def store_sds(self, sd):
        '''store rewards'''
        self.sd_memory.append(sd)
        
    def learn(self):
        '''learn - done after each episode'''
        self.value.optimizer.zero_grad()
        
        deltas = [] # to store delta values
        G = [] # to store gain values
        for j in range(N):
            R = 0
            for k in range(j,N):
                R += self.reward_memory[k+1]
            G.append(R)
        for j in range(N):
            deltas.append(G[j] - self.value_memory[j].item())
        #mean = np.mean(deltas)
        self.score = G[0]
        #std = np.std(deltas) if np.std(deltas) > 0 else 1
        #deltas = (deltas-mean)/std
        
        
        #deltas.requires_grad = False
        
        '''obtain total losses'''
        val_loss = 0
        for d, vals in zip(deltas, self.value_memory):
            val_loss += -d*vals
        
        policy_mean_loss = [0,0]
        for d, logmeanprob in zip(deltas, self.deriv_log_mu_memory[:,]):
            policy_mean_loss[0] += d*logmeanprob[0]
            policy_mean_loss[1] += d*logmeanprob[1]
            
        policy_sd_loss = [0,0]
        for d, logsdprob in zip(deltas, self.deriv_log_sd_memory[:,]):
            policy_sd_loss[0] += d*logsdprob[0]
            policy_sd_loss[1] += d*logsdprob[1]
            
        val_loss.backward() # compute gradients
        
        # take steps
        self.value.optimizer.step() 
        self.mean_weight[0] += self.alpha*policy_mean_loss[0]
        self.mean_weight[1] += self.alpha*policy_mean_loss[1]
        self.sd_weight[0] += self.alpha*policy_sd_loss[0]
        self.sd_weight[1] += self.alpha*policy_sd_loss[1]
                
        # empty caches
        self.reward_memory = []
        self.action_memory = []
        self.value_memory = []
        self.deriv_log_mu_memory = np.zeros(2) 
        self.deriv_log_sd_memory = np.zeros(2)

def mean_sample(mean_weights, current_wealth):
    return mean_weights[0]*current_wealth + mean_weights[1]

def sd_sample(sd_weights, t):
    '''Returns softplus function instead of regular exp function as 
    exp function was rounding to zero and causing problems'''
    #np.exp(sd_weights[0]*(T-t) + sd_weights[1])
    x = sd_weights[0]*(T-t) + sd_weights[1]
    return np.log(1+np.exp(-abs(x))) + max(x,0) # softplus

def deriv_log_mean(action, current_wealth, mean_weights, sd_weights):
    main = (action-mean_sample(mean_weights, current_wealth))/(sd_sample(sd_weights, t)**2)
    d1 = main*current_wealth
    d2 = main
    return d1, d2

def deriv_log_sd(action, current_wealth, mean_weights, sd_weights):
    main = (((action-mean_sample(mean_weights, current_wealth))/sd_sample(sd_weights, t))**2 - 1)
    d1 = main*(T-t)
    d2 = main
    return d1, d2
    
def wealth( x, sample):
    '''obtain new wealth sample'''
    x_new =  x + sigma*sample*(rho*dt + np.sqrt(dt)*np.random.randn())
    return x_new

def wealth2(x, sample):
    '''wealth process when sample is fraction of wealth'''
    x_new = x + x*(sample*(mu-r)+r)*dt + x*sample*sigma*np.sqrt(dt)*np.random.randn()
    return x_new

def util(x):
    '''utility function'''
    return x**gamma

def true_value(x, t):
    beta = (gamma*rho**2)/(2*(1-gamma))
    y = np.exp(beta*(T-t))*x**gamma
    return y

def true_mean(x):
    y = (rho*x)/(sigma*(1-gamma))
    return y 


def surface_plot(matrix1, matrix2, x_vec, y_vec, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(x_vec, y_vec)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf1 = ax.plot_surface(x, y, matrix1, label = 'Approximated Surface', **kwargs)
    surf2 = ax.plot_surface(x, y, matrix2, label = 'True Surface', **kwargs)
    return (fig, ax, surf1, surf2)
# ------ Main ------ # 
'''
    Learning rates:
        lam = 0.1 , alpha = 0.5, beta = 0.005, T = 2*dt, x**(0.5) utility
        lam = 0.001 , alpha = 0.5, beta = 0.005, T = 2*dt, x**(0.5) utility
        
        
'''
# keep beta at around 0.005 as thats when delta amounts seem to be small
# dont go above 0.3 for alpha or performance starts deteriorating
agent = Agent(alpha = 0.5, beta = 0.005, starting_mean_w = [10,0], starting_sd_w = [0,0], input_dims = [2], gamma = 1, l1_size = 32, l2_size = 32)

# -------------- TRAINING ----------- #
episode_scores = np.array([])
episode_values = []
for epoch in range(EPOCHS):
    episode_wealths = []
    curr_wealth = x_0
    #curr_wealth = np.random.uniform(0.0,1.5)
    for i in range(N):
        t = i*dt
        episode_wealths.append(curr_wealth)
        value = agent.get_value( curr_wealth, t) # get value of current state from NN
        action = agent.choose_action( curr_wealth, t) # choose actions 
        new_wealth = wealth(curr_wealth, action) # obtain new wealth
        agent.store_value(value) # store this value
        agent.store_rewards(agent.reward) # store reward from taking this action
        if i == 0:
            episode_values.append(value.item())
            agent.store_sds(agent.sd)
            agent.store_means(agent.mean)
        if new_wealth < 0:
            new_wealth = 0
        curr_wealth = new_wealth # set new wealth to current wealth
        if curr_wealth == 0:
            for k in range(N-1-i):
                agent.store_rewards(0)
                agent.store_value(torch.Tensor([0]))
            break
    agent.store_rewards(util(new_wealth)) # add terminal wealth to 
                 
    agent.learn()
    episode_scores = np.append(episode_scores,agent.score)

# ----------- TESTING ---------- #
'''
terminal_wealths = []
for epoch in range(int(EPOCHS/10)):
    episode_wealths = []
    curr_wealth = x_0
    for i in range(N):
        t = i*dt
        episode_wealths.append(curr_wealth)
        action = agent.choose_action( curr_wealth, t) # choose actions 
        new_wealth = wealth(curr_wealth, action) # obtain new wealth
        if new_wealth < 0:
            new_wealth = 0
        curr_wealth = new_wealth # set new wealth to current wealth
        if curr_wealth == 0:
            break
    terminal_wealths.append(new_wealth)

mean_tw = np.mean(terminal_wealths)
'''

# This is where .grad is stored
value_params = list(agent.value.parameters()) 

textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ),
    r'$r=%.2f$' % (r, ),
    r'$\sigma=%.2f$' % (sigma, ),
    r'$\rho=%.2f$' % (rho, ),
    r'$\lambda=%.2f$' % (lam, )))



'''
plt.figure()
plt.plot(range(EPOCHS), episode_scores)
plt.title('Learning Curve for Reinforce with Baseline - T = 0.5 year')
plt.xlabel('Episodes')
plt.ylabel('G_0 - total reward on episode')
'''

'''
plt.figure()
plt.plot(range(EPOCHS), terminal_wealths)
plt.title('Terminal Wealth - T = 0.5 year')
plt.xlabel('Episodes')
plt.ylabel('Terminal Wealth')
'''

plt.figure()
plt.plot(range(EPOCHS), agent.mean_memory, label = 'episode mean control')
plt.axhline(y=true_mean(x_0), color='r', linestyle='-', label = 'optimal control')
plt.title('Learning Curve Mean of Policy Distribution (of initial state)')
plt.xlabel('Episodes')
plt.ylabel('Mean')
plt.text(EPOCHS-(EPOCHS/10),2.5, textstr)
plt.legend()

'''
plt.figure()
plt.plot(range(EPOCHS), agent.sd_memory)
plt.title('Learning Curve SD of Policy Distribution')
plt.xlabel('Episodes')
plt.ylabel('sd')
'''
# convergence graph for value of inital state given analytical solution
# should tend toward red line

plt.figure()
plt.plot(range(EPOCHS), episode_values, label = 'episode values')
plt.axhline(y=true_value(1,0), color='r', linestyle='-', label = 'optimal value')
plt.title('Value Network Convergence (of initial state)')
plt.xlabel('Episodes')
plt.ylabel('value of inital state')
plt.text(EPOCHS-(EPOCHS/10),0.2, textstr)
plt.legend()

# ------- 3D Surface Splot -------- #

'''
x_points = list(np.linspace(0.0, 1.2, 100))
t_points = list(np.linspace(0, T, 100))
tmat_points = [T-i for i in t_points]
values = np.zeros((len(x_points),len(t_points)))
true_values = np.zeros((len(x_points),len(t_points)))
mean_values = []
true_means = []

xg,tg = np.meshgrid(x_points,tmat_points)

for i in range(len(t_points)):
    for j in range(len(x_points)):
        values[i,j] = agent.value.forward(torch.Tensor([x_points[j],tmat_points[i]]))
        true_values[i,j] = true_value(x_points[j], t_points[i])
        
for i in range(len(x_points)):
    mean_values.append(mean_sample(agent.mean_weight, x_points[i]))
    true_means.append(true_mean(x_points[i]))


plt.figure()
plt.plot(x_points, mean_values, c = 'r', label = 'Approx Means')
plt.plot(x_points, true_means, c = 'b', label = 'Optimal Means')
plt.legend()
plt.xlabel('Wealth')
plt.ylabel('Mean of Policy Dist.')
plt.show()


(fig, ax, surf1, surf2) = surface_plot(values, true_values, x_points, tmat_points)#, cmap=plt.cm.coolwarm)
#(fig1,ax1,surf1) = surface_plot(true_values, x_points, tmat_points)

#fig.colorbar(surf1)
#fig.colorbar(surf2)

ax.set_xlabel('Wealth (cols)')
ax.set_ylabel('T-t (rows)')
ax.set_zlabel('Value')
#fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
fake2Dline2 = mpl.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
#ax.legend([fake2Dline], ['True Surface'], numpoints = 1)
ax.legend([fake2Dline2], ['True Surface'], numpoints = 1)


plt.show()
'''

