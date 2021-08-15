# Project Done By:
# Mohammed Junaid Alam (2019503025)
# Gurbani Bedi (2019503518)
# Abhishek Manoharan (2019503502)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as functional
from random import sample

class Network(nn.Module):
    
    def __init__(self,nb_inputs,nb_actions):
        super(Network,self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_actions
        self.first_connection = nn.Linear(self.nb_inputs,30)
        self.second_connection = nn.Linear(30,self.nb_outputs)
        
    def forward(self,state):
        fc1_activated = functional.relu(self.first_connection(state))
        q_values = self.second_connection(fc1_activated)
        return q_values
    
class Memory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self,state):
        self.memory.append(state)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def pull(self,batch_size):
        rand_sample = zip(*sample(self.memory,batch_size))
        return map(lambda x:Variable(torch.cat(x,0)),rand_sample)
    
class Brain():
    
    def __init__(self,input_nodes,nb_actions,gamma):
        self.gamma = gamma
        self.reward_mean = []
        self.memory = Memory(100000)
        self.model = Network(input_nodes,nb_actions)
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        self.last_state = torch.Tensor(input_nodes).unsqueeze(0)
        self.last_reward = 0
        self.last_action = 0
        
    def select_action(self,state):
        probs = functional.softmax(self.model.forward(Variable(state,volatile=True))*100)
        action = probs.multinomial(1)
        return action.data[0,0]
    
    def learn(self,prev_state,current_state,prev_action,prev_reward):
        outputs = self.model.forward(prev_state).gather(1,prev_action.unsqueeze(1)).squeeze(1)
        max_futures = self.model.forward(current_state).detach().max(1)[0]
        targets = self.gamma*max_futures + prev_reward
        loss = functional.smooth_l1_loss(outputs,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update(self,prev_reward,current_state):
        new_state = torch.Tensor(current_state).float().unsqueeze(0)
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            train_last_state,train_next_state,train_last_action,train_last_reward = self.memory.pull(100)
            self.learn(train_last_state,train_next_state,train_last_action,train_last_reward)
        self.last_state = new_state
        self.last_action = action
        self.last_reward = prev_reward
        self.reward_mean.append(prev_reward)
        if len(self.reward_mean) > 1000:
            del self.reward_mean[0]
        return action
    
    def score(self):
        mean = sum(self.reward_mean)/(len(self.reward_mean) + 1.0)
        return mean
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict
                    },'recent_brain.pth')
    
    def load(self):
        if os.path.isfile('recent_brain.pth'):
            print('---Loading checkpoint---')
            checkpoint = torch.load('recent_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])    
            print('Model successfully loaded!')
        else:
            print('There isn\'t an instance of a saved model!')
    

        