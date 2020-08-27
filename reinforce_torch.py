import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
    #66, Alpha is the learning rate, which is used for policy update 
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions  # 66, the n_actions is just a number 
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        # 66, nn.Linear just applies a linear transformation of the incoming data 
        # notice later on, we use F.relu as the non-linear activation function 
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        #66, this is where the nn module comes in, it will optimize the parameters of our 
        # network, even if we do not explicitly assign it

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1') # check GPU availability
        self.to(self.device) #66, send the entire network to the device (GUP or CPU)

    def forward(self, observation):
        '''
        we need to know the required type of observation that can be converted to tensor 
        '''
        state = T.Tensor(observation).to(self.device) # Tensor converts the numpy observation to a tensor
        x = F.relu(self.fc1(state)) 
        #66 applies the non-linear Relu activation function on the result of linear transformation 
        x = self.fc3(x) #66, why layer 3 not need a non-linear transformation Relu?? 
        return x

class PolicyGradientAgent(object):
    # 66, an agent has a policy network, but it also has memory to choose actions and some
    # funcionality for learning 
    def __init__(self, ALPHA, input_dims, GAMMA=0.99, n_actions=4,
                 layer1_size=256, layer2_size=256):
        self.gamma = GAMMA # the discount factor for future reward
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(ALPHA, input_dims, layer1_size, layer2_size,
                                    n_actions) 

    def choose_action(self, observation):
        '''
        the policy network only outputs the action distribution, and might optionally sample real actions
        But it is the env object which takes in the action distribution and execute the real actions
        Thus the env would take the output of choose_action() of the policy network
        
        choose_action function returns an action vector for all traffic lights

        the observation is a dictionary with key of ts id, thus we need to iterate over the ts id, qeury its 
        local observation, apply the policy nn, and find the corresponding action for the ts. 

        '''

        ################ for the representative agent selected for training ##################
        ##66, maybe a better idea is to randomly pick an agent from all identical agents, not hard code 
        probabilities = F.softmax(self.policy.forward(observation['F5'])) # hard-coded 
        action_probs = T.distributions.Categorical(probabilities) # a vector 
        action = action_probs.sample()        
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs) # important step, for learning
        # 66. action memory is saving the log probability of the action distribution


        ############# also need to generate actions for other agents ##################################
        Action_dict = {} # 33, consider remove the above agent
        for key in observation:
            probabilities = F.softmax(self.policy.forward(observation[key])) # probabilities have diamension 
            action_probs = T.distributions.Categorical(probabilities) # a vector 
            action = action_probs.sample() # action is a sampled action, such as giving green to NS direction
            # one is turning to north-south, the other is west-east
            Action_dict[key] = action.item() # item() is copying the element of an array


        return Action_dict #66, returns a dictionary of integers

    def store_rewards(self,reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad() #66, this will clear your gradient from last time step
        # 66, it will prevent slowing down your leanring process
        # Assumes only a single episode for reward_memory
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        # 66, every step it calculates the cumulative future reward 
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        # 66, there is significant amount of variance in rewards, we need to normalize the reward
        # deep neural network does not like significant variance in the inputs
        # therefore we need to scale and normalize the rewards 
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = T.tensor(G, dtype=T.float).to(self.policy.device) #66, transform numpy array to a Tensor

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            # zip returns an iterator of tuples where the first item in each passed iterator 
            # is paired together, and then the second item in each passed iterator are paired together 
            loss += -g * logprob

        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []
        # 66, we need to zero out the action and reward memory 
