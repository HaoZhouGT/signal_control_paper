
'''

Episodic setting
Reinforce algorithm
using pytorch

The main funciton only calls for env.reset() and env.step(action)
In fact env is just an object that can use the reset function to get observation and 
call the step function to run the simulation, output the next state, and reward 

Therefore, defining the env function is just to define some  member functions related to
the simulation process. 

'''


from signalEnv import SumoEnvironment
import traci

import numpy as np
import gym
from reinforce_torch import PolicyGradientAgent # 66 import a class 
import matplotlib.pyplot as plt
# from utils import plotLearning
from gym import wrappers # 66, this is to save the footage of the model 

if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA=0.001, input_dims=[8], GAMMA=0.99,
                                n_actions=2, layer1_size=64, layer2_size=64)
    #66, output should be a probability distribution over the two actions
    # two actions are giving green to north-south direction or east-west direction

    #agent.load_checkpoint()
    env = SumoEnvironment(net_file='grid.net.xml',
                          route_file='grid.rou.xml',
                          sumo_cfg = 'grid.sumo.cfg',
                          target_density = 0.1, # let's try free flow first
                          use_gui=False,
                          num_seconds=40000, # how long one episode will last? 
                          phases=[
                              traci.trafficlight.Phase(35, "GGggrrrrGGggrrrr"),   # north-south
                              traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
                              traci.trafficlight.Phase(35, "rrrrGGggrrrrGGgg"),   # west-east
                              traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
                            ],
                          time_to_load_vehicles=300, # for vehicle to accumulate
                          green_time=50)
    score_history = []
    score = 0
    num_episodes = 2000
    #env = wrappers.Monitor(env, "tmp/lunar-lander",
    #                        video_callable=lambda episode_id: True, force=True)
    for i in range(num_episodes):
      '''
      # 66 the logic is that, environment is used for getting the observation and compute the reward
      # the agent is another object, it takes in the observation and use the policy nn to execute actions
      '''
      print('episode: ', i,'score: ', score)
      done = False
      score = 0
      ################ parse the observation matrix ################
      observation = env.reset() # observation should be a dictionary, with n elements, value is 8 vector
      while not done:
          ############### chooose actions for all ts based on full observations ####################
          action = agent.choose_action(observation) #66, choose action based on its observation
          # the action should be an action vector for all ts 
          # we need to correspond the action to the real action (green light phase)
          print('see the action dictionary of all agents (ts): ',action)

          ############# parse the action integer and run the simulation in SUMO #####################
          observation_, reward, done, info = env.step(action) # should have a loop for those actions
          # step function need to take an action vector and execute all the actions
          # 66 Need to compute new state, compute reward after execute one action
          agent.store_rewards(reward) # store the reward
          observation = observation_ # update the state (observation)
          score += reward
      score_history.append(score)
      agent.learn() # 66, this is the important part, agent learns in every episode 
      #agent.save_checkpoint()
  # filename = 'grid9x9.png'
  # plotLearning(score_history, filename=filename, window=25)
