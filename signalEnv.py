import os
import sys
import random
import time
import math

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from traffic_signal import TrafficSignal
#66, we need such a traffic light object to retrieve traffic light 
# and also apply the actions related to phase

from side_functions import child
from side_functions import opposite_lane
from side_functions import netLength


from set_veh_num_limit import gene_config  
#66. this is use sumo.cfg file to control the total vehicle number




class SumoEnvironment(MultiAgentEnv):

    def __init__(self, net_file, sumo_cfg, phases, training_density, use_gui=False,
                  green_ratio=1.0, vehicle_length=4.5,min_gap=2.0):

        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self._cfg = sumo_cfg
        self._net = net_file
        self.net = sumolib.net.readNet(self._net)
        self._density = training_density 
        self._vehicle_length = vehicle_length
        self._min_gap = min_gap
        self.jam_density = int(1000/(self._vehicle_length+self._min_gap))
        self.netLength = netLength(self.net)
        self.target_accumulation=int(self._density*self.jam_density*self.netLength)


        self.phases = phases  
        self.yellow_time = 2

        EdgeList = [self.net.getEdges()[i].getID() for i in range(len(self.net.getEdges()))]
        LaneList = [str(r)+'_0' for r in EdgeList]
        self.Num_detectors = len(EdgeList)
        self.EdgeList = EdgeList
        self.LaneList = LaneList

        gene_config(self.net, training_density,vehicle_length,min_gap) # this will write a new grid.sumo.cfg file
        time.sleep(2) # give some time to generate the new config file

        #####################3 green time should be computed using the parameter lambda ##################
        # ratio = E(l)/E(g) # here is a limitation, the green light has to be identical in simualtion 
        self.green_ratio = green_ratio
        edges = self.net.getEdges()
        sum=0
        count=0
        for edge in edges:
            sum+=edge.getLength()
            count+=1
        average_block_length = sum/count
        print('Initializing the network, the average block length is ',average_block_length)
        self.green_time = int(average_block_length/self.green_ratio) # green length is computed
        print('According to lambda, average green time is:', self.green_time)

        #get the all traffic light ID as an attribute of the class ###########
        traci.start([sumolib.checkBinary('sumo'), '-c', self._cfg]) # we need to start traci to query value
        self.ts_ids = traci.trafficlight.getIDList() # get all traffic light ID 
        traci.close()

        
    def reset(self):
        '''
        we need to initialize the simulation in the reset() function
        need to call traci to generate vehicles and let the network reach a certain density 
        level need to route vehicles to avoid congestion 
        '''
        print('Initialize a new episode')
        print('initialize the traffic, insert vehicles to target density')

        sumo_cmd = [self._sumo_binary, '-c', self._cfg]
        if self.use_gui:
            sumo_cmd.append('--start') # --start can automatically start traci instead
            sumo_cmd.append('--quit-on-end') # of requiring you click the button
        traci.start(sumo_cmd) # must start traci to execute traci commands

        ################## the following code is to generate vehicles and reach the target density ############################
        rou_id = 0 # let rou_id start from 0, and vehicles accumulate
        veh_count = 0
        while traci.vehicle.getIDCount()<self.target_accumulation:
            self._sumo_step()
            ############################ random routing of vehicles at every time step ################
            for edge in self.EdgeList:
                target_edge_list= child(edge,self.net) # this is downstream edge list
                next_edge = random.choice(target_edge_list) # random turing at intersections
                next_next_edge = random.choice(child(next_edge,self.net))
                lane_id = edge+'_0'
                if traci.lane.getLastStepVehicleNumber(lane_id) >=1: # if there exist vehicles on the lane
                	traci.vehicle.setRoute(traci.lane.getLastStepVehicleIDs(lane_id)[-1], [edge, next_edge, next_next_edge])

            ######################## generate and add new vehicles to SUMO #############################
            for edge in self.EdgeList:
                route_id= 'route_'+str(rou_id)
                rou_id += 1
                target_edge_list= child(edge,self.net)
                next_edge = target_edge_list[0]
                next_next_edge = child(next_edge,self.net)[0]
                traci.route.add(route_id,[edge,next_edge,next_next_edge])
                veh_id = 'add_new_veh_id'+str(veh_count)
                veh_count += 1
                typeID="MFD"
                # we can slow down the frequency of generating such vehicles 
                traci.vehicle.add(veh_id,route_id,typeID)    
		#66######################### return the inital observation after reset()##################
        self._sumo_step()
        print('target density achieved, the real accumulation is %s'%traci.vehicle.getIDCount())
        # print('after reaching the target density, the initial observation is:')
        observations = self._compute_observations()
        # traci.close()
        return observations
 


def reset(self, Initial_conditions):
    # restart the simulator with same initial conditions
    env.start(Initial_conditions)

    # might need some warm-up steps    
    for i in range(N):
        env.step()

    # take snapshot as the initial observation
    new_state = env.snapshot 

    return new_state




def step(self,action):
    # apply the action to the simulator
    env.apply_action(action) 

    # some code to calculate the reward    
    reward = env.calculate_reward 

    # take another snapshot at the end of the step
    new_state = env.snapshot 

    return new_state, reward, {}









    @property
    def sim_step(self):#Return current simulation second on SUMO
        return traci.simulation.getCurrentTime()/1000  # milliseconds to seconds

    def step(self, action):
        '''
        step() function is to:
        1. execute the action generated by policy network
        2. compute the reward of taking such action over one MPV step
        3. compute the new state (observation)
        '''
        ################## apply the actions generated by the policy network ################3
        self._apply_actions(action)
        current_den = traci.vehicle.getIDCount()/self.netLength/self.jam_density
        # print('New MDP step, apply actions, current density: %s' %round(current_den,2))

        throu_F5 = [[],[],[],[]]
        for _ in range(self.green_time+self.yellow_time):
            self._sumo_step()
            # here in each simulation step, don't forget to reroute all vehicles and make sure them don't leave the network
            for edge in self.EdgeList:
              target_edge_list= child(edge,self.net) # this is target edge list
              next_edge = random.choice(target_edge_list)
              next_next_edge = random.choice(child(next_edge,self.net))
              lane_id = edge+'_0'
              if traci.lane.getLastStepVehicleNumber(lane_id) >=1: # if there exist vehicles on the lane
                  traci.vehicle.setRoute(traci.lane.getLastStepVehicleIDs(lane_id)[-1],[edge, next_edge, next_next_edge])
            ########### compute the throughput at intersection F5, outgoing loop inductinos are 161~614 ###################### 
            throu_F5[0].append(traci.inductionloop.getLastStepVehicleIDs(str(161)))
            throu_F5[1].append(traci.inductionloop.getLastStepVehicleIDs(str(162)))
            throu_F5[2].append(traci.inductionloop.getLastStepVehicleIDs(str(163)))
            throu_F5[3].append(traci.inductionloop.getLastStepVehicleIDs(str(164)))
        #################### after one phase, count unique vehicles passing the loop inductions #######################
        throu_F5[0]= [ID_tuple for ID_tuple in throu_F5[0] if ID_tuple] # remove vacant tuple ()
        throu_F5[1]= [ID_tuple for ID_tuple in throu_F5[1] if ID_tuple]
        throu_F5[2]= [ID_tuple for ID_tuple in throu_F5[2] if ID_tuple]
        throu_F5[3]= [ID_tuple for ID_tuple in throu_F5[3] if ID_tuple]
        # print('the lane corresponding to the inductionloop is', traci.inductionloop.getLaneID(str(161)))
        sum_F5 = 0 
        sum_F5 += len(throu_F5[0]) + len(throu_F5[1]) + len(throu_F5[2]) + len(throu_F5[3])
        # print('the # of passing vehicles on F5E5_0 is:',len(set(throu_F5[0])))
        # print('the # of passing vehicles on F5F4_0 is:',len(set(throu_F5[1])))
        # print('the # of passing vehicles on F5E6_0 is:',len(set(throu_F5[2])))
        # print('the # of passing vehicles on F5E4_0 is:',len(set(throu_F5[3])))

        F5_passing_rate = int(sum_F5 / 4 /(self.yellow_time + self.green_time) *3600) # intersection throughput of F5
        print('throughput of F5 in last iteration is:', F5_passing_rate)
        ###################### observe new state and reward ################################
        new_observation = self._compute_observations()
        reward = F5_passing_rate # change the reward to be the reward of 
        return new_observation, reward

    def _apply_actions(self, action): #66 we need to know the data type 
        """
        action given by the policy network would be an action dict with key values of ts id
        we need to parse integer values to real actions of the ts 
        """
        # print('print action F5 is:',action['F5'])
        # print('when', action['F5']==0, 'give NS green phase')
        # print('when', action['F5']==1, 'give EW green phase')

        for ts in action: 
            # print('traffic light', ts, 'has the action', action[ts])
            if action[ts] == 0:
                # print('choose 0, give NS green phase')
                traci.trafficlight.setPhase(ts,1) # not sure the number corresponding to yellow phase
                # print('according to the policy agent, set ts %s to phase %d'%(ts,1))     
            if action[ts] == 1:
                # print('choose 1, give EW green phase')
                traci.trafficlight.setPhase(ts,3)
                # print('according to the policy agent, set ts %s to phase %d'%(ts,3))     
    

    def _compute_observations(self):
        observations = {} # using a dictionary to save ts id and its local observation 
        for ts in self.ts_ids:
            incoming_lanes = traci.trafficlight.getControlledLanes(ts)
            seen = set()
            incoming_lanes = [x for x in incoming_lanes if not (x in seen or seen.add(x))] # this get the clockwise 4 incoming links
            outgoing_lanes = [opposite_lane(x) for x in incoming_lanes] # clockwise 4 outgoing lanes
            cont_lanes = incoming_lanes + outgoing_lanes
            # print('8 controlled lanes of traffic light ',ts, 'is', cont_lanes)           
            Q_den = [min(1, traci.lane.getLastStepHaltingNumber(laneID)/traci.lane.getLength(laneID)*1000/self.jam_density) for laneID in cont_lanes]
            lane_den = [min(1, traci.lane.getLastStepVehicleNumber(laneID)/traci.lane.getLength(laneID)*1000/self.jam_density) for laneID in cont_lanes]
            observations[ts] = Q_den + lane_den
            # print('observation of traffic light %s is %s' %(ts,observations[ts]))
        return observations # observations is a dictionary

    def _sumo_step(self):
        traci.simulationStep()

    def close(self):
        traci.close()

    def space_availabe(self,edge):
        net = sumolib.net.readNet(self._net) # parse the netfile 
        length = net.getEdge(edge).getLength()
        # just compute the vehicle number on a lane
        max_num = math.floor(length/(self._vehicle_length+self._min_gap))
        veh_num = traci.lane.getLastStepVehicleNumber(edge+'_0')
        return (veh_num < max_num)



    # def _compute_observations(self):
    #     """
    #     Observation needs to be constructed as a sequence ordered by traffic lights
    #     its element needs to be a numpy array because later on it will be converted to a tensor
    #     and fed to the policy network as input  
    #     """
    #     # print('compute new observation for the next MP step')
    #     observations = {} # using a dictionary to save ts id and its local observation 
    #     # check_obs_seq = {}
    #     for ts in self.ts_ids:
    #         incoming_lanes = traci.trafficlight.getControlledLanes(ts)
    #         seen = set()
    #         incoming_lanes = [x for x in incoming_lanes if not (x in seen or seen.add(x))]
    #         outgoing_lanes = [opposite_lane(x) for x in incoming_lanes]
    #         # print('the incoming lanes are:', incoming_lanes)
    #         # print('the outgoing_lanes are:', outgoing_lanes)
    #         cont_lanes = incoming_lanes + outgoing_lanes
    #         # print('controlled lanes of a traffic light is ',cont_lanes)

    #         #todo, we need to check, print the observations to see whether it is correct
    #         # this get the clockwise 4 incoming links
    #         # we need a function to get the opposite lane 
    #         ## 66, we choose the queue length as the observation 
    #         Q_den = [min(1, traci.lane.getLastStepHaltingNumber(laneID)/traci.lane.getLength(laneID)*1000/self.jam_density) for laneID in cont_lanes]
    #         lane_den = [min(1, traci.lane.getLastStepVehicleNumber(laneID)/traci.lane.getLength(laneID)*1000/self.jam_density) for laneID in cont_lanes]
    #         observations[ts] = Q_den + lane_den
    #         # print('observation of traffic light %s is %s' %(ts,observations[ts]))
    #         # obsevation is queue_density and true vehicle density 
    #         # observations[ts] = [traci.lane.getLastStepHaltingNumber(laneID)/ for laneID in cont_lanes]
    #         # print('the observation at traffic signal %s is'%ts, type(observations[ts]))
    #         # check_obs_seq[ts] = [laneID for laneID in cont_lanes]
    #         # print('the sequence of observation oriented at ts %s is'%ts, check_obs_seq[ts])
    #         # print('observation list is ',observation)
    #         # note that occupancy is a value from 0~100 percentage
    #     return observations # observations is a dictionary
