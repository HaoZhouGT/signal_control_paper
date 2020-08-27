import math
import sumolib

from xml.etree import ElementTree as et    # use ElementTree module to edit xml
import numpy as np

import matplotlib.pyplot as plt


def opposite_lane(laneID): #'find the opposite lane on the same edge by reverting the lane name'
    reversed_lane = laneID[2]+laneID[3]+laneID[0]+laneID[1]+laneID[4]+laneID[5]
    return reversed_lane


def read_detector_results(dir,green,maximum_flow=2400):
    tree = et.parse(dir)    # open file and parse xml content
    interval_list = tree.findall(".//interval")
    begin_list = [float(i.get('begin')) for i in interval_list if i.get('id')=="0"]
    # print(begin_list)
    max_time = max(begin_list)
    # print('the biggest time is ',max_time)
    N = int(int(max_time)/green)
    # print('N is ',N)


    loop_list = [i.get('id') for i in interval_list]
    detector_num = len(list(set(loop_list)))
    # print('detector num is ', detector_num)
    data = [[] for i in range(N)]

    for n in range(N):
        for i in interval_list:
            if i.get('begin')==(str(n*green)+'.00') and i.get('end')==(str((n+1)*green)+'.00'):
                data[n].append(float(i.get('flow')))
    data = [np.sum(d)/detector_num for d in data]
    normalized_data = [d/maximum_flow for d in data]

    plt.figure()
    plt.plot(data)
    plt.title('flow output by loop detectors')
    plt.xlabel('interval #')
    plt.ylabel('flow (veh/hr)')
    plt.savefig('loopinduction.png')

    plt.figure()
    plt.plot(normalized_data)
    plt.title('normalized flow output by loop detectors')
    plt.xlabel('interval #')
    plt.ylabel('normalized_flow')
    plt.savefig('normalized_flow.png')
    return data, normalized_data



def change(current_phase):
    return (current_phase+2)%4


def child(myEdgeID,net):
    nextEdges = net.getEdge(myEdgeID).getOutgoing()
    nextEdges = list(nextEdges)
    nextEdge_list = [nextEdges[i].getID() for i in range(len(nextEdges))]
    nextEdge_list = [str(r) for r in nextEdge_list]
    return nextEdge_list

def netLength(net):
    sum = 0
    for i in range(len(net.getEdges())):
        sum += net.getEdges()[i].getLength()
    return sum/1000.0 # unit is in meters


def wave_speed(vehicle_length, min_gap, tau):
    return (vehicle_length+min_gap)/tau*3.6


def simulation_go(N):
    for i in range(N):
        traci.simulationStep()


def jam_density(vehicle_length, min_gap):
    return int(1000.0/(vehicle_length+min_gap))


def mfd_green_time(net, ratio = 1.0, u = 60, w = 20):
    # compute the average edge length
    average_length = int(netLength(net)*1000/len(net.getEdges()))
    # print('the average edge length is ', average_length)
    u = u / 3.6
    w = w / 3.6
    g = average_length/ratio*(1.0/w+1.0/u)
    return int(g)

def mfd_critical_length(green, u = 60, w = 20):
    return green * (1.0/w+1.0/u)


def set_loop_detector_time(net,green):
    green = int(green)
    line1 = "<?xml version='1.0' encoding='UTF-8'?>"
    line2 = '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">'
    line3 = '</additional>'
    config_file = open('./grid.add.xml','w+')
    print('write sumo add file, set loop detectors on each lane', 'with frequenct equal to green time', green)
    line= line1+'\n'+line2
    edge_list = net.getEdges()
    num_detector = 0
    for i in range(len(edge_list)): # get edge, then use the edge method, getLanes
        for lane in edge_list[i].getLanes(): # note that a variable is a class object, not its name
            line = line +'\n' +'<inductionLoop id="%d" lane="%s" pos="%d" freq="%d" file="loopout.xml" friendlyPos="true"/>' %(i,lane.getID(),5,green)
            num_detector = num_detector+1
    line = line + '\n'+line3
    config_file.writelines(line)
    return num_detector
