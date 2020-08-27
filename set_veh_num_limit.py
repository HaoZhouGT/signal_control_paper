
from side_functions import *


def gene_config(net, target_density,vehicle_length=4.5,min_gap=2.0):
    line1 = "<?xml version='1.0' encoding='UTF-8'?>"
    line2 = '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">'
    line3 = "  <input>"
    line4 = '    <net-file value="grid.net.xml"/>'
    line5 = '    <route-files value="grid.rou.xml"/>'
    # line6 = '    <additional-files value="grid.add.xml"/>'
    line7 = '  </input>'
    line8 = '  <time>'
    line9 = '    <begin value="0"/>'
    line10 = '  </time>'
    line11 = '  <Processing>'
    line12 = '    <time-to-teleport value="120"/>' # note that vehicle waiting more than 120s will be teleported
    line14 = '  </Processing>'
    line15 = '</configuration>'
    # first_half = [line1,line2,line3,line4,line5,line6, line7,line8,line9,line10,line11,line12]
    ## note that here I removed the grid.add.xml file, which means I didn't use the loop detectors
    first_half = [line1,line2,line3,line4,line5,line7,line8,line9,line10,line11,line12]
    second_half= [line14,line15]
    config_file = open('./grid.sumo.cfg','w+')
    target_max_vehicle = netLength(net)*jam_density(vehicle_length, min_gap)*target_density
    print('write sumo config file, with max vehicle num is', str(int(target_max_vehicle)))
    line13 = '    <max-num-vehicles value="' + str(int(target_max_vehicle))+ '"/>'
    contents = first_half+ [line13] + second_half
    config_file.writelines(contents)
