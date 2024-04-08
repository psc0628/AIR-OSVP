import sys
import os
import time
import numpy as np

name_of_objects = []
with open('object_name.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        name_of_objects.append(line)
        
rotate_ids = []
rotate_ids.append(3)

first_view_ids = []
first_view_ids.append(27)

only_observation_points = False

train_epoch = 10

for object_name in name_of_objects:
    print('poco rendering ' + object_name)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            while os.path.isfile('./data/'+object_name+'_r'+str(rotate_id)+'_v'+str(view_id)+'_vs.txt') == False:
                time.sleep(0.1)
            if only_observation_points == True:
                # ablation study on input with only observation points
                fin = open('./data/' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + '_surface.txt', 'r')
                lines = fin.read()
                fin.close()
                fout = open('./data/' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + '_pc.txt', "w")
                fout.write(lines)
                fout.close()
            else:
                # Refine input with POCO
                os.system('python ./POCO/train_and_infer_one.py --config ./POCO/configs/config_cosc.yaml --epoch ' + str(train_epoch) + ' --obj_name ' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id))

            f = open('./data/poco_ready.txt', 'a')
            f.close()
            print('poco rendering ' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + ' over.')
print('all over.')
