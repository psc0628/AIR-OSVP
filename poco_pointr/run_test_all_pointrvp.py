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

for object_name in name_of_objects:
    print('testing ' + object_name)
    for rotate_id in rotate_ids:
        for view_id in first_view_ids:
            while os.path.isfile('./data/poco_ready.txt') == False:
                time.sleep(0.1)
            os.system('python ./pointrVP/infer_once.py --infer_file ./data/' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + '_pc.txt')
            os.remove('./data/poco_ready.txt')
            f = open('./data/ready.txt', 'a')
            f.close()
            print('testing ' + object_name + '_r' + str(rotate_id) + '_v' + str(view_id) + ' over.')
print('all over.')
