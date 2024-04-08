#!/usr/bin/env python3

import os
import sys
import threading
import time
import numpy as np

RunObjects = []
with open('./object_name.txt', 'r', encoding='utf-8') as f:
	for line in f.readlines():
		line = line.strip()
		RunObjects.append(line)
# print(RunObjects)

gen_last_iter_only = True

if __name__ == '__main__':

	Finals_path = sys.argv[1]

	Methods = []

	Methods.append('OSVP')
	# Methods.append('PCNBV')
	# Methods.append('MA-SCVP_NBV')
	# Methods.append('OSVP_NBV')
	# Methods.append('MCMF')
	# Methods.append('RSE')
	# Methods.append('SCVP')
	# Methods.append('GMC')

	for method in Methods:
		method_path = os.path.join(Finals_path, method)
		Objects = os.listdir(method_path)
		for obj in Objects:
			method_obj_path = os.path.join(Finals_path, method, obj)
			Iterations = os.listdir(method_obj_path)

			last_iter = np.max(np.asarray(Iterations).astype(int))
			if gen_last_iter_only:
				Iterations = [str(last_iter)]

			for ite in Iterations:
				method_obj_ite_path = os.path.join(Finals_path, method, obj, ite)
				obj_name = obj.rsplit("_", maxsplit=2)[0]

				cmd = ''
				cmd += ' python ' + './POCO/mesh_gen_iterative_bycase.py'
				cmd += ' --case_path ' + method_obj_ite_path
				cmd += ' --obj_name ' + obj_name
				cmd += ' --config_default ' + './POCO/configs/config_default.yaml'
				cmd += ' --config ' + './POCO/configs/config_cosc.yaml'
				# print(cmd)
				os.system(cmd)
