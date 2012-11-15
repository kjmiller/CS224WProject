import os
import sys
import numpy.random
import math

seed = 0
numpy.random.seed(seed)

for i in range(90, 100):
	print(str(i))
	feature_dict = {}
	old_feature_file = open("../synthetic_data/synthetic_data.feature%d"%(i), "r")
	for line in old_feature_file:
		stuff = line.rstrip("\n").split(" ")
		feature_dict[(int(stuff[0]), int(stuff[1]))] = numpy.random.randn(2)
		feature_dict[(int(stuff[1]), int(stuff[0]))] = numpy.random.randn(2)
		
	old_feature_file.close()
	new_feature_file = open("../synthetic_data/synthetic_data.antisymmetric_feature%d"%(i), "w")
	for item in feature_dict:
		new_feature_file.write("%d %d %f %f\n"%(item[0], item[1], feature_dict[item][0], feature_dict[item][1]))

	new_feature_file.close()

	for sigma_squared in [0.0, 0.5, 1.0, 1.5, 2.0]:
		print(str(sigma_squared))
		noisy_feature_file = open("../synthetic_data/synthetic_data.antisymmetric_feature%d_noisy%f"%(i, sigma_squared), "w")
		for item in feature_dict:
			noisy_feature = feature_dict[item] + math.sqrt(sigma_squared) * numpy.random.randn(2)
			noisy_feature_file.write("%d %d %f %f\n"%(item[0], item[1], noisy_feature[0], noisy_feature[1]))

		noisy_feature_file.close()
