import os
import sys
import random

def usage():
	print("Usage: python train_test_splitter.py <total_num_networks> <num_trainers> <seed> <trainer_file_name> <tester_file_name>")

if __name__ == "__main__":
	total_num_networks = int(sys.argv[1])
	num_trainers = int(sys.argv[2])
	seed = int(sys.argv[3])
	trainer_file_name = sys.argv[4]
	tester_file_name = sys.argv[5]

	random.seed(seed)

	trainers = random.sample(range(total_num_networks), num_trainers)
	testers = list(set(range(total_num_networks)) - set(trainers))
	
	trainer_file = open(trainer_file_name, "w")
	tester_file = open(tester_file_name, "w")
	for trainer in trainers:
		trainer_file.write("%d\n"%(trainer))

	trainer_file.close()

	for tester in testers:
		tester_file.write("%d\n"%(tester))

	tester_file.close()
