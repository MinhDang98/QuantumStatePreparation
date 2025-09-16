import numpy as np
import matplotlib.pyplot as plt
import libs.utils as utils
from libs.target_state import TargetState, TargetStateName, GeneralTargetState
from libs.agent import QuantumnAgent
from libs.benchmark import Benchmark
from typing import Union


GENERAL_TARGET_STATE_LIST = [
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=2),
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=3),
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=4),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=2),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=3),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=4),
	TargetState(target_state_name=TargetStateName.BELL_STATE_PSI),				# This state cannot be scaled
	TargetState(target_state_name=TargetStateName.COMPUTATIONAL_BASIS_STATE),	# This state cannot be scaled
]

TARGET_STATE_LIST = [
	TargetState(target_state_name=TargetStateName.GHZ_STATE),
	TargetState(target_state_name=TargetStateName.BELL_STATE),
	TargetState(target_state_name=TargetStateName.BELL_STATE_PSI),
	TargetState(target_state_name=TargetStateName.COMPUTATIONAL_BASIS_STATE),
	TargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION)
]

def train(using_general_target_states: bool = False,
		  folder_name: str = None):
	if (folder_name is None) or (folder_name.strip() == ""):
		raise ValueError("Folder name must be provided for training mode.")
	
	target_states_list = []
	if using_general_target_states:
		target_states_list = GENERAL_TARGET_STATE_LIST
	else:
		target_states_list = TARGET_STATE_LIST

	total_timesteps = 3000000
	eval_frequency = total_timesteps // 20
	eval_episode = 50
	QuantumnAgent(model_folder_name=folder_name,
	 			target_states_list=target_states_list, 
				total_timesteps=total_timesteps,
				eval_frequency=eval_frequency,
				eval_episode=eval_episode,
				training_mode=True, 
				verbose=1,
				use_alp=True)

def test(test_states: list[Union[GeneralTargetState, TargetState]],
		 folder_name: str = None):
	if test_states is None:
		if (folder_name is None) or (folder_name.strip() == ""):
			raise ValueError("Either test states or folder name must be provided for testing mode.")
  
	agent = QuantumnAgent(model_folder_name=folder_name, training_mode=False)
	for i in test_states:
		agent.build_circuit(i)

def benchmark(folder_name: list[str] = None):
	if (folder_name is None):
		raise ValueError("Folder name must be provided for benchmarking mode.")
	
	agent = []
	for fn in folder_name:
		agent.append(QuantumnAgent(model_folder_name=fn, training_mode=False))
	benchmark = Benchmark(agent)

	# Add suites
	benchmark.add_suite("fixed", [
		TargetState(TargetStateName.BELL_STATE_PSI),
		TargetState(TargetStateName.COMPUTATIONAL_BASIS_STATE),
	])

	benchmark.add_suite("scalable", [
		GeneralTargetState(TargetStateName.GHZ_STATE, num_qubits=3),
		GeneralTargetState(TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=3),
	])

	# Run benchmarks
	results = benchmark.run_all(n_eval_episodes=100)
	print(results)

if __name__ == "__main__":
	utils.clean_log()

	# Example usage for Training
	genral_folder_name = "General_Target_States"
	fixed_folder_name = "Fixed_Target_States"
	# train(using_general_target_states=True, folder_name=genral_folder_name)
	
	# Example usage for Testing
	# test(test_states=TARGET_STATE_LIST, folder_name=genral_folder_name)

	# Example usage for Benchmarking
	benchmark(folder_name=[genral_folder_name, fixed_folder_name])
