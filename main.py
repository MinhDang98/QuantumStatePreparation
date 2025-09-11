import numpy as np
import matplotlib.pyplot as plt
import libs.utils as utils
from libs.target_state import TargetState, TargetStateName, GeneralTargetState
from libs.agent import QuantumnAgent

GENERAL_TARGET_STATE_LIST = [
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=2),
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=3),
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=4),
	GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=5),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=2),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=3),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=4),
	GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=5)
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

	total_timesteps = 10_000_000
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

def test(using_general_target_states: bool = False):
	target_states_list = []
	if using_general_target_states:
		target_states_list = GENERAL_TARGET_STATE_LIST
	else:
		target_states_list = TARGET_STATE_LIST
  
	agent = QuantumnAgent(training_mode=False)
	for i in target_states_list:
		agent.build_circuit("TargetState", i)
 

if __name__ == "__main__":
	utils.clean_log()
	train(using_general_target_states=True, folder_name="GeneralTargetStates")
	# test()
	
