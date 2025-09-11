import numpy as np
import matplotlib.pyplot as plt
import libs.utils as utils
from libs.target_state import TargetState, TargetStateName, GeneralTargetState
from libs.agent import QuantumnAgent

def train():
	general_target_state_list = [
		GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=2),
  		GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=3),
		GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=4),
		GeneralTargetState(target_state_name=TargetStateName.GHZ_STATE, num_qubits=5),
		GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=2),
		GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=3),
		GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=4),
		GeneralTargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION, num_qubits=5)
	]
 
	target_states_list = [
		TargetState(target_state_name=TargetStateName.GHZ_STATE),
		TargetState(target_state_name=TargetStateName.BELL_STATE),
		TargetState(target_state_name=TargetStateName.BELL_STATE_PSI),
		TargetState(target_state_name=TargetStateName.COMPUTATIONAL_BASIS_STATE),
		TargetState(target_state_name=TargetStateName.UNIFORM_SUPERPOSITION)
	]

	total_timesteps = 10_000_000
	eval_frequency = total_timesteps // 20
	eval_episode = 50
	QuantumnAgent(target_states_list, 
				total_timesteps=total_timesteps,
				eval_frequency=eval_frequency,
				eval_episode=eval_episode,
				training_mode=True, 
				verbose=1,
				use_alp=True)
 
if __name__ == "__main__":
	utils.clean_log()
	train()
	
