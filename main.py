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

	total_timesteps = 600000
	eval_frequency = total_timesteps // 10
	eval_episode = 50
	agent = QuantumnAgent(general_target_state_list, 
						total_timesteps=total_timesteps,
						eval_frequency=eval_frequency,
						eval_episode=eval_episode,
						training_mode=True, 
						verbose=1,
						is_curriculum=True,
						use_alp=True)
 
if __name__ == "__main__":
	utils.clean_log()
	train()
	
