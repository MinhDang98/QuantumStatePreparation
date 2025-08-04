import json
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from .quantum_state_preparation import QuantumStatePreparation, TargetState

DIR_NAME = "QSP_RL/"
MODEL_NAME = "DQN_RL"

class QuantumnAgent():
	def __init__(self, 
				 target_states_list: list[TargetState] = None, 
				 total_timesteps: int = None, 
				 eval_frequency: float = None, 
				 eval_episode: float = None,
				 training_mode: bool = False,
				 verbose: int = 0):
					
		self.log_dir = "./logs/" + DIR_NAME
		os.makedirs(self.log_dir, exist_ok=True)
		
		self.model_dir = "./model/" + DIR_NAME
		os.makedirs(self.model_dir, exist_ok=True)
  
		self.config_path = self.model_dir + "env_config.json"
  
		if training_mode:
			print("Agent initialize in traning mode.")

			if target_states_list is None:
				raise ExceptionType("Missing Target State for the model")
			self.target_states_list = target_states_list
   
			# Set up basic properties
			self.total_timesteps = total_timesteps
			self.eval_frequency = eval_frequency
			self.eval_episode = eval_episode
			self.verbose = verbose

			# Determine the device to use
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			print(f"Using device: {self.device}")
   
			# Save env config
			self.save_env_config(self.config_path)
      
			# Set up environment
			self.initialize_environment()
			
			# Prepare the callback during training
			self.set_up_callbacks()
			
			# Set up model
			self.set_up_model()
			
			# Evaluate after training
			self.train_model()

			self.env.close()
			self.evaluate_env.close()
		else:
			print("Agent initialize in testing mode.")
			config = self.load_env_config(self.config_path)
			self.max_env_qubits = config.get('max_env_qubits')
			self.max_env_gates = config.get('max_env_gates')
	
	def save_env_config(self, path):
		self.max_env_qubits = max(ts.num_qubits for ts in self.target_states_list)
		self.max_env_gates = max(ts.max_gates for ts in self.target_states_list)
  
		config_data = {
			'max_env_qubits': self.max_env_qubits,
			'max_env_gates': self.max_env_gates # Saving this too for consistency
		}
		with open(path, 'w') as f:
			json.dump(config_data, f)
		print(f"Saved environment config to {path}")
  
	def load_env_config(self, path):
		"""Loads environment configuration from a JSON file."""
		with open(path, 'r') as f:
			config_data = json.load(f)
		print(f"Loaded environment config from {path}")
		return config_data
	
	def initialize_environment(self, target_states_list: list[TargetState] = None):
		# Make_vec_env handles creation and wrapping for stable-baselines3 compatibility.
		# env_kwargs passes arguments to your QuantumStatePreparation.__init__
		self.env = make_vec_env(QuantumStatePreparation,
						n_envs=1,
						env_kwargs={"target_states_list": self.target_states_list,
									"max_env_qubits": self.max_env_qubits,
									"max_env_gates": self.max_env_gates})
		
		self.evaluate_env = QuantumStatePreparation(self.target_states_list, self.max_env_qubits, self.max_env_gates)
   
	def initialize_environment_for_infererence(self, target_state: TargetState):
		# For inference, we just need 1 state
		self.env = make_vec_env(QuantumStatePreparation,
			n_envs=1,
			env_kwargs={"target_states_list": [target_state],
						"max_env_qubits": self.max_env_qubits,
						"max_env_gates": self.max_env_gates})
	
		self.evaluate_env = QuantumStatePreparation([target_state], self.max_env_qubits, self.max_env_gates)

	def set_up_callbacks(self):
		stop_train_callback = StopTrainingOnNoModelImprovement(
			max_no_improvement_evals=50,
			min_evals=5,
			verbose=1
		)
		
		eval_callback = EvalCallback(
			self.env,
			log_path=self.log_dir,
			eval_freq=self.eval_frequency,
			n_eval_episodes=self.eval_episode,
			deterministic=True,
			render=False,
			callback_after_eval=stop_train_callback
		)

		self.callbacks = [eval_callback]
	
	def set_up_model(self):
		self.model = DQN(
			"MlpPolicy",
			self.env,
			learning_rate=1e-3,
			buffer_size=1000000,
			learning_starts=100,
			batch_size=128,
			train_freq=(4, "step"),
			target_update_interval=100,
			exploration_fraction=0.3,
			exploration_final_eps=0.02,
			policy_kwargs=dict(net_arch=[256, 256]),
			verbose=self.verbose,
			tensorboard_log=self.log_dir,
			device=self.device
		)
		
	def train_model(self):
		print(f"Starting DQN training for {self.total_timesteps} timesteps...")
		self.model.learn(
			total_timesteps=self.total_timesteps,
			callback=self.callbacks,
			progress_bar=True
		)
		print("Training finished.")

		# Save the final model
		self.model.save(self.model_dir + MODEL_NAME + ".zip")
		print(f"Final model saved to {self.model_dir}")

	def build_circuit(self, 
					  target_state: TargetState):
		best_model_path = os.path.join(self.model_dir, MODEL_NAME + ".zip")
		
  
		# Set up environment
		self.initialize_environment_for_infererence(target_state)
   
		if os.path.exists(best_model_path):
			best_model = DQN.load(best_model_path, env=self.env)
			total_eval_episode = 50
			for i in range(total_eval_episode):
				obs, info = self.evaluate_env.reset(target_state_object=target_state)
				done = False
				steps = 0

				while not done and steps < self.evaluate_env.max_gates:
					action, state = best_model.predict(obs, deterministic=True)
					obs, reward, terminated, truncated, info_eval = self.evaluate_env.step(action)
					done = terminated or truncated
					steps += 1
					# We found the circuit
					if terminated:
						print(f"Circuit successfully built in {steps} steps with fidelity: {info_eval['fidelity']:.4f}")
						return self.evaluate_env.qc
					elif truncated:
						print(f"Circuit building truncated after {steps} steps (max gates reached). Fidelity: {info_eval['fidelity']:.4f}")
						return None
					
			print(f"Circuit building finished without reaching target fidelity within {self.evaluate_env.max_gates} steps. Fidelity: {info_eval['fidelity']:.4f}")
			return None
		else:
			print(f"No best model found at {best_model_path}")