import json
import os
import numpy as np
import torch
from typing import Union
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from libs.alp import ALPBandTeacher, TeacherEnvWrapper
from .quantum_state_preparation import QuantumStatePreparation
from .target_state import TargetState, GeneralTargetState


DIR_NAME = "QSP_RL/"
MODEL_NAME = "best_model.zip"

class QuantumnAgent():
	"""
	This class manages the training and evaluation of a quantum state preparation agent using reinforcement learning.
	It supports both standard training and curriculum learning approaches, including an ALP-based curriculum.

	Attributes:
		log_dir (str): Directory to store logs.
		model_dir (str): Directory to store trained models.
		config_path (str): Path to the environment configuration file.
		target_states_list (list): A list of target quantum states for training.
		total_timesteps (int): The total number of timesteps for training.
		eval_frequency (float): The frequency of evaluation.
		eval_episode (float): The number of episodes for evaluation.
		verbose (int): The verbosity level for printing information.
		device (str): The computing device (cpu or cuda).
		max_env_qubits (int): The maximum number of qubits in the environment.
		max_env_gates (int): The maximum number of gates in the environment.
		env (gym.Env): The training environment.
		callbacks (list): A list of callbacks for training.
		model (PPO): The PPO agent model.
	"""
	def __init__(self,
				 model_folder_name: str = None,
				 target_states_list: list[Union[TargetState, GeneralTargetState]] = None,
				 total_timesteps: int = None,
				 eval_frequency: float = None,
				 eval_episode: float = None,
				 training_mode: bool = False,
	 			 verbose: int = 0,
	  			 use_alp: bool = False):
		"""
		Initializes the QuantumAgent.

		Args:
			model_folder_name (str): The folder where the model is stored.
			target_states_list (list, optional): List of quantum states. Defaults to None.
			total_timesteps (int, optional): Total training timesteps. Defaults to None.
			eval_frequency (float, optional): Evaluation frequency. Defaults to None.
			eval_episode (float, optional): Number of episodes for evaluation. Defaults to None.
			training_mode (bool, optional): If true, initializes in training mode. Defaults to False.
			verbose (int, optional): Verbosity level. Defaults to 0.
			is_curriculum (bool, optional): If true, uses curriculum learning. Defaults to False.
			use_alp (bool, optional): If true, uses ALP for curriculum learning. Defaults to False.
		"""
		self.log_dir = "./logs/" + DIR_NAME
		os.makedirs(self.log_dir, exist_ok=True)

		self.model_dir = "./model/" + DIR_NAME
		os.makedirs(self.model_dir, exist_ok=True)

		self.config_path = self.model_dir + "env_config.json"

		if model_folder_name is None:
			raise ValueError("Missing Folder Name for the model")
		self.model_folder_name = model_folder_name
		
		if training_mode:
			print("Agent initialize in traning mode.")

			if target_states_list is None:
				raise Exception("Missing Target State for the model")
			self.target_states_list = target_states_list

			# Set up basic properties
			self.total_timesteps = total_timesteps
			self.eval_frequency = eval_frequency
			self.eval_episode = eval_episode
			self.verbose = verbose

			# Determine the device to use
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
			print(f"Using device: {self.device}")

			self.save_env_config(self.config_path)

			if use_alp:
				self.alp_train_curriculum()
			else:
				self.train_model()

			self.env.close()
		else:
			print("Agent initialize in testing mode.")
			config = self.load_env_config(self.config_path)
			self.max_env_qubits = config.get('max_env_qubits')
			self.max_env_gates = config.get('max_env_gates')

	def save_env_config(self, path):
		"""
		Saves the environment configuration to a JSON file.
		
		Args:
			path (str): The path to save the configuration file.
		"""
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
		"""
		Loads environment configuration from a JSON file.

		Args:
			path (str): The path to the configuration file.

		Returns:
			dict: The loaded configuration data.
		"""
		with open(path, 'r') as f:
			config_data = json.load(f)
		print(f"Loaded environment config from {path}")
		return config_data

	def initialize_environment(self, target_states_list=None):
		"""
		Initializes the training environment.

		Args:
			target_states_list (list, optional): List of target states. Defaults to None.
		"""
		if target_states_list is None:
			target_states_list = self.target_states_list

		def mask_fn(env):
			return env.compute_valid_action_mask()

		raw_env = QuantumStatePreparation(
			target_states_list=target_states_list,
			max_env_qubits=self.max_env_qubits,
			max_env_gates=self.max_env_gates
		)
		self.env = Monitor(ActionMasker(raw_env, mask_fn))
		self.eval_env = Monitor(ActionMasker(raw_env, mask_fn))
  
	def initialize_environment_with_teacher(self,
											n_bins: int = 5,
											window_size: int = 100,
											replay_prob: float = 0.15):
		"""
		Create an ALPBandTeacher and set self.env to a (vectorizable) TeacherEnvWrapper instance.
		Usage:
			self.target_states_list must already be set.
			Call this in place of initialize_environment(...) when you want teacher-guided sampling.
		
		Args:
			n_bins (int): The number of bins for the ALP teacher.
			window_size (int): The window size for the ALP teacher.
			replay_prob (float): The probability of replaying a state.
		"""
		# create teacher
		self.teacher = ALPBandTeacher(
			target_states_list=self.target_states_list,
			n_bins=n_bins,
			window_size=window_size,
			replay_prob=replay_prob
		)
  
		def make_wrapper_env():
			def mask_fn(env):
				return env.action_masks()
			return ActionMasker(
				TeacherEnvWrapper(
					target_states_list=self.target_states_list,
					teacher=self.teacher,
					max_env_qubits=self.max_env_qubits,
					max_env_gates=self.max_env_gates
				),
				mask_fn
			)
   
		self.env = make_vec_env(make_wrapper_env, n_envs=1)
		self.eval_env = make_vec_env(make_wrapper_env, n_envs=1)
  
	def set_up_callbacks(self):
		"""
		Sets up the standard evaluation and stop training callbacks.
		"""
		stop_train_callback = StopTrainingOnNoModelImprovement(
			max_no_improvement_evals=5,
			min_evals=5,
			verbose=1
		)

		self.callbacks = EvalCallback(
			self.eval_env,
			log_path=self.log_dir,
			eval_freq=self.eval_frequency,
			n_eval_episodes=self.eval_episode,
			deterministic=True,
			render=False,
			callback_after_eval=stop_train_callback,
			best_model_save_path=self.model_dir + self.model_folder_name,
		)

	def set_up_curriculum_callback(self):
		"""
		Sets up the custom curriculum evaluation callback.
		"""
		stop_train_callback = StopTrainingOnNoModelImprovement(
			max_no_improvement_evals=5,
			min_evals=10,
			verbose=1
		)
  
		self.curriculum_callback = EvalCallback(
			self.eval_env,
			callback_on_new_best=stop_train_callback,
			eval_freq=self.eval_frequency,
			n_eval_episodes=self.eval_episode,
			log_path=self.log_dir,
			best_model_save_path=self.model_dir + self.model_folder_name,
			verbose=1,
		)

	def set_up_model(self):
		"""
		Initializes the PPO model with a specified policy network architecture and hyperparameters.
		"""
		self.policy_kwargs = dict(
			net_arch=dict(pi=[256,256], vf=[256,256])
		)

		self.model = MaskablePPO(
			"MultiInputPolicy",
			self.env,
			learning_rate=1e-3,
			n_steps=4096,
			batch_size=64,
			n_epochs=10,
			gamma=0.995,
			gae_lambda=0.95,
			clip_range=0.2,
			ent_coef=0.01,
			policy_kwargs=self.policy_kwargs,
			verbose=0,
			tensorboard_log=self.log_dir,
			device=self.device
		)

	def train_model(self):
		"""
		Trains the PPO model in standard mode.
		"""
		self.initialize_environment()

		self.set_up_callbacks()

		self.set_up_model()

		print(f"Starting PPO training for {self.total_timesteps} timesteps...")
		self.model.learn(
			total_timesteps=self.total_timesteps,
			callback=self.callbacks,
			progress_bar=True,
			tb_log_name="PPO"
		)
		print("Training finished.")

	def close_env(self):
		"""
		Safely closes the environment to prevent resource leaks.
		"""
		try:
			if hasattr(self, "env") and self.env is not None:
				try:
					self.env.close()
				except Exception:
					pass
		except Exception:
			pass

	def alp_train_curriculum(self, n_bins=5, window_size=150, replay_prob=0.2):
		"""
		Trains the model using an ALP-based curriculum.

		Args:
			n_bins (int): The number of bins for the ALP teacher.
			window_size (int): The window size for the ALP teacher.
			replay_prob (float): The probability of replaying a state.
		"""
		print("Starting curriculum training with ALP...")
   
		self.initialize_environment_with_teacher(n_bins=n_bins, window_size=window_size, replay_prob=replay_prob)

		# First stage: create model from scratch
		self.set_up_model()

		# Bind callback to the current env
		self.set_up_curriculum_callback()

		# learn for this curriculum stage
		print(f"Starting training for {self.total_timesteps} timesteps...")
		self.model.learn(
			total_timesteps=self.total_timesteps,
			callback=self.curriculum_callback,
			progress_bar=True,
			tb_log_name=f"PPO_ALP"
		)

	def build_circuit(self, 
				   folder_name: str,
				   target_state: Union[TargetState, GeneralTargetState]):
		"""
		Builds a quantum circuit for a given target state using the trained model.

		Args:
			folder_name (str): The folder where the model is stored.
			target_state (Union[TargetState, GeneralTargetState]): The target quantum state.

		Returns:
			The quantum circuit if successful, otherwise None.
		"""
		best_model_path = os.path.join(self.model_dir + folder_name, MODEL_NAME)
		if not os.path.exists(best_model_path):
			print(f"[Error] No best model found at {best_model_path}")
			return None

		# Direct env (no VecEnv) for inference
		eval_env = QuantumStatePreparation(
			target_states_list=[target_state],
			max_env_qubits=self.max_env_qubits,
			max_env_gates=self.max_env_gates
		)

		print(f"[Info] Loading trained model from {best_model_path}")
		best_model = MaskablePPO.load(best_model_path)

		obs, info = eval_env.reset()
		steps, done = 0, False

		while not done and steps < target_state.max_gates:
			mask = eval_env.compute_valid_action_mask()
			action, _ = best_model.predict(obs, deterministic=True, action_masks=mask)
			obs, reward, terminated, truncated, info_eval = eval_env.step(action)
			done = terminated or truncated
			steps += 1

		fid = info_eval.get("fidelity", 0.0)
		print(f"[End] {target_state.state_name} circuit finished in {steps} steps with fidelity {fid:.4f}")
		print(eval_env.qc.draw())   # <-- Now guaranteed correct circuit
		return
