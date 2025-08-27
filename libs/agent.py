import json
import os
import numpy as np
import torch
from typing import Union
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from libs.alp import ALPBandTeacher, TeacherEnvWrapper
from .quantum_state_preparation import QuantumStatePreparation
from .target_state import TargetState, GeneralTargetState


DIR_NAME = "QSP_RL/"
MODEL_NAME = "best_model.zip"

from stable_baselines3.common.callbacks import BaseCallback

class CurriculumEvalCallback(BaseCallback):
	def __init__(self, eval_env, best_model_save_path, eval_freq, n_eval_episodes, verbose=1):
		super(CurriculumEvalCallback, self).__init__(verbose)
		self.eval_env = eval_env
		self.best_model_save_path = best_model_save_path
		self.eval_freq = eval_freq
		self.n_eval_episodes = n_eval_episodes
		self.best_mean_reward = -np.inf

	def _on_step(self) -> bool:
		if self.n_calls % self.eval_freq == 0:
			# Evaluate the agent on the evaluation environment using the function
			episode_rewards, _ = evaluate_policy(
				self.model,
				self.eval_env,
				n_eval_episodes=self.n_eval_episodes,
				deterministic=True,
				return_episode_rewards=True
			)
			mean_reward = np.mean(episode_rewards)

			if self.verbose > 0:
				print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {np.std(episode_rewards):.2f}")
			
			# Check if this is the new overall best model
			if mean_reward > self.best_mean_reward:
				if self.verbose > 0:
					print("New overall best mean reward! Saving model.")
				self.best_mean_reward = mean_reward
				self.model.save(self.best_model_save_path)
				return True
			
		return True
	
class QuantumnAgent():
	def __init__(self,
				 target_states_list: list[Union[TargetState, GeneralTargetState]] = None,
				 total_timesteps: int = None,
				 eval_frequency: float = None,
				 eval_episode: float = None,
				 training_mode: bool = False,
				 verbose: int = 0,
	 			 is_curriculum: bool = False,
	  			 use_alp: bool = False):

		self.log_dir = "./logs/" + DIR_NAME
		os.makedirs(self.log_dir, exist_ok=True)

		self.model_dir = "./model/" + DIR_NAME
		os.makedirs(self.model_dir, exist_ok=True)

		self.config_path = self.model_dir + "env_config.json"

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

			if is_curriculum:
				if use_alp:
					self.alp_train_curriculum()
				else:
					self.train_curriculum()
			else:
				self.train_model()

			self.env.close()
			self.env.close()
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

	def initialize_environment(self, target_states_list=None):
		if target_states_list is None:
			target_states_list = self.target_states_list

		# Make_vec_env handles creation and wrapping for stable-baselines3 compatibility.
		# env_kwargs passes arguments to your QuantumStatePreparation.__init__
		self.env = QuantumStatePreparation(target_states_list=target_states_list,
											max_env_qubits=self.max_env_qubits,
											max_env_gates=self.max_env_gates)

	def initialize_environment_with_teacher(self,
											n_bins: int = 5,
											window_size: int = 100,
											replay_prob: float = 0.15):
		"""
		Create an ALPBandTeacher and set self.env to a (vectorizable) TeacherEnvWrapper instance.
		Usage:
			self.target_states_list must already be set.
			Call this in place of initialize_environment(...) when you want teacher-guided sampling.
		"""
		# create teacher
		self.teacher = ALPBandTeacher(
			target_states_list=self.target_states_list,
			n_bins=n_bins,
			window_size=window_size,
			replay_prob=replay_prob
		)
  
		def make_wrapper_env():
			return TeacherEnvWrapper(
				target_states_list=self.target_states_list,
				teacher=self.teacher,
				max_env_qubits=self.max_env_qubits,
				max_env_gates=self.max_env_gates
			)

		# use make_vec_env with a callable factory (keeps SB3 speed & wrapping)
		self.env = make_vec_env(make_wrapper_env, n_envs=1)

	def set_up_callbacks(self):
		stop_train_callback = StopTrainingOnNoModelImprovement(
			max_no_improvement_evals=10,
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
			callback_after_eval=stop_train_callback,
			best_model_save_path=self.model_dir,
		)

		self.callbacks = [eval_callback]

	def set_up_curriculum_callback(self):
		self.curriculum_callback = CurriculumEvalCallback(
			eval_env=self.env,
			best_model_save_path=os.path.join(self.model_dir, MODEL_NAME),
			eval_freq=self.eval_frequency,
			n_eval_episodes=self.eval_episode,
			verbose=1
		)

	def set_up_model(self):
		self.policy_kwargs = dict(
			net_arch=dict(pi=[256,256], vf=[256,256])
		)

		self.model = PPO(
			"MultiInputPolicy",
			self.env,
			learning_rate=1e-3,
			n_steps=4096,
			batch_size=64,
			n_epochs=10,
			gamma=0.995,
			gae_lambda=0.95,
			clip_range=0.2,
			ent_coef=0.05,
			policy_kwargs=self.policy_kwargs,
			verbose=0,
			tensorboard_log=self.log_dir,
			device=self.device
		)

	def train_model(self):
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
		try:
			if hasattr(self, "env") and self.env is not None:
				try:
					self.env.close()
				except Exception:
					pass
		except Exception:
			pass

	def alp_train_curriculum(self, n_bins=5, window_size=150, replay_prob=0.2):
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
   
	def train_curriculum(self):
		print("Starting curriculum training...")
   
		for i, target_state in enumerate(self.target_states_list):
			print(f"\n--- Training on State {i+1}: {target_state.to_string()} ---")
   
			# Close old env to avoid leaking resources
			self.close_env()
   
			# Initialize env for current stage
			self.initialize_environment(target_states_list=[target_state])
	
			if i == 0:
				# First stage: create model from scratch
				self.set_up_model()

			# Bind callback to the current env
			self.set_up_curriculum_callback()

			# Quick evaluation before training to observe baseline on this stage
			try:
				# Ask for episode rewards explicitly to avoid ambiguity
				eval_res = evaluate_policy(
					self.model, self.env,
					n_eval_episodes=5,
					deterministic=True,
					return_episode_rewards=True
				)

				episode_rewards, episode_lengths = eval_res
				mean_reward = float(np.mean(episode_rewards))
				std_reward = float(np.std(episode_rewards))

				print(f"[Pre-Train Eval] mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
			except Exception as e:
				print(f"[Pre-Train Eval] evaluation failed: {e}")

			# learn for this curriculum stage
			print(f"Starting training stage {i+1} for {self.total_timesteps} timesteps...")
			self.model.learn(
				total_timesteps=self.total_timesteps,
				callback=self.curriculum_callback,
				progress_bar=True,
				tb_log_name=f"PPO_stage_{i+1}"
			)

		# final joint training on all states (conservative re-create again)
		print("\n--- Final joint training on all states to promote generalization ---")
		self.close_env()
		self.initialize_environment(target_states_list=[target_state])
	
		self.set_up_curriculum_callback()

		self.model.learn(
			total_timesteps=self.total_timesteps * 2,
			callback=self.curriculum_callback,
			progress_bar=True,
			tb_log_name="PPO_Joint_Training"
		)

		print("\nCurriculum training finished.")

	def build_circuit(self, target_state: Union[TargetState, GeneralTargetState]):
		best_model_path = os.path.join(self.model_dir, MODEL_NAME)
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
		best_model = PPO.load(best_model_path)

		obs, info = eval_env.reset()
		steps, done = 0, False

		while not done and steps < target_state.max_gates:
			action, _ = best_model.predict(obs, deterministic=True)
			obs, reward, terminated, truncated, info_eval = eval_env.step(action)
			done = terminated or truncated
			steps += 1

		fid = info_eval.get("fidelity", 0.0)
		print(f"[End] {target_state.state_name} circuit finished in {steps} steps with fidelity {fid:.4f}")
		print(eval_env.qc.draw())   # <-- Now guaranteed correct circuit
		return