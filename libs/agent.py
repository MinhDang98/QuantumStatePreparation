import gymnasium as gym
import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from .quantum_state_preparation import QuantumStatePreparation, TargetState

class QuantumnAgent():
    def __init__(self, target_state: TargetState, total_timesteps, eval_frequency, eval_episode, verbose):
        # Set up basic properties
        self.target_state = target_state
        self.total_timesteps = total_timesteps
        self.eval_frequency = eval_frequency
        self.eval_episode = eval_episode
        self.verbose = verbose
        
        self.log_dir = "./logs/DQN_Bell_State"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model_name = "DQN_Bell_State"
        
        # Determine the device to use
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
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
        
    def initialize_environment(self):
        # make_vec_env handles creation and wrapping for stable-baselines3 compatibility.
        # env_kwargs passes arguments to your QuantumStatePreparation.__init__
        self.env = make_vec_env(QuantumStatePreparation,
                           n_envs=1,
                           env_kwargs={'target_state': self.target_state})
        
        self.evaluate_env = QuantumStatePreparation(self.target_state)
        
    def set_up_callbacks(self):
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=os.path.join(self.log_dir, "DQN_Bell_State_Model"),
            log_path=self.log_dir,
            eval_freq=self.eval_frequency,
            n_eval_episodes=self.eval_episode,
            deterministic=True,
            render=False,
        )

        self.callbacks = [eval_callback]
    
    def set_up_model(self):
        self.model = DQN(
            "MlpPolicy",                # Multi-Layer Perceptron policy (standard feedforward neural network)
            self.env,
            learning_rate=1e-3,
            buffer_size=100000,
            learning_starts=100,       # Number of steps before learning begins (to fill replay buffer)
            batch_size=32,
            train_freq=(4, "step"),     # Train every 4 steps
            target_update_interval=100,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[128,128]), # Two hidden layers of 64 neurons each
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
        self.model.save(os.path.join(self.log_dir, self.model_name))
        print(f"Final model saved to {os.path.join(self.log_dir, 'DQN_Bell_State_Model')}")

    def eval_model(self):
        print("\n--- Evaluating the Best Model ---")
        best_model_path = os.path.join(self.log_dir, self.model_name + ".zip")
        
        if os.path.exists(best_model_path):
            best_model = DQN.load(best_model_path, env=self.env)
            
            success_count = 0
            total_eval_episode = 50
            
            for i in range(total_eval_episode):
                obs, info = self.evaluate_env.reset()
                done = False
                steps = 0

                while not done and steps < self.evaluate_env.max_gates + 2:
                    action, state = best_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info_eval = self.evaluate_env.step(action)
                    done = terminated or truncated
                    steps += 1
                    if terminated:
                        success_count += 1
                        break
                    elif truncated:
                        break
                    
            print(f"\n--- Best Model Evaluation Summary ---")
            print(f"Successfully in {success_count}/{total_eval_episode} episodes ({success_count/total_eval_episode:.2%}).")
        else:
            print(f"No best model found at {best_model_path}")

    def build_circuit(self):
        best_model_path = os.path.join(self.log_dir, self.model_name + ".zip")
        if os.path.exists(best_model_path):
            best_model = DQN.load(best_model_path, env=self.env)
            
            success_count = 0
            total_eval_episode = 50
            
            for i in range(total_eval_episode):
                obs, info = self.evaluate_env.reset()
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