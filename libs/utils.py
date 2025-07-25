import numpy as np
import os
import shutil
from libs.quantum_state_preparation import QuantumStatePreparation
from qiskit.quantum_info import Statevector

def test_gym():
    env = QuantumStatePreparation(num_qubits=2, max_gates=5)
    print(f"Environment instantiated. Max gates: {env.max_gates}, Num qubits: {env.num_qubits}")
    
    obs, info = env.reset()
    print("\n--- After Reset ---")
    env.obs_to_string(obs)
    print(f"Initial Fidelity: {info['fidelity']:.4f}")
    print(f"Initial Gates Applied: {info['gates_count']}")
    
    expected_initial_fidelity = np.abs(env.state_vector.inner(Statevector.from_int(0, 2**env.num_qubits)))**2
    print(f"Expected Initial Fidelity: {expected_initial_fidelity:.4f}")
    
    print("\n--- Performing Steps ---")
    done = False
    current_step = 0
    while not done and current_step < env.max_gates + 3: # Add a few extra step to see if we will get truncated
        # This is the optimal sequence for Bell state from |00>
        action_list = [0, 2]
        
        if current_step < len(action_list):
            action = action_list[current_step]
        else:
            action = env.action_space.sample()
        
        print(f"\nStep {current_step + 1}: Taking action {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.obs_to_string(obs)
        print(f"\tFidelity: {info['fidelity']:.4f}")
        print(f"\tGates Applied: {info['gates_count']}")
        print(f"\tReward: {reward:.2f}")
        print(f"\tTerminated: {terminated}, Truncated: {truncated}")

        current_step += 1

def clean_log(log_folder_path: str = "./logs"):
    if not os.path.exists(log_folder_path):
        print(f"Log folder '{log_folder_path}' does not exist.")
        return

    print(f"Cleaning contents of '{log_folder_path}'...")
    for item in os.listdir(log_folder_path):
        item_path = os.path.join(log_folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or link
                print(f"  Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
                print(f"  Deleted directory: {item_path}")
        except Exception as e:
            print(f"  Error deleting {item_path}: {e}")
    print(f"Finished cleaning '{log_folder_path}'.")