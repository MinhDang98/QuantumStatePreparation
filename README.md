# Quantum State Preparation using Reinforcement Learning

This project demonstrates how to use a Deep Q-Network (DQN) agent to find optimal quantum circuits for preparing specific target quantum states. The agent learns by interacting with a custom Gymnasium environment that simulates quantum circuit operations.

## Features

  * **Reinforcement Learning Agent:** Utilizes a Stable Baselines3 DQN agent to learn circuit construction.
  * **Custom Quantum Environment:** A Gymnasium environment (`QuantumStatePreparation`) where actions correspond to applying quantum gates.
  * **Multiple Target States:** Pre-defined target states including Bell states (Φ+, Ψ+), GHZ state, computational basis states, and uniform superposition.
  * **Circuit Visualization:** Built circuits can be visualized using Qiskit's `draw` method.
  * **GPU Support:** Configured for training on NVIDIA GPUs using PyTorch with CUDA, if available.

## Project Structure

  * `agent.py`: Contains the `QuantumnAgent` class, which handles the DQN model setup, training, evaluation, and circuit building.
  * `quantum_state_preparation.py`: Defines the `TargetState` and `QuantumStatePreparation` (Gymnasium environment) classes, which include the quantum state simulation, action space definition, and reward function.
  * `main.ipynb`: A Jupyter Notebook demonstrating how to use the `QuantumnAgent` to train a model and build circuits for various target states.
  * `requirements.txt`: Lists all necessary Python dependencies for the project.

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies from `requirements.txt`:**
    This command will install all required libraries, including `numpy`, `gymnasium`, `stable-baselines3[extra]`, `qiskit`, `matplotlib`, and `torch`.

    ```bash
    pip install -r requirements.txt
    ```

    **Note on PyTorch (torch):** The `requirements.txt` will install a CPU-only version of PyTorch by default. If you intend to use a GPU, you should follow the specific instructions on the official PyTorch website to install the CUDA-enabled version *after* installing from `requirements.txt` (or before, if you prefer to manage it manually). Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and select your specific CUDA version.

    For example, for pip and CUDA 11.8:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

4.  **Verify GPU setup (optional):**
    Open a Python interpreter or a Jupyter cell and run:

    ```python
    import torch
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    ```

    This should output `True` if your GPU is detected and available to PyTorch.

## Usage

The primary way to interact with this project is through the `main.ipynb` Jupyter Notebook.

1.  **Start Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

2.  **Open `main.ipynb`**

3.  **Run the cells:**

      * **Import necessary libraries:** The first cell imports all required modules.
      * **Define Target State:** Instantiate a `TargetState` object by choosing one of the pre-defined target states from `TargetStateName`.
        ```python
        from libs.quantum_state_preparation import TargetState, TargetStateName
        target_state = TargetState(targetStateName=TargetStateName.BELL_STATE)
        ```
      * **Initialize and Train the Agent:** Create a `QuantumnAgent` instance, passing your chosen `target_state` and training parameters. The agent will automatically start training upon initialization.
        ```python
        from libs.agent import QuantumnAgent
        agent = QuantumnAgent(target_state,
                              total_timesteps=8000,
                              eval_frequency=1000, # Example values, adjust as needed
                              eval_episode=10,
                              verbose=0) # Set to 1 for more training logs
        ```
        The `device` used for training (CPU or CUDA) will be printed during agent initialization.
      * **Build and Draw the Circuit:** After training, you can use the `agent.build_circuit()` method to generate a Qiskit `QuantumCircuit` that attempts to prepare the target state.
        ```python
        qc = agent.build_circuit()
        if qc:
            # Print ASCII drawing
            print(qc.draw())
            # Draw using Matplotlib for better visualization
            import matplotlib.pyplot as plt
            qc.draw(output='mpl')
            plt.show() # Crucial for showing plot in scripts/some Jupyter setups
        else:
            print("Failed to build circuit.")
        ```

### Example Target States:

You can switch the `targetStateName` to prepare different states:

```python
from libs.quantum_state_preparation import TargetState, TargetStateName
import numpy as np

# Bell State (|Φ+>)
bell_state_target = TargetState(targetStateName=TargetStateName.BELL_STATE)

# GHZ State for 3 Qubits (|GHZ>)
# Note: Ensure `num_qubits` is set to 3 for GHZ_STATE in quantum_state_preparation.py
# The action space is automatically generated based on num_qubits.
ghz_state_target = TargetState(targetStateName=TargetStateName.GHZ_STATE)

# Another Bell State (|Ψ+>)
bell_state_psi_target = TargetState(targetStateName=TargetStateName.BELL_STATE_PSI)

# Computational Basis State (|10>) for 2 Qubits
comp_basis_target = TargetState(targetStateName=TargetStateName.COMPUTATIONAL_BASIS_STATE)

# Uniform Superposition for 2 Qubits
uniform_superposition_target = TargetState(targetStateName=TargetStateName.UNIFORM_SUPERPOSITION)
```

## Key Components

  * **`TargetState` Class:**

      * Initializes parameters for a specific target quantum state, including `num_qubits`, `max_gates` allowed for its preparation, and the `target_vector` (the desired quantum state in vector form).
      * Pre-defines common quantum states (Bell, GHZ, etc.).

  * **`QuantumStatePreparation` Class (Gymnasium Environment):**

      * **Observation Space:** Represents the current quantum state as a flattened array of real and imaginary parts of the statevector.
      * **Action Space:** Dynamically generated based on `num_qubits`, including single-qubit gates (H, X, Z) and two-qubit CNOT (CX) gates between all pairs of qubits.
      * **`reset()`:** Initializes the circuit and state to `|0...0⟩`.
      * **`step(action)`:** Applies the chosen gate to the quantum circuit (`self.qc`) and evolves the current statevector (`self.current_circuit_state`). It calculates the reward, and determines if the episode is `terminated` (target state reached) or `truncated` (max gates exceeded).
      * **`reward()`:** Calculates fidelity between the current and target state, providing rewards for increased fidelity and penalties for reaching `max_gates`.

  * **`QuantumnAgent` Class:**

      * **`__init__`:** Sets up the environment, Stable Baselines3 DQN model, callbacks, and initiates training. Automatically detects and uses GPU (`cuda`) if available.
      * **`set_up_model()`:** Configures the DQN model with a Multi-Layer Perceptron policy and specifies the `device` for training.
      * **`train_model()`:** Executes the training loop for a specified number of `total_timesteps`.
      * **`build_circuit()`:** Loads the best-trained model and uses it to construct a `QuantumCircuit` object for the target state.

## Customization and Extension

  * **Add New Target States:** In `quantum_state_preparation.py`, you can add new `elif` conditions within the `TargetState` `__init__` method to define more target quantum vectors. Remember to specify `num_qubits` and a reasonable `max_gates` for each new state.
  * **Modify Action Set:** The `generate_action_state` function in `quantum_state_preparation.py` can be modified to include more quantum gates (e.g., Ry, Rz, SWAP, Toffoli) or different qubit interactions.
  * **Adjust RL Parameters:** In `agent.py`, you can experiment with different DQN hyperparameters (e.g., `learning_rate`, `buffer_size`, `exploration_fraction`, `net_arch`) to optimize training performance.
  * **Change RL Algorithm:** You can easily switch from DQN to other Stable Baselines3 algorithms like PPO, A2C, etc., by changing `from stable_baselines3 import DQN` to the desired algorithm and adjusting parameters accordingly.