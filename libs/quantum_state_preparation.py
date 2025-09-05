import gymnasium as gym
import numpy as np
import random
from enum import Enum
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import HGate, CXGate, XGate, ZGate
from qiskit.quantum_info import SparsePauliOp
from .target_state import TargetState, GeneralTargetState, TargetStateName
from typing import Union


class QuantumStatePreparation(gym.Env):
	"""
	This is a custom Gymnasium environment for training a reinforcement learning agent to
	prepare a specific quantum state. The agent learns to apply a sequence of quantum gates
	to an initial state to transform it into the desired target state.

	The state is represented by Pauli expectation values of the current circuit state
	and meta information like fidelity and remaining gates. The actions are a discrete set of
	quantum gates (H, CX, X, Z) applied to specific qubits.

	Attributes:
		metadata (dict): Gymnasium environment metadata.
		target_states_list (list): A list of possible target states for the agent to prepare.
		current_gates_count (int): The number of gates applied in the current episode.
		current_fidelity (float): The current fidelity of the circuit state compared to the target state.
		previous_fidelity (float): The fidelity from the previous step.
		base_gates (list): A list of the basic gate classes available to the agent.
		max_env_qubits (int): The maximum number of qubits the environment can handle.
		max_env_gates (int): The maximum number of gates allowed per episode.
		action_set (list): A list of dictionaries representing all possible gate actions.
		action_space (spaces.Discrete): The discrete action space.
		observation_space (spaces.Dict): The observation space for the agent.
		target_state_object (Union[TargetState, GeneralTargetState]): The target state for the current episode.
		state_vector (Statevector): The Qiskit Statevector object representing the target state.
		num_qubits (int): The number of qubits for the current episode.
		max_gates (int): The max gates for the current episode.
		qc (QuantumCircuit): The Qiskit QuantumCircuit object representing the current circuit.
		current_circuit_state (Statevector): The Qiskit Statevector object representing the current circuit's state.
		valid_actions (np.ndarray): A boolean mask of valid actions for the current episode.
	"""
	metadata = { "render_modes": ["human"], "render_fps": 30}

	def __init__(self,
				 target_states_list: list[Union[TargetState, GeneralTargetState]],
				 max_env_qubits: int,
				 max_env_gates: int):
		"""
		Initializes the QuantumStatePreparation environment.

		Args:
			target_states_list (list): A list of possible target states.
			max_env_qubits (int): The maximum number of qubits the environment should support.
			max_env_gates (int): The maximum number of gates the environment should support.
		"""
		super().__init__()

		# General parameters
		self.target_states_list = target_states_list
		self.current_gates_count = 0
		self.current_fidelity = -1
		self.previous_fidelity = 0
		self.base_gates = [HGate, CXGate, XGate, ZGate, None]

		if not self.target_states_list:
			raise ValueError("target_states_list cannot be empty.")

		# Determine the maximum number of qubits and max gates for the environment's fixed spaces
		self.max_env_qubits = max_env_qubits
		self.max_env_gates = max_env_gates

		# Generate appropriate action set
		self.action_set = self.generate_action_state(self.max_env_qubits)
		self.action_space = spaces.Discrete(len(self.action_set))

		# Initialize the quantum circuit and state to their default starting values.
		self.reset()

		self.observation_space = spaces.Dict({
			"pauli": spaces.Box(low=-1.0, high=1.0, shape=(3 * self.max_env_qubits,), dtype=np.float32),
			"meta": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) # Fidelity & Gates Remaining
		})

	def set_up_target_state(self, target_state_object: Union[TargetState, GeneralTargetState] = None):
		"""
		Sets the target state for the current episode.

		Args:
			target_state_object (Union[TargetState, GeneralTargetState], optional): The target state to use.
			If None, a random state is chosen from the list. Defaults to None.
		"""
		self.target_state_object: Union[TargetState, GeneralTargetState] = None
		if target_state_object is not None:
			self.target_state_object = target_state_object
		else:
			self.target_state_object = random.choice(self.target_states_list)
		self.state_vector = Statevector(self.target_state_object.target_vector)
		self.num_qubits = self.target_state_object.num_qubits
		self.max_gates = self.target_state_object.max_gates

	def compute_pauli_expectations(self, statevector, num_qubits):
		"""
		Computes the Pauli expectation values for a given statevector.

		Args:
			statevector (Statevector): The statevector for which to compute the expectations.
			num_qubits (int): The number of qubits.

		Returns:
			np.ndarray: An array of Pauli expectation values (X, Y, Z for each qubit).
		"""
		expectations = []
		for i in range(num_qubits):
			for pauli_str in ['X', 'Y', 'Z']:
				label = ['I'] * num_qubits
				label[i] = pauli_str
				pauli_op = SparsePauliOp.from_list([("".join(label), 1.0)])
				expectation = np.real(statevector.expectation_value(pauli_op))
				expectations.append(expectation)
		return np.array(expectations, dtype=np.float32)

	def _get_obs(self):
		"""
		Generates the observation for the agent.

		Returns:
			dict: A dictionary containing the 'pauli' vector and 'meta' data.
		"""
		# Compute Pauli expectations for only the active qubits
		pauli_obs = self.compute_pauli_expectations(self.current_circuit_state, self.num_qubits)  # shape: 3 * num_qubits

		# Pad the Pauli vector if num_qubits < max_env_qubits
		pad_length = 3 * (self.max_env_qubits - self.num_qubits)
		if pad_length > 0:
			pauli_obs = np.pad(pauli_obs, (0, pad_length), mode='constant')

		gates_remaining = self.max_gates - self.current_gates_count
		gates_remaining_norm = np.clip(gates_remaining / max(1, self.max_gates), 0.0, 1.0)

		obs = {
			"pauli": pauli_obs.astype(np.float32),
			"meta": np.array([self.current_fidelity, gates_remaining_norm], dtype=np.float32)
		}
		return obs

	def obs_to_string(self, obs):
		"""
		Prints a human-readable representation of the observation.

		Args:
			obs (dict): The observation dictionary.
		"""
		num_qubits = self.num_qubits
		pauli_len = 3 * num_qubits

		pauli_obs = obs[:pauli_len]
		fidelity = obs[pauli_len]
		gates_remaining_norm = obs[pauli_len + 1]

		print("\tNew Observation:")
		# Pauli coefficients
		for q in range(num_qubits):
			x = pauli_obs[3*q]
			y = pauli_obs[3*q + 1]
			z = pauli_obs[3*q + 2]
			print(f"\t\tQubit {q}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

		# Meta info
		print(f"\t\tFidelity: {fidelity:.4f}")
		print(f"\t\tGates Remaining (normalized): {gates_remaining_norm:.4f}")

	def _get_info(self):
		"""
		Generates the info dictionary for the agent.

		Returns:
			dict: A dictionary containing fidelity, gates count, and Pauli expectations.
		"""
		from qiskit.quantum_info import Pauli

		# Fidelity
		fidelity = np.abs(self.current_circuit_state.inner(self.state_vector)) ** 2

		# Pauli expectation values
		pauli_labels = ["X", "Y", "Z"]
		expectations = {}
		for q in range(self.num_qubits):
			for pauli in pauli_labels:
				# Build operator: pauli on q-th qubit, I elsewhere
				label = ["I"] * self.num_qubits
				label[q] = pauli
				op = Pauli("".join(label))
				exp_val = self.current_circuit_state.expectation_value(op).real
				expectations[f"{pauli}{q}"] = exp_val

		return {
			"fidelity": fidelity,
			"gates_count": self.current_gates_count,
			"pauli_expectations": expectations
		}

	def info_to_string(self, info):
		"""
		Prints a human-readable representation of the info dictionary.

		Args:
			info (dict): The info dictionary.
		"""
		fidelity = info.get("fidelity", None)
		gates_count = info.get("gates_count", None)

		print("\tInfo:")
		# Fidelity and gate count
		print(f"\t\tCurrent Fidelity: {fidelity:.4f}" if fidelity is not None else "\t\tCurrent Fidelity: N/A")
		print(f"\t\tCurrent Gate Count: {gates_count}" if gates_count is not None else "\t\tCurrent Gate Count: N/A")

		# Pauli expectation values for each qubit
		pauli_labels = ["X", "Y", "Z"]
		for q in range(self.num_qubits):
			expectations = []
			for pauli in pauli_labels:
				# Calculate expectation value: ⟨ψ|P|ψ⟩
				from qiskit.quantum_info import Pauli, Statevector
				ev = Statevector(self.current_circuit_state).expectation_value(Pauli(f"{pauli}{'I'*(self.num_qubits-1-q)}"))
				expectations.append(f"{pauli}={ev.real:.4f}")
			print(f"\t\tQubit {q}: " + ", ".join(expectations))

	def generate_action_state(self, num_qubits: int):
		"""
		Generates the complete set of possible actions.

		Args:
			num_qubits (int): The number of qubits to consider for actions.

		Returns:
			list: A list of action dictionaries.
		"""
		action_space = []
		for gate_cls in self.base_gates:
			if gate_cls is CXGate:
				for i in range(num_qubits):
					for j in range(num_qubits):
						if i == j:
							continue
						action_space.append({"gate_class": CXGate, "qubits": [i, j]})
			else:
				for i in range(num_qubits):
					action_space.append({"gate_class": gate_cls, "qubits": [i]})
		# Added a None action so that we can terminate early
		action_space.append({"gate_class": None, "qubits": None})
		return action_space

	def compute_valid_action_mask(self):
		"""
		Computes a boolean mask indicating which actions are valid for the current episode.

		Returns:
			np.ndarray: A boolean array where True indicates a valid action.
		"""
		mask = []
		for a in self.action_set:
			if a["qubits"] is None:
				mask.append(True)
				continue

			# invalid if any qubit index >= current num_qubits
			if any(q >= self.num_qubits for q in a["qubits"]):
				mask.append(False)
				continue

			mask.append(True)
   
		return np.array(mask, dtype=bool)

	def reset(self, seed=None, target_state_object: Union[TargetState, GeneralTargetState] = None):
		"""
		Resets the environment for a new episode.

		Args:
			seed (int, optional): The random seed. Defaults to None.
			target_state_object (Union[TargetState, GeneralTargetState], optional): The target state for the new episode.
			Defaults to None.

		Returns:
			tuple: A tuple containing the initial observation and info dictionary.
		"""
		super().reset(seed=seed)

		self.set_up_target_state(target_state_object)

		# Initialize a fresh quantum circuit with the specified number of qubits.
		self.qc = QuantumCircuit(self.num_qubits)

		# Statevector.from_int(0, dim) creates a statevector where only the |0...0> component is 1.
		self.current_circuit_state = Statevector.from_int(0, 2 ** self.num_qubits)

		# Reset the gate counter
		self.current_gates_count = 0

		# Re-calculate the valid action
		self.valid_actions = self.compute_valid_action_mask()

		observation = self._get_obs()
		info = self._get_info()
		return observation, info

	def step(self, action):
		"""
		Takes a step in the environment by applying the chosen action.

		Args:
			action (int): The index of the action to take.

		Returns:
			tuple: A tuple containing the new observation, reward, terminated flag, truncated flag, and info dictionary.
		"""
		if not self.valid_actions[action]:
			# Penalize and ignore the illegal action
			self.current_gates_count += 1
			reward = -10.0
			terminated = False
			truncated = False
			if self.current_gates_count >= self.max_gates:
				truncated = True
			observation = self._get_obs()
			info = self._get_info()
			return observation, reward, terminated, truncated, info

		# Get the gate, and qubits from the action set
		gate_info = self.action_set[action]

		# Check if the model choose the None action
		if gate_info["gate_class"] is None:
			reward, terminated, truncated = self.reward(isNone=True)
			observation = self._get_obs()
			info = self._get_info()
			# Need to follow this order with stable-baseline3
			return observation, reward, terminated, truncated, info

		# Create a fresh instance of the target gate since we are only using it as a Class so far
		gate = gate_info["gate_class"]()
		qubits = gate_info["qubits"]

		# Append the gate to the actual circuit object (self.qc)
		self.qc.append(gate, qubits)

		# We need to create a tempmorary circuit so that we can correctly evolve our current circuit
		# Create a temporary QuantumCircuit to correctly apply the gate to our circuit
		temp_qc = QuantumCircuit(self.num_qubits)
		temp_qc.append(gate, qubits)

		# Get the unitary matrix representation of this single-gate circuit
		gate_unitary_matrix = Operator(temp_qc).data

		# Apply the chosen gate to the current quantum state.
		gate_op = Operator(gate_unitary_matrix)
		self.current_circuit_state = self.current_circuit_state.evolve(gate_op)
		self.current_gates_count += 1

		# Calculate the reward
		reward, terminated, truncated = self.reward()

		observation = self._get_obs()
		info = self._get_info()

		# Need to follow this order with stable-baseline3
		return observation, reward, terminated, truncated, info

	def reward(self, isNone: bool = False):
		"""
		Improved reward shaping:
		- Dense shaping from fidelity delta
		- Negative reward if fidelity stagnates
		- Balanced terminal bonuses to reduce variance
		"""
		FID_THRESHOLD = 0.99        # Slightly relaxed for learning stability
		SUCCESS_BONUS = 10.0
		FAILURE_PENALTY = -1.0
		BASE_STEP_PENALTY = 0.1    # Mild gate cost

		# Update fidelity
		self.previous_fidelity = self.current_fidelity if self.current_fidelity != -1 else 0.0
		self.current_fidelity = np.abs(self.current_circuit_state.inner(self.state_vector)) ** 2

		terminated = False
		truncated = False

		# Success
		if self.current_fidelity >= FID_THRESHOLD:
			reward = SUCCESS_BONUS - (self.current_gates_count * BASE_STEP_PENALTY)
			terminated = True
			return reward, terminated, truncated

		# Out of gates
		if self.current_gates_count >= self.max_gates:
			reward = FAILURE_PENALTY
			truncated = True
			return reward, terminated, truncated

		# 'None' (early stop)
		if isNone:
			terminated = True
			reward = (self.current_fidelity * SUCCESS_BONUS) - (self.current_gates_count * BASE_STEP_PENALTY)
			return reward, terminated, truncated

		# Step shaping
		delta_fid = self.current_fidelity - self.previous_fidelity

		if delta_fid > 0:
			reward = delta_fid * 2.0 - BASE_STEP_PENALTY
		else:
			# Actively punish stagnation or regression
			reward = delta_fid * 2.0 - 0.05  

		return reward, terminated, truncated

	def render(self):
		"""
		Renders the environment.
		"""
		pass

	def close(self):
		"""
		Closes the environment.
		"""
		pass
