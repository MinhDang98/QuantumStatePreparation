import gymnasium as gym
import numpy as np
import random
from enum import Enum
from gymnasium import spaces
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import HGate, CXGate, XGate, ZGate

class TargetStateName(Enum):
	BELL_STATE = "BELL_STATE"
	GHZ_STATE = "GHZ_STATE"
	BELL_STATE_PSI = "BELL_STATE_PSI"
	COMPUTATIONAL_BASIS_STATE = "COMPUTATIONAL_BASIS_STATE"
	UNIFORM_SUPERPOSITION = "UNIFORM_SUPERPOSITION"

class GeneralTargetState:
	def __init__(self,
				 target_state_name: TargetStateName,
				 num_qubits: int):
		self.state_name = target_state_name.value
		self.num_qubits = num_qubits
		self.max_gates =  2 * num_qubits
  
		if target_state_name == TargetStateName.GHZ_STATE:
			self.target_vector = self.generate_ghz_target_vector()
		elif target_state_name == TargetStateName.UNIFORM_SUPERPOSITION:
			self.target_vector = self.generate_uniform_superposition_target_vector()
		else:
			print(f"Unexpected Target State: {target_state_name}")
   
	def generate_ghz_target_vector(self):
		vector_size = 2 ** self.num_qubits
		ghz_vector = np.zeros(vector_size, dtype=complex)
		amplitude = 1 / np.sqrt(2)
		
		# Set the first element (|0...0>) and the last element (|1...1>)
		ghz_vector[0] = amplitude
		ghz_vector[vector_size - 1] = amplitude
		
		return ghz_vector

	def generate_uniform_superposition_target_vector(self):
		vector_size = 2 ** self.num_qubits
		uniform_amplitude = 1 / np.sqrt(vector_size)
		return np.full(vector_size, uniform_amplitude, dtype=complex)

	def to_string(self):
		return self.state_name

class TargetState:
	def __init__(self, 
			  	 target_state_name: TargetStateName = None,
	   			 target_vector: np.ndarray = None,
				 num_qubits: int = None,
				 max_gates: int = None):
		self.state_name = target_state_name.value
  
		if target_state_name == TargetStateName.BELL_STATE:
			# Bell State (|Φ+>)
			# For example, Bell state: |Phi+> = (1/sqrt(2)) * (|00> + |11>)
			# Amplitude P = a^2 => (1/sqrt(2))^2 would be 1/2 which is 50%
			# For 2 qubits, the computational basis states are |00>, |01>, |10>, |11>
			# The statevector is a 4-element complex array: [amplitude_00, amplitude_01, amplitude_10, amplitude_11]
			# So, for |Phi+>, it's [1/sqrt(2), 0, 0, 1/sqrt(2)]
			bell_state_phi_plus_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 5
			self.target_vector = bell_state_phi_plus_vector
		elif target_state_name == TargetStateName.GHZ_STATE:
			# GHZ State for 3 Qubits (|GHZ>)
			# A 3-qubit entangled state: (1/sqrt(2)) * (|000> + |111>)
			# The vector will have 2^3 = 8 elements
			ghz_state_3_qubits_vector = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)], dtype=complex)
			self.num_qubits = 3
			self.max_gates = 8
			self.target_vector = ghz_state_3_qubits_vector
		elif target_state_name == TargetStateName.BELL_STATE_PSI:
	  		# Another Bell State (|Ψ+>)
			# (1/sqrt(2)) * (|01> + |10>)
			bell_state_psi_plus_vector = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 5
			self.target_vector = bell_state_psi_plus_vector
		elif target_state_name == TargetStateName.COMPUTATIONAL_BASIS_STATE:
	  		# Computational Basis State (|10>) for 2 Qubits
			# Represents the state where the first qubit is 1 and the second is 0
			computational_basis_10_vector = np.array([0, 0, 1, 0], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 3
			self.target_vector = computational_basis_10_vector
		elif target_state_name == TargetStateName.UNIFORM_SUPERPOSITION:
			# Uniform Superposition for 2 Qubits
			# (1/2) * (|00> + |01> + |10> + |11>)
			uniform_superposition_2_qubits_vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 6
			self.target_vector = uniform_superposition_2_qubits_vector
		elif target_vector is not None and num_qubits is not None and max_gates is not None:
			# This path allows direct instantiation with custom parameters
			self.target_vector = target_vector
			self.num_qubits = num_qubits
			self.max_gates = max_gates
		else:
			print(f"Unexpected Target State: {target_state_name}")

	def to_string(self):
		return self.state_name
		
class QuantumStatePreparation(gym.Env):
	metadata = { "render_modes": ["human"], "render_fps": 30}
	
	def __init__(self,
				 target_states_list: list[TargetState],
	 			 max_env_qubits: int,
				 max_env_gates: int):
		super().__init__()

		# General parameters
		self.target_states_list = target_states_list
		self.current_gates_count = 0
		self.current_fidelity = -1
		self.previous_fidelity = 0
		
		if not self.target_states_list:
			raise ValueError("target_states_list cannot be empty.")

		# --- Determine the maximum number of qubits and max gates for the environment's fixed spaces ---
		self.max_env_qubits = max_env_qubits
		self.max_env_gates = max_env_gates
		
		# Initialize the quantum circuit and state to their default starting values.
		self.reset()

		# Action space
		self.action_set = self.generate_action_state(self.max_env_qubits)
		self.action_space = spaces.Discrete(len(self.action_set))

		# Observer includes both current state AND target state
		# We represent the quantum state as the full statevector (complex numbers).
		# Since RL models typically work with real numbers, we flatten the real and imaginary parts.
		# For N qubits, current_state is 2 * 2 ^ N, target_state is 2 * 2 ^ N
		# So, observation dimension = 2 * (2 ^ N) real numbers.
		obs_dim = 4 * (2 ** self.max_env_qubits)
		# When we define gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32), 
		# this shape is immutable for the lifetime of the environment instance. 
		# Stable Baselines3 initializes internal buffers and network structures based on this fixed shape.
		self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))

	def set_up_target_state(self, target_state_object: TargetState = None):
		self.target_state_object: TargetState = None
		if target_state_object is not None:
			self.target_state_object = target_state_object
		else:
			self.target_state_object = random.choice(self.target_states_list)
		self.state_vector = Statevector(self.target_state_object.target_vector)
		self.num_qubits = self.target_state_object.num_qubits
		self.max_gates = self.target_state_object.max_gates
  
	def _get_obs(self):
		# Get the data for the current and target statevectors
		current_sv_data = self.current_circuit_state.data
		target_sv_data = self.state_vector.data

		# Pad current_circuit_state and target_vector if their num_qubits is less than max_env_qubits
		# The dimension of the statevector is 2^N.
		padding_dim = 2**self.max_env_qubits - 2**self.num_qubits
		
		# Create zero arrays for padding
		zero_pad = np.zeros(padding_dim, dtype=complex)

		# Pad the actual state vectors if necessary
		padded_current_sv_data = np.concatenate((current_sv_data, zero_pad)) if padding_dim > 0 else current_sv_data
		padded_target_sv_data = np.concatenate((target_sv_data, zero_pad)) if padding_dim > 0 else target_sv_data
  
		# Ensure all parts are float32
		observation = np.concatenate((padded_current_sv_data.real, padded_current_sv_data.imag,
									  padded_target_sv_data.real, padded_target_sv_data.imag)).astype(np.float32)
		return observation

	def obs_to_string(self, obs):
		quarter = int(len(obs) / 4) # Now 4 parts: current_real, current_imag, target_real, target_imag
		print(f"\tNew Observation (Current | Target):")
		for i in range(quarter):
			print(f"\t\tCurrent Real: {obs[i]:.4f} - Current Imag: {obs[i + quarter]:.4f}")
			print(f"\t\tTarget Real: {obs[i + 2*quarter]:.4f} - Target Imag: {obs[i + 3*quarter]:.4f}")

	def _get_info(self):
		# Calculate fidelity: Measures how close the current state is to the target state.\
		# fidelity = |<target_state|current_state>|^2
		# np.abs() calculates the magnitude of the complex inner product.
		fidelity = np.abs(self.current_circuit_state.inner(self.state_vector)) ** 2
		return {"fidelity": fidelity, "gates_count": self.current_gates_count}

	def info_to_string(self, info):
		print(f"\t\tCurrent Fidelity: {info["fidelity"]}, Current Gate Count: {info["gates_count"]}")
	
	def generate_action_state(self, num_qubits: int):
		action_space = []
		gates = [HGate(), XGate(), ZGate(), CXGate()]
		qubit_index = list(range(num_qubits))
  
		for g in gates:
			for i in qubit_index:
				if (g == HGate() or g == XGate() or g == ZGate()):
					# { "gate": HGate(), "qubits": [0]},
					action_space.append({"gate": g, "qubits": [i]})
				elif (g == CXGate()):
					for j in qubit_index:
						if i == j:
							continue
						# { "gate": CXGate(), "qubits": [0, 1]},
						action_space.append({"gate": g, "qubits": [i, j]})
		return action_space
	
	def reset(self, seed=None, target_state_object: TargetState = None):
		super().reset(seed=seed)
  
		self.set_up_target_state(target_state_object)
		
		# Initialize a fresh quantum circuit with the specified number of qubits.
		self.qc = QuantumCircuit(self.num_qubits)

  		# Statevector.from_int(0, dim) creates a statevector where only the |0...0> component is 1.
		self.current_circuit_state = Statevector.from_int(0, 2**self.num_qubits)

		# reset the gate counter
		self.current_gates_count = 0
  
		observation = self._get_obs()
		info = self._get_info()
		return observation, info

	def step(self, action):
		# Get the gate, and qubits from the action set
		gate_info = self.action_set[action]
		gate = gate_info["gate"]
		qubits = gate_info ["qubits"]
          
        # If the chosen action targets a qubit index beyond the current target's num_qubits,
		# or involves qubits not relevant to the current state,
		# penalize heavily and truncate, or ignore (penalize is better for RL).
		# This is a simple check; more complex masking might be needed for very large action spaces.
		if any(q >= self.num_qubits for q in qubits):
			# Penalize for applying gate to non-existent or inactive qubit
			reward = -10.0 # Large penalty
			terminated = False
			truncated = True
			observation = self._get_obs() # Still return observation
			info = self._get_info()
			return observation, reward, terminated, truncated, info

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
  
	def reward(self):
		reward = 0.0

		# Update the previous fidelity
		if self.current_fidelity != -1:
			self.previous_fidelity = self.current_fidelity
   
		# If fidelity = 1.0, the states are exactly the same.
		# If fidelity = 0, the states are completely orthogonal (as different as possible).
		self.current_fidelity = np.abs(self.current_circuit_state.inner(self.state_vector)) ** 2

		terminated = False
		truncated = False
  
		if self.current_fidelity > 0.99:
			# This is our target
			reward += 10
			terminated = True
		elif self.current_gates_count >= self.max_gates:
			# We cannot reach the target fidelity within the limit
			reward -= 5
			truncated = True
		else:
			# Lastly, we need to encourage the model to find the shortest circuits
			reward = (self.current_fidelity - self.previous_fidelity) * 5.0 - 0.5

		return reward, terminated, truncated

	def render(self):
		pass

	def close(self):
		pass
