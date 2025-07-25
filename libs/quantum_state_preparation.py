import gymnasium as gym
import numpy as np
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
	
class TargetState:
	def __init__(self, 
			  	 targetStateName: TargetStateName):
		if targetStateName == TargetStateName.BELL_STATE:
			# Bell State (|Φ+>)
			# A 2-qubit entangled state
			bell_state_phi_plus_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 5
			self.target_vector = bell_state_phi_plus_vector
		elif targetStateName == TargetStateName.GHZ_STATE:
			# GHZ State for 3 Qubits (|GHZ>)
			# A 3-qubit entangled state: (1/sqrt(2)) * (|000> + |111>)
			# The vector will have 2^3 = 8 elements
			ghz_state_3_qubits_vector = np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)], dtype=complex)
			self.num_qubits = 3
			self.max_gates = 8
			self.target_vector = ghz_state_3_qubits_vector
		elif targetStateName == TargetStateName.BELL_STATE_PSI:
	  		# Another Bell State (|Ψ+>)
			# (1/sqrt(2)) * (|01> + |10>)
			bell_state_psi_plus_vector = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 5
			self.target_vector = bell_state_psi_plus_vector
		elif targetStateName == TargetStateName.COMPUTATIONAL_BASIS_STATE:
	  		# Computational Basis State (|10>) for 2 Qubits
			# Represents the state where the first qubit is 1 and the second is 0
			computational_basis_10_vector = np.array([0, 0, 1, 0], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 3
			self.target_vector = computational_basis_10_vector
		elif targetStateName == TargetStateName.UNIFORM_SUPERPOSITION:
			# Uniform Superposition for 2 Qubits
			# (1/2) * (|00> + |01> + |10> + |11>)
			uniform_superposition_2_qubits_vector = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 6
			self.target_vector = uniform_superposition_2_qubits_vector
		else:
			print(f"Unexpected Target State: {targetStateName}")
		
class QuantumStatePreparation(gym.Env):
	metadata = { "render_modes": ["human"], "render_fps": 30}
	
	def __init__(self,
				 target_state: TargetState):
		super().__init__()

		# General parameters
		self.num_qubits = target_state.num_qubits
		self.max_gates = target_state.max_gates
		self.current_gates_count = 0
		self.current_fidelity = -1
		self.previous_fidelity = 0
		
		# Target state
		# For example, Bell state: |Phi+> = (1/sqrt(2)) * (|00> + |11>)
		# Amplitude P = a^2 => (1/sqrt(2))^2 would be 1/2 which is 50%
		# For 2 qubits, the computational basis states are |00>, |01>, |10>, |11>
		# The statevector is a 4-element complex array: [amplitude_00, amplitude_01, amplitude_10, amplitude_11]
		# So, for |Phi+>, it's [1/sqrt(2), 0, 0, 1/sqrt(2)]
		target_vector = target_state.target_vector
		self.state_vector = Statevector(target_vector)
		
		# Action space
		self.action_set = self.generate_action_state(self.num_qubits)
		self.action_space = spaces.Discrete(len(self.action_set))

		# Observer
		# We represent the quantum state as the full statevector (complex numbers).
		# Since RL models typically work with real numbers, we flatten the real and imaginary parts.
		# For N qubits, the statevector has 2^N complex components.
		# So, observation dimension = 2 * (2^N) real numbers.
		obs_dim = 2 * (2 ** self.num_qubits)
		self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))
		
  		# Initialize the quantum circuit and state to their default starting values.
		self.reset()

	def _get_obs(self):
		current_state_vector = self.current_circuit_state.data
		# Concatenate the real parts and imaginary parts into a single 1D numpy array.
		# For exmaple, for 2 qubits, it will an 8-dimensional vector with 4 amplitude. 
		# So, we can understand it as a+ib. Hence, it 4 for real part and 4 for imaginary part
		return np.array(np.concatenate((current_state_vector.real, current_state_vector.imag)), dtype=np.float32)

	def obs_to_string(self, obs):
		half = int(len(obs) / 2)
		print(f"\tNew Observation:")
		# This is the real part
		for i in range(half):
			print(f"\t\tReal value: {obs[i]} - Imaginary value: {obs[i + half]}")

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
    
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
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
