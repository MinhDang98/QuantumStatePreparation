from enum import Enum
import numpy as np


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
		self.max_gates =  2 * num_qubits  + 1
  
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