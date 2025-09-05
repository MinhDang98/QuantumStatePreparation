from enum import Enum
import numpy as np


class TargetStateName(Enum):
	"""
	An enumeration of predefined target quantum states.
	"""
	BELL_STATE = "BELL_STATE"
	GHZ_STATE = "GHZ_STATE"
	BELL_STATE_PSI = "BELL_STATE_PSI"
	COMPUTATIONAL_BASIS_STATE = "COMPUTATIONAL_BASIS_STATE"
	UNIFORM_SUPERPOSITION = "UNIFORM_SUPERPOSITION"

class GeneralTargetState:
	"""
	Represents a generalized quantum target state, which can be defined dynamically
	for an arbitrary number of qubits. This is useful for scaling training
	to larger systems.
	"""
	def __init__(self,
				 target_state_name: TargetStateName,
				 num_qubits: int):
		"""
		Initializes the GeneralTargetState.

		Args:
			target_state_name (TargetStateName): The name of the target state type.
			num_qubits (int): The number of qubits for this state.
		"""
		self.state_name = target_state_name.value
		self.num_qubits = num_qubits
		self.max_gates =  2 * num_qubits - 1
  
		# Generate the target vector based on the state name
		if target_state_name == TargetStateName.GHZ_STATE:
			self.target_vector = self.generate_ghz_target_vector()
		elif target_state_name == TargetStateName.UNIFORM_SUPERPOSITION:
			self.target_vector = self.generate_uniform_superposition_target_vector()
		else:
			print(f"Unexpected Target State: {target_state_name}")
   
	def generate_ghz_target_vector(self):
		"""
		Generates the vector for a GHZ state for a given number of qubits.
		A GHZ state is an entangled state of the form:
		(1/sqrt(2)) * (|0...0> + |1...1>)

		Returns:
			np.ndarray: The state vector for the GHZ state.
		"""
		vector_size = 2 ** self.num_qubits
		ghz_vector = np.zeros(vector_size, dtype=complex)
		amplitude = 1 / np.sqrt(2)
		
		# Set the first element (|0...0>) and the last element (|1...1>)
		ghz_vector[0] = amplitude
		ghz_vector[vector_size - 1] = amplitude
		
		return ghz_vector

	def generate_uniform_superposition_target_vector(self):
		"""
		Generates the vector for a uniform superposition state.
		This state has equal amplitude for all computational basis states.
		For N qubits, the amplitude is 1 / sqrt(2^N).

		Returns:
			np.ndarray: The state vector for the uniform superposition state.
		"""
		vector_size = 2 ** self.num_qubits
		uniform_amplitude = 1 / np.sqrt(vector_size)
		return np.full(vector_size, uniform_amplitude, dtype=complex)

	def to_string(self):
		"""
		Returns a string representation of the state.

		Returns:
			str: The name of the target state.
		"""
		return self.state_name

class TargetState:
	"""
	Represents a specific, pre-defined quantum target state with a fixed
	number of qubits and a corresponding state vector.
	"""
	def __init__(self, 
				 target_state_name: TargetStateName = None,
				 target_vector: np.ndarray = None,
				 num_qubits: int = None,
				 max_gates: int = None):
		"""
		Initializes the TargetState.

		Args:
			target_state_name (TargetStateName, optional): The name of the target state. Defaults to None.
			target_vector (np.ndarray, optional): A numpy array representing the state vector. Defaults to None.
			num_qubits (int, optional): The number of qubits. Defaults to None.
			max_gates (int, optional): The maximum number of gates allowed. Defaults to None.
		"""
		self.state_name = target_state_name.value
  
		# Initialize the state based on the predefined state name
		if target_state_name == TargetStateName.BELL_STATE:
			# Bell State (|Φ+>)
			# (1/sqrt(2)) * (|00> + |11>)
			bell_state_phi_plus_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
			self.num_qubits = 2
			self.max_gates = 5
			self.target_vector = bell_state_phi_plus_vector
		elif target_state_name == TargetStateName.GHZ_STATE:
			# GHZ State for 3 Qubits (|GHZ>)
			# (1/sqrt(2)) * (|000> + |111>)
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
		"""
		Returns a string representation of the state.

		Returns:
			str: The name of the target state.
		"""
		return self.state_name
