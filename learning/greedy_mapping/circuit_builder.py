from swap_router import swap_router
from initial_mapper import greedy_initial_mapping
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation, Unroll3qOrMore

def check_gate_directions(qc, coupling_map) -> QuantumCircuit:
    """Return a list of the gates that doesn't support bi-direction"""
    new_circuit = QuantumCircuit(qc.num_qubits)
    supported_directions = set(tuple(pair) for pair in coupling_map)
    for gate in qc.data:
        if gate.operation.name == "cx":
            qubit1 = qc.qubits.index(gate.qubits[0])
            qubit2 = qc.qubits.index(gate.qubits[1])
            
            if (qubit1, qubit2) in supported_directions:
                # If the (qubit1, quibit2) exists within the coupling_map then we can build a new cirut based on this
                new_circuit.cx(qubit1, qubit2)
            elif (qubit2, qubit1) in supported_directions:
                # We have to flip the direction of the CX using the h gate
                new_circuit.h(qubit1)
                new_circuit.h(qubit2)
                new_circuit.cx(qubit1, qubit2)
                new_circuit.h(qubit1)
                new_circuit.h(qubit2)
            else:
                raise ValueError(f"CX({qubit1}, {qubit2}) not supported in either direction!")
        else:
            new_circuit.append(gate)
    return new_circuit

def update_circuit_with_physical_mapping(original_circuit, logical_to_physical_mapping) -> QuantumCircuit:
    # Find the highest physical qubit
    num_physical_qubits = max(logical_to_physical_mapping.values()) + 1
    new_circuit = QuantumCircuit(num_physical_qubits)
    
    for instr, qargs, cargs in original_circuit.data:
        # Remap all qubits involved in the operation
        new_qargs = []
        for q in qargs:
            current_qubit = original_circuit.qubits.index(q)
            physical_qubit = logical_to_physical_mapping[current_qubit]
            new_qargs.append(physical_qubit)
        new_circuit.append(instr, new_qargs, cargs)

    return new_circuit

def build_compile_circuit(qc, backend) -> QuantumCircuit:
    # Get the logical and physical mapping using the Greedy approach
    logical_to_physical_mapping, physical_to_logical_mapping = greedy_initial_mapping(qc, backend)
    
    # Verify and update the circuit based on the physical mapping from the backend
    new_circuit = swap_router(qc, backend, logical_to_physical_mapping, physical_to_logical_mapping)
        
    # Verify if we need to fix the CX gates' directions
    pass_manager = PassManager()
    coupling_map = backend.configuration().coupling_map
    new_circuit = check_gate_directions(new_circuit, coupling_map)
    new_circuit = pass_manager.run(new_circuit)

    # 4. Decompose SWAPs into native gates
    pass_manager = PassManager([Unroll3qOrMore(['u3', 'cx'])])
    circuit = pass_manager.run(new_circuit)

    # 5. Optimize
    pm = PassManager([Optimize1qGates(), CommutativeCancellation()])
    new_circuit = pm.run(new_circuit)


    return new_circuit