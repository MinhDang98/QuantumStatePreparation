from collections import defaultdict

def count_logical_qubit_usage(circuit) -> dict:
    """Count how often each logical qubit is used in 2-qubit gates."""
    usage = defaultdict(int)
    for gate in circuit.data:
        if gate.operation.name == 'cx':
            for qubit in gate.qubits:
                usage[circuit.qubits.index(qubit)] += 1
    return dict(usage)

def get_physical_qubit_degrees(coupling_map) -> dict:
    """Return a dictionary of physical qubit degrees."""
    degrees = defaultdict(int)
    for src, tgt in coupling_map:
        degrees[src] += 1
        degrees[tgt] += 1
    return dict(degrees)

def greedy_initial_mapping(circuit, backend) -> tuple[dict, dict]:
    """Return a mapping from logical to physical qubits."""
    logical_usage = count_logical_qubit_usage(circuit)
    print(f"Logical qubits usage: {logical_usage}")
    
    coupling_map = backend.configuration().coupling_map
    
    physical_degrees = get_physical_qubit_degrees(coupling_map)
    print(f"Physical degree: {physical_degrees}")
    
    # Sort logical qubits by usage (descending)
    # Prioritizing logical qubits that participate in the most CX gates
    sorted_logical = sorted(logical_usage.items(), key=lambda x: -x[1])
    logical_qubits = [q for q, _ in sorted_logical]

    # Fill missing ones with unused logical qubits
    all_logical = set(range(len(circuit.qubits)))
    print(f"All logical: {all_logical}")
    unused = list(all_logical - set(logical_qubits))
    print(f"Unused: {unused}")
    logical_qubits += unused

    # Sort physical qubits by connectivity degree (descending)
    # Prioritizing physical qubits with the highest connectivity in the hardware coupling map
    sorted_physical = sorted(physical_degrees.items(), key=lambda x: -x[1])
    physical_qubits = [q for q, _ in sorted_physical]

    # Return mapping: logical → physical
    logical_to_physical_mapping = {l: p for l, p in zip(logical_qubits, physical_qubits)}
    physical_to_logical_mapping = {p: l for l, p in zip(logical_qubits, physical_qubits)}
    print(f"Logical → Physical qubit mapping: {logical_to_physical_mapping}")
    print(f"Physical → Logical qubit mapping: {physical_to_logical_mapping}")
    return logical_to_physical_mapping, physical_to_logical_mapping
