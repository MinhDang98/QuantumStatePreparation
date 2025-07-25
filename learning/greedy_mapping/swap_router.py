from collections import deque, defaultdict
from qiskit import QuantumCircuit

def build_adj_list(coupling_map) -> dict:
    adj_list = defaultdict(set)
    for current, next in coupling_map:
        adj_list[current].add(next)
        adj_list[next].add(current)
    return dict(adj_list)

def find_valid_path(adj_list, source, target)  -> list:
    "Find the shortest path between from current vertex to the target vertex"
    queue = deque([[source]]) # We need to queue up the whole path of the current source
    visited = set()
    
    while(queue):
        current_path = queue.popleft()
        # extract the last node in the path to move to the next neighbor
        next_node = current_path[-1]

        if next_node == target:
            return current_path
        
        if next_node not in visited:
            visited.add(next_node)
            for neighbor in adj_list[next_node]:
                # Now we need to add each of the current neighbor of the next_node from the adj_list
                # to find the path
                new_path = list(current_path)
                new_path.append(neighbor)
                queue.append(new_path)
    return []
        
def update_mappings_for_swap(physical_a, physical_b, logical_to_physical_mapping, physical_to_logical_mapping):
    """
    Updates the logical <-> physical mappings after a SWAP operation on physical_a and physical_b.
    """
    logical_a = physical_to_logical_mapping.get(physical_a)
    logical_b = physical_to_logical_mapping.get(physical_b)

    if logical_a is not None:
        logical_to_physical_mapping[logical_a] = physical_b
    if logical_b is not None:
        logical_to_physical_mapping[logical_b] = physical_a
    
    physical_to_logical_mapping[physical_a] = logical_b
    physical_to_logical_mapping[physical_b] = logical_a
    
    print(f"Updated Logical -> Physical mapping: {logical_to_physical_mapping}")
    print(f"Updated Physical -> Logical mapping: {physical_to_logical_mapping}")

def swap_router(circuit, backend, logical_to_physical_mapping, physical_to_logical_mapping) -> QuantumCircuit:
    coupling_map = backend.configuration().coupling_map
    adj_matrix = build_adj_list(coupling_map)
    print(f"Current adjacency matrix from the backend: {adj_matrix}")
    
    # Determine the number of physical qubits needed based on the backend's coupling map
    # or by finding the max qubit index in the coupling map.
    num_physical_qubits = 0
    if coupling_map:
        num_physical_qubits = max(max(pair) for pair in coupling_map) + 1
    else: # Handle backends with no coupling map (e.g., simulators with all-to-all connectivity)
        num_physical_qubits = circuit.num_qubits # Or some other default

    # Create a new circuit that will hold the transpiled (physical) gates
    new_circuit = QuantumCircuit(num_physical_qubits, circuit.num_clbits)
 
    print(f"\nRouting process started. Initial mappings are used internally.")

    # We will iterate through the original logical circuit's gates
    for gate_instruction in circuit.data:
        op = gate_instruction.operation
        # Get logical qubit indices from the original circuit's instruction
        logical_qargs = [circuit.qubits.index(q) for q in gate_instruction.qubits]
        clargs = gate_instruction.clbits

        # For 2-qubit gates (like 'cx'), we need to handle connectivity
        if op.name == "cx":
            logical_q1, logical_q2 = logical_qargs[0], logical_qargs[1]

            # Get the CURRENT physical locations of these logical qubits
            current_physical_q1 = logical_to_physical_mapping[logical_q1]
            current_physical_q2 = logical_to_physical_mapping[logical_q2]

            # Check if these current physical qubits are connected
            if current_physical_q2 not in adj_matrix[current_physical_q1]:
                print(f"  Connectivity required for logical CX({logical_q1}, {logical_q2}) "
                      f"currently mapped to physical ({current_physical_q1}, {current_physical_q2})")
                
                # Find the shortest path between the CURRENT physical locations
                shortest_path = find_valid_path(adj_matrix, current_physical_q1, current_physical_q2)

                if len(shortest_path) > 1:
                    print(f"  Shortest path found: {shortest_path}")
                    # Insert SWAP gates along the path and update mappings dynamically
                    for i in range(len(shortest_path) - 1):
                        swap_q1, swap_q2 = shortest_path[i], shortest_path[i+1]
                        
                        # IMPORTANT: Update mappings AFTER EACH SWAP
                        update_mappings_for_swap(swap_q1, swap_q2, logical_to_physical_mapping, physical_to_logical_mapping)

                        print(f"  Inserting SWAP({swap_q1}, {swap_q2})")
                        new_circuit.swap(swap_q1, swap_q2)
                else:
                    print(f"  Warning: No valid path found between {current_physical_q1} and {current_physical_q2}. "
                          f"This gate might not be executable.")

            # After any necessary SWAPs are inserted and mappings updated,
            # retrieve the final physical locations for the CX gate.
            final_physical_q1 = logical_to_physical_mapping[logical_q1]
            final_physical_q2 = logical_to_physical_mapping[logical_q2]
            
            # Add the CX gate to the new circuit with its final physical operands
            new_circuit.cx(final_physical_q1, final_physical_q2)
        else:
            # For single-qubit gates or other gates that don't need routing
            current_physical_qargs = [logical_to_physical_mapping[lq] for lq in logical_qargs]
            new_circuit.append(op, current_physical_qargs, clargs)

    return new_circuit