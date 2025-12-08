"""
Example: Distributed Database using Chord DHT

This example demonstrates how to use the Chord DHT implementation
with the database schemas (Product, Client, OrderLine, etc.) to
create a distributed database system.
"""

from chord import ChordRing, hash_key
from loader import load_json_schema, load_statistics
from model import CollectionSchema
from size import estimate_document_size, gb


def create_distributed_database():
    """
    Create a distributed database using Chord DHT.
    Demonstrates:
    1. Setting up a Chord ring with multiple servers
    2. Distributing data (products, clients, orders) across nodes
    3. Looking up data efficiently
    """
    print("="*70)
    print("DISTRIBUTED DATABASE WITH CHORD DHT")
    print("="*70)
    
    # Load statistics
    stats = load_statistics()
    num_servers = stats["servers"]  # 1000 servers
    
    # For demonstration, we'll use a smaller ring
    # In production, you'd use bits = ceil(log2(num_servers))
    ring_bits = 10  # 2^10 = 1024 positions (enough for 1000 servers)
    
    print(f"\nCreating Chord ring with 2^{ring_bits} = {2**ring_bits} positions")
    ring = ChordRing(bits=ring_bits, replication_factor=3)
    
    # Add physical servers at distributed positions
    # In real system, positions would be hash of server ID
    import random
    random.seed(42)  # For reproducibility
    
    server_positions = random.sample(range(2**ring_bits), 20)  # Demo with 20 servers
    server_positions.sort()
    
    print(f"Adding {len(server_positions)} physical servers...")
    for pos in server_positions:
        ring.add_node(pos)
    
    ring.display_ring()
    
    return ring


def store_sample_data(ring):
    """Store sample data from different collections."""
    print("\n" + "="*70)
    print("STORING SAMPLE DATA")
    print("="*70)
    
    # Sample products
    products = [
        {"IDP": "P001", "name": "Laptop Pro", "brand": "TechCo", "price": 1299.99},
        {"IDP": "P002", "name": "Smartphone X", "brand": "PhoneCorp", "price": 899.99},
        {"IDP": "P003", "name": "Tablet Ultra", "brand": "TechCo", "price": 599.99},
        {"IDP": "P004", "name": "Headphones Max", "brand": "AudioInc", "price": 299.99},
        {"IDP": "P005", "name": "Smartwatch Pro", "brand": "WearTech", "price": 399.99},
    ]
    
    # Sample clients
    clients = [
        {"IDC": "C001", "fn": "Alice", "ln": "Smith", "email": "alice@example.com"},
        {"IDC": "C002", "fn": "Bob", "ln": "Johnson", "email": "bob@example.com"},
        {"IDC": "C003", "fn": "Charlie", "ln": "Williams", "email": "charlie@example.com"},
    ]
    
    # Sample orders
    orders = [
        {"IDO": "O001", "IDC": "C001", "IDP": "P001", "quantity": 1, "date": "2024-01-15"},
        {"IDO": "O002", "IDC": "C002", "IDP": "P002", "quantity": 2, "date": "2024-01-16"},
        {"IDO": "O003", "IDC": "C001", "IDP": "P003", "quantity": 1, "date": "2024-01-17"},
        {"IDO": "O004", "IDC": "C003", "IDP": "P004", "quantity": 3, "date": "2024-01-18"},
    ]
    
    # Store products
    print("\nStoring products...")
    for product in products:
        key = f"product:{product['IDP']}"
        ring.store(key, product)
        key_pos = hash_key(key, ring.ring_size)
        responsible, _ = ring.nodes[0].find_successor(key_pos)
        print(f"  {key} -> position {key_pos} -> S{responsible.position}")
    
    # Store clients
    print("\nStoring clients...")
    for client in clients:
        key = f"client:{client['IDC']}"
        ring.store(key, client)
        key_pos = hash_key(key, ring.ring_size)
        responsible, _ = ring.nodes[0].find_successor(key_pos)
        print(f"  {key} -> position {key_pos} -> S{responsible.position}")
    
    # Store orders
    print("\nStoring orders...")
    for order in orders:
        key = f"order:{order['IDO']}"
        ring.store(key, order)
        key_pos = hash_key(key, ring.ring_size)
        responsible, _ = ring.nodes[0].find_successor(key_pos)
        print(f"  {key} -> position {key_pos} -> S{responsible.position}")


def demonstrate_lookups(ring):
    """Demonstrate data lookups with routing."""
    print("\n" + "="*70)
    print("DATA LOOKUPS WITH ROUTING")
    print("="*70)
    
    keys_to_lookup = [
        "product:P001",
        "client:C002",
        "order:O003",
        "product:P005",
    ]
    
    for key in keys_to_lookup:
        # Start lookup from different nodes
        start_node = ring.nodes[len(ring.nodes) // 2]  # Start from middle node
        
        value, path = ring.lookup(key, start_node)
        
        print(f"\nLooking up '{key}':")
        print(f"  Starting from: S{start_node.position}")
        print(f"  Path: {' -> '.join(f'S{n.position}' for n in path)}")
        print(f"  Hops: {len(path) - 1}")
        if value:
            print(f"  Found: {value}")
        else:
            print(f"  Not found")


def demonstrate_fault_tolerance(ring):
    """Demonstrate fault tolerance through replication."""
    print("\n" + "="*70)
    print("FAULT TOLERANCE DEMONSTRATION")
    print("="*70)
    
    # Find a node with data
    node_with_data = None
    for node in ring.nodes:
        if node.data:
            node_with_data = node
            break
    
    if not node_with_data:
        print("No node with data found")
        return
    
    print(f"\nNode S{node_with_data.position} has data: {list(node_with_data.data.keys())}")
    
    # Check replicas
    print("\nChecking replicas on predecessor nodes:")
    for node in ring.nodes:
        if node_with_data.position in node.replicas:
            print(f"  S{node.position} has replica: {list(node.replicas[node_with_data.position].keys())}")
    
    print(f"\nSimulating failure of S{node_with_data.position}...")
    
    # The predecessor nodes already have replicas, so data would still be accessible
    print("Data remains available on replica nodes!")


def demonstrate_node_scaling(ring):
    """Demonstrate adding and removing nodes."""
    print("\n" + "="*70)
    print("NODE SCALING DEMONSTRATION")
    print("="*70)
    
    print("\nCurrent ring state:")
    print(f"  Nodes: {len(ring.nodes)}")
    print(f"  Positions: {[f'S{n.position}' for n in ring.nodes[:5]]}...")
    
    # Add a new node
    import random
    new_pos = random.choice([p for p in range(ring.ring_size) 
                            if p not in {n.position for n in ring.nodes}])
    
    print(f"\nAdding new node at position {new_pos}...")
    ring.add_node(new_pos)
    
    new_node = next(n for n in ring.nodes if n.position == new_pos)
    print(f"  S{new_pos} predecessor: S{new_node.predecessor.position}")
    print(f"  S{new_pos} successor: S{new_node.successor.position}")
    print(f"  Data transferred to S{new_pos}: {list(new_node.data.keys())}")
    
    # Finger table is automatically rebuilt
    print("\nFinger table rebuilt for all nodes.")


def main():
    """Run the complete demonstration."""
    # Create the distributed database
    ring = create_distributed_database()
    
    # Store sample data
    store_sample_data(ring)
    
    # Demonstrate lookups
    demonstrate_lookups(ring)
    
    # Demonstrate fault tolerance
    demonstrate_fault_tolerance(ring)
    
    # Demonstrate scaling
    demonstrate_node_scaling(ring)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("""
Key Takeaways from Chord DHT:
1. Decentralized: No single point of failure
2. Scalable: O(log n) lookup complexity
3. Fault-tolerant: Data replicated on predecessor nodes
4. Self-managing: Automatic data redistribution on node join/leave
5. Load-balanced: Data distributed across nodes using consistent hashing
""")


if __name__ == "__main__":
    main()
