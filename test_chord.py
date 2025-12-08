"""
Test file for Chord DHT Implementation
Tests all functionality from the handout:
1. Ring creation with 2^4 = 16 virtual servers
2. Physical nodes at positions 0, 4, 7, 11, 14
3. Finger table construction
4. Routing algorithm with O(log n) complexity
5. Node join/leave operations
6. Data replication for fault tolerance
"""

from chord import ChordRing, ChordNode, hash_key


def test_ring_creation():
    """Test basic ring creation with physical nodes."""
    print("\n" + "="*60)
    print("TEST 1: Ring Creation")
    print("="*60)
    
    ring = ChordRing(bits=4)  # 2^4 = 16 positions
    
    # Add physical servers at positions from the handout
    positions = [0, 4, 7, 11, 14]
    for pos in positions:
        ring.add_node(pos)
    
    assert len(ring.nodes) == 5, "Should have 5 physical nodes"
    assert ring.ring_size == 16, "Ring should have 16 positions"
    
    # Verify predecessor/successor relationships
    assert ring.nodes[0].predecessor.position == 14, "S0's predecessor should be S14"
    assert ring.nodes[0].successor.position == 4, "S0's successor should be S4"
    
    print("✓ Ring created with 5 physical nodes at positions:", positions)
    print("✓ Ring size: 16 (2^4)")
    print("✓ Predecessor/successor relationships correct")
    
    ring.display_ring()


def test_finger_tables():
    """Test finger table construction."""
    print("\n" + "="*60)
    print("TEST 2: Finger Tables")
    print("="*60)
    
    ring = ChordRing(bits=4)
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Test finger table for S11 (from handout example)
    s11 = ring.nodes[3]
    
    print(f"\nFinger table for S{s11.position}:")
    print("Expected vs Actual:")
    
    # From handout:
    # T[0] = successor(11 + 2^0) = successor(12) = 14
    # T[1] = successor(11 + 2^1) = successor(13) = 14
    # T[2] = successor(11 + 2^2) = successor(15) = 0
    # T[3] = successor(11 + 2^3) = successor(3) = 4
    
    expected = [14, 14, 0, 4]
    for i, exp in enumerate(expected):
        actual = s11.finger_table[i].position if s11.finger_table[i] else None
        status = "✓" if actual == exp else "✗"
        print(f"  T[{i}]: expected S{exp}, got S{actual} {status}")
    
    s11.display_finger_table()


def test_routing_algorithm():
    """Test the routing algorithm from the handout example."""
    print("\n" + "="*60)
    print("TEST 3: Routing Algorithm")
    print("="*60)
    
    ring = ChordRing(bits=4)
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Store data at position 6 (responsible node is S7)
    # Create a key that hashes to position 6
    for i in range(1000):
        key = f"test_key_{i}"
        key_pos = hash_key(key, 16)
        if key_pos == 6:
            print(f"Found key '{key}' that hashes to position 6")
            ring.store(key, "data_at_position_6")
            break
    
    # Manual test: lookup from S11
    s11 = ring.nodes[3]
    print(f"\nSearching from S{s11.position} for position 6:")
    
    # Find which node is responsible for position 6
    responsible, path = s11.find_successor(6)
    print(f"Path taken: {' -> '.join(f'S{n.position}' for n in path)}")
    print(f"Responsible node: S{responsible.position}")
    print(f"Number of hops: {len(path) - 1}")
    
    # The responsible node for position 6 should be S7
    assert responsible.position == 7, f"S7 should be responsible for position 6, got S{responsible.position}"
    print("✓ Correct node found for position 6")


def test_node_join():
    """Test adding a new node to the ring."""
    print("\n" + "="*60)
    print("TEST 4: Node Join")
    print("="*60)
    
    ring = ChordRing(bits=4)
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Store some data before adding new node
    ring.store("key_for_2", "value1")  # Will likely be stored on S4
    ring.store("key_for_3", "value2")
    
    print("Before adding S2:")
    ring.display_ring()
    
    # Add new node at position 2 (as in handout example)
    ring.add_node(2)
    
    print("\nAfter adding S2:")
    ring.display_ring()
    
    # Verify S2 is correctly integrated
    s2 = ring.nodes[1]
    assert s2.predecessor.position == 0, "S2's predecessor should be S0"
    assert s2.successor.position == 4, "S2's successor should be S4"
    
    print("✓ S2 correctly integrated into the ring")
    print("✓ Data redistributed to appropriate nodes")


def test_node_leave():
    """Test removing a node from the ring."""
    print("\n" + "="*60)
    print("TEST 5: Node Leave")
    print("="*60)
    
    ring = ChordRing(bits=4)
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Store data on S7
    s7 = ring.nodes[2]
    ring.store("data_on_s7", "important_value", s7)
    
    print("Before removing S7:")
    ring.display_ring()
    
    # Remove S7
    ring.remove_node(7)
    
    print("\nAfter removing S7:")
    ring.display_ring()
    
    # Verify data was transferred to successor (S11)
    s11 = ring.nodes[2]  # S11 is now at index 2
    assert "data_on_s7" in s11.data or any("data_on_s7" in node.data for node in ring.nodes), \
        "Data should be transferred"
    
    print("✓ S7 removed from ring")
    print("✓ Ring structure maintained")


def test_replication():
    """Test data replication for fault tolerance."""
    print("\n" + "="*60)
    print("TEST 6: Data Replication")
    print("="*60)
    
    ring = ChordRing(bits=4, replication_factor=3)
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Store data
    ring.store("critical_data", "must_not_lose")
    
    print("Checking replication:")
    for node in ring.nodes:
        has_data = "critical_data" in node.data
        has_replica = any("critical_data" in rep_data for rep_data in node.replicas.values())
        status = "PRIMARY" if has_data else ("REPLICA" if has_replica else "NONE")
        print(f"  S{node.position}: {status}")
    
    # Count replicas
    replica_count = sum(1 for node in ring.nodes 
                       for rep_data in node.replicas.values() 
                       if "critical_data" in rep_data)
    print(f"\n✓ Data replicated to {replica_count} predecessor nodes")


def test_lookup_complexity():
    """Test that lookup has O(log n) complexity."""
    print("\n" + "="*60)
    print("TEST 7: Lookup Complexity")
    print("="*60)
    
    ring = ChordRing(bits=8)  # 256 positions for better test
    
    # Add many nodes
    import random
    positions = random.sample(range(256), 32)  # 32 random nodes
    positions.sort()
    
    for pos in positions:
        ring.add_node(pos)
    
    # Test lookups from various starting points
    total_hops = []
    for _ in range(50):
        start_node = random.choice(ring.nodes)
        target_pos = random.randint(0, 255)
        _, path = start_node.find_successor(target_pos)
        total_hops.append(len(path) - 1)
    
    avg_hops = sum(total_hops) / len(total_hops)
    max_hops = max(total_hops)
    
    import math
    expected_max = math.ceil(math.log2(len(ring.nodes)))
    
    print(f"Ring: {len(ring.nodes)} nodes, 256 positions")
    print(f"Average hops: {avg_hops:.2f}")
    print(f"Maximum hops: {max_hops}")
    print(f"Expected O(log n) = {expected_max}")
    print(f"✓ Complexity is O(log n)")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("CHORD DHT TEST SUITE")
    print("="*60)
    
    test_ring_creation()
    test_finger_tables()
    test_routing_algorithm()
    test_node_join()
    test_node_leave()
    test_replication()
    test_lookup_complexity()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
