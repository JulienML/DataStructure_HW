"""
Chord DHT (Distributed Hash Table) Implementation

This module implements the Chord protocol for distributed hash tables.
Key features:
- Ring-based structure with 2^n virtual positions
- Finger tables for O(log n) lookup
- Data replication for fault tolerance
- Dynamic node join/leave with data redistribution
"""

import hashlib
from typing import Dict, List, Optional, Any, Tuple


def hash_key(key: Any, ring_size: int) -> int:
    """
    Hash a key to a position on the Chord ring.
    Uses SHA-1 and takes modulo ring_size.
    """
    if isinstance(key, int):
        key = str(key)
    h = hashlib.sha1(str(key).encode()).hexdigest()
    return int(h, 16) % ring_size


class ChordNode:
    """
    Represents a physical server in the Chord ring.
    
    Each node:
    - Has a position on the ring
    - Maintains a finger table for routing
    - Stores data for keys it's responsible for
    - Replicates data to predecessor nodes for fault tolerance
    """
    
    def __init__(self, position: int, ring_bits: int):
        """
        Initialize a Chord node.
        
        Args:
            position: Position of this node on the ring (0 to 2^ring_bits - 1)
            ring_bits: Number of bits in the ring (ring has 2^ring_bits positions)
        """
        self.position = position
        self.ring_bits = ring_bits
        self.ring_size = 2 ** ring_bits
        
        # Finger table: finger[i] points to successor of (position + 2^i) mod ring_size
        # Size is ring_bits entries (indices 0 to ring_bits-1)
        self.finger_table: List[Optional['ChordNode']] = [None] * ring_bits
        
        # Predecessor and successor pointers
        self.predecessor: Optional['ChordNode'] = None
        self.successor: Optional['ChordNode'] = None
        
        # Local data storage: key -> value
        self.data: Dict[Any, Any] = {}
        
        # Replicated data from successor nodes (for fault tolerance)
        self.replicas: Dict[int, Dict[Any, Any]] = {}  # node_position -> data
    
    def __repr__(self):
        return f"ChordNode(S{self.position})"
    
    def is_responsible_for(self, key_position: int) -> bool:
        """
        Check if this node is responsible for the given key position.
        A node is responsible for keys in the range (predecessor.position, self.position].
        """
        if self.predecessor is None:
            return True
        
        pred_pos = self.predecessor.position
        
        # Handle wrap-around in the ring
        if pred_pos < self.position:
            # Normal case: key is between predecessor and self
            return pred_pos < key_position <= self.position
        else:
            # Wrap-around case: ring wraps around 0
            return key_position > pred_pos or key_position <= self.position
    
    def find_successor(self, key_position: int) -> Tuple['ChordNode', List['ChordNode']]:
        """
        Find the node responsible for the given key position.
        Returns the responsible node and the path of hops taken.
        
        This is the core Chord routing algorithm using finger tables.
        """
        path = [self]
        
        # If we're responsible, return self
        if self.is_responsible_for(key_position):
            return self, path
        
        # If the key is between us and our successor, successor is responsible
        if self.successor and self._in_range(key_position, self.position, self.successor.position, inclusive_end=True):
            path.append(self.successor)
            return self.successor, path
        
        # Use finger table to find the closest preceding node
        next_node = self._closest_preceding_node(key_position)
        
        if next_node == self:
            # We are the only node or can't find a better one
            if self.successor:
                path.append(self.successor)
                return self.successor, path
            return self, path
        
        # Recursively find the successor from the next node
        result, sub_path = next_node.find_successor(key_position)
        return result, path + sub_path
    
    def _closest_preceding_node(self, key_position: int) -> 'ChordNode':
        """
        Find the closest preceding node to the key from our finger table.
        Search from the highest finger entry to the lowest.
        """
        for i in range(self.ring_bits - 1, -1, -1):
            finger = self.finger_table[i]
            if finger and self._in_range(finger.position, self.position, key_position, inclusive_end=False):
                return finger
        return self
    
    def _in_range(self, value: int, start: int, end: int, inclusive_end: bool = False) -> bool:
        """
        Check if value is in range (start, end) or (start, end] on a circular ring.
        """
        if start == end:
            return True
        
        if start < end:
            if inclusive_end:
                return start < value <= end
            else:
                return start < value < end
        else:
            # Wrap-around
            if inclusive_end:
                return value > start or value <= end
            else:
                return value > start or value < end
    
    def store(self, key: Any, value: Any, ring: 'ChordRing') -> bool:
        """
        Store a key-value pair. The key is hashed to find the responsible node.
        """
        key_pos = hash_key(key, self.ring_size)
        responsible_node, path = self.find_successor(key_pos)
        responsible_node.data[key] = value
        
        # Replicate to predecessors for fault tolerance
        ring._replicate_to_predecessors(responsible_node, key, value)
        
        return True
    
    def lookup(self, key: Any) -> Tuple[Optional[Any], List['ChordNode']]:
        """
        Look up a key and return its value and the path taken.
        """
        key_pos = hash_key(key, self.ring_size)
        responsible_node, path = self.find_successor(key_pos)
        
        value = responsible_node.data.get(key)
        return value, path
    
    def build_finger_table(self, ring: 'ChordRing'):
        """
        Build the finger table for this node.
        finger[i] = successor of (position + 2^i) mod ring_size
        """
        for i in range(self.ring_bits):
            target = (self.position + (2 ** i)) % self.ring_size
            self.finger_table[i] = ring.find_physical_node(target)
    
    def display_finger_table(self):
        """Display the finger table in a readable format."""
        print(f"\nFinger Table for S{self.position}:")
        print("-" * 40)
        print(f"{'i':<5} {'Start (pos + 2^i)':<20} {'Successor':<10}")
        print("-" * 40)
        for i in range(self.ring_bits):
            start = (self.position + (2 ** i)) % self.ring_size
            finger = self.finger_table[i]
            finger_str = f"S{finger.position}" if finger else "None"
            print(f"{i:<5} {start:<20} {finger_str:<10}")


class ChordRing:
    """
    The Chord ring structure that manages all nodes.
    
    Features:
    - 2^n virtual positions
    - Physical nodes placed at specific positions
    - Automatic finger table updates
    - Data replication for fault tolerance
    """
    
    def __init__(self, bits: int = 4, replication_factor: int = 3):
        """
        Initialize a Chord ring.
        
        Args:
            bits: Number of bits (ring has 2^bits positions)
            replication_factor: Number of predecessor nodes for replication
        """
        self.bits = bits
        self.ring_size = 2 ** bits
        self.replication_factor = replication_factor
        
        # All physical nodes, sorted by position
        self.nodes: List[ChordNode] = []
    
    def add_node(self, position: int) -> ChordNode:
        """
        Add a new physical node at the given position.
        
        Steps:
        1. Create the node
        2. Insert it in the sorted list
        3. Update predecessor/successor pointers
        4. Rebuild finger tables
        5. Transfer data from successor
        """
        if position < 0 or position >= self.ring_size:
            raise ValueError(f"Position must be in range [0, {self.ring_size})")
        
        # Check if position is already taken
        for node in self.nodes:
            if node.position == position:
                raise ValueError(f"Position {position} is already occupied")
        
        # Create the new node
        new_node = ChordNode(position, self.bits)
        
        # Insert in sorted order
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda n: n.position)
        
        # Update all pointers and finger tables
        self._update_all_pointers()
        self._update_all_finger_tables()
        
        # Transfer data from successor if any
        if new_node.successor and new_node.successor != new_node:
            self._transfer_data_on_join(new_node)
        
        return new_node
    
    def remove_node(self, position: int) -> bool:
        """
        Remove a node from the ring.
        
        Steps:
        1. Transfer data to successor
        2. Remove from nodes list
        3. Update all pointers and finger tables
        """
        node_to_remove = None
        for node in self.nodes:
            if node.position == position:
                node_to_remove = node
                break
        
        if not node_to_remove:
            return False
        
        # Transfer data to successor
        if node_to_remove.successor and node_to_remove.successor != node_to_remove:
            node_to_remove.successor.data.update(node_to_remove.data)
        
        # Remove the node
        self.nodes.remove(node_to_remove)
        
        # Update pointers and finger tables
        if self.nodes:
            self._update_all_pointers()
            self._update_all_finger_tables()
        
        return True
    
    def find_physical_node(self, position: int) -> Optional[ChordNode]:
        """
        Find the physical node responsible for a virtual position.
        This is the first node with position >= given position (with wrap-around).
        """
        if not self.nodes:
            return None
        
        # Find the first node with position >= target
        for node in self.nodes:
            if node.position >= position:
                return node
        
        # Wrap around to the first node
        return self.nodes[0]
    
    def _update_all_pointers(self):
        """Update predecessor and successor pointers for all nodes."""
        n = len(self.nodes)
        if n == 0:
            return
        
        for i, node in enumerate(self.nodes):
            node.predecessor = self.nodes[(i - 1) % n]
            node.successor = self.nodes[(i + 1) % n]
    
    def _update_all_finger_tables(self):
        """Rebuild finger tables for all nodes."""
        for node in self.nodes:
            node.build_finger_table(self)
    
    def _transfer_data_on_join(self, new_node: ChordNode):
        """Transfer keys from successor that now belong to the new node."""
        successor = new_node.successor
        if not successor:
            return
        
        keys_to_transfer = []
        for key in successor.data:
            key_pos = hash_key(key, self.ring_size)
            if new_node.is_responsible_for(key_pos):
                keys_to_transfer.append(key)
        
        for key in keys_to_transfer:
            new_node.data[key] = successor.data.pop(key)
    
    def _replicate_to_predecessors(self, node: ChordNode, key: Any, value: Any):
        """Replicate data to predecessor nodes for fault tolerance."""
        current = node.predecessor
        for _ in range(self.replication_factor):
            if current and current != node:
                if node.position not in current.replicas:
                    current.replicas[node.position] = {}
                current.replicas[node.position][key] = value
                current = current.predecessor
            else:
                break
    
    def lookup(self, key: Any, starting_node: Optional[ChordNode] = None) -> Tuple[Optional[Any], List[ChordNode]]:
        """
        Look up a key starting from a given node (or the first node).
        Returns the value and the path of nodes visited.
        """
        if not self.nodes:
            return None, []
        
        start = starting_node if starting_node else self.nodes[0]
        return start.lookup(key)
    
    def store(self, key: Any, value: Any, starting_node: Optional[ChordNode] = None) -> bool:
        """Store a key-value pair in the ring."""
        if not self.nodes:
            return False
        
        start = starting_node if starting_node else self.nodes[0]
        return start.store(key, value, self)
    
    def display_ring(self):
        """Display the ring structure."""
        print(f"\n{'='*50}")
        print(f"Chord Ring (2^{self.bits} = {self.ring_size} positions)")
        print(f"{'='*50}")
        print(f"Physical nodes: {len(self.nodes)}")
        print(f"Positions: {[f'S{n.position}' for n in self.nodes]}")
        print()
        
        for node in self.nodes:
            pred = f"S{node.predecessor.position}" if node.predecessor else "None"
            succ = f"S{node.successor.position}" if node.successor else "None"
            print(f"S{node.position}: predecessor={pred}, successor={succ}, data_keys={list(node.data.keys())}")
    
    def visualize_ring_ascii(self):
        """Create an ASCII visualization of the ring."""
        print(f"\n{'='*50}")
        print("Chord Ring Visualization")
        print(f"{'='*50}")
        
        node_positions = {n.position for n in self.nodes}
        
        for pos in range(self.ring_size):
            if pos in node_positions:
                node = next(n for n in self.nodes if n.position == pos)
                data_count = len(node.data)
                print(f"[S{pos:2d}] ◆ Physical Node (data: {data_count} keys)")
            else:
                print(f" {pos:2d}   ○ (virtual)")


def demo_search_example():
    """
    Demonstrate the search example from the handout:
    Search for value 54 on server S11, where H(54) = 6
    """
    print("\n" + "="*60)
    print("DEMO: Search Example from Handout")
    print("="*60)
    
    # Create a ring with 2^4 = 16 positions
    ring = ChordRing(bits=4)
    
    # Add physical servers at positions 0, 4, 7, 11, 14
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    ring.display_ring()
    
    # Show finger tables
    for node in ring.nodes:
        node.display_finger_table()
    
    # Store some data
    # We want H(54) = 6, so let's store data at position 6
    # The responsible node for position 6 would be S7 (first node >= 6)
    ring.store("key_54", "value_for_54")
    
    # Simulate search from S11
    print("\n" + "-"*40)
    print("Searching from S11...")
    print("-"*40)
    
    s11 = ring.nodes[3]  # S11 is at index 3
    
    # Search for key_54
    value, path = s11.lookup("key_54")
    
    print(f"Starting node: S{s11.position}")
    print(f"Path: {' -> '.join(f'S{n.position}' for n in path)}")
    print(f"Found value: {value}")


def demo_node_operations():
    """
    Demonstrate node join and leave operations.
    """
    print("\n" + "="*60)
    print("DEMO: Node Join and Leave Operations")
    print("="*60)
    
    # Create initial ring
    ring = ChordRing(bits=4)
    
    print("\n1. Creating ring with nodes at positions 0, 4, 7, 11, 14")
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    ring.display_ring()
    
    # Store some data
    print("\n2. Storing data...")
    ring.store("product_1", {"name": "Laptop", "price": 999})
    ring.store("product_2", {"name": "Phone", "price": 499})
    ring.store("client_100", {"name": "Alice", "email": "alice@example.com"})
    
    ring.display_ring()
    
    # Add a new node
    print("\n3. Adding new node at position 2...")
    ring.add_node(2)
    
    ring.display_ring()
    ring.visualize_ring_ascii()
    
    # Remove a node
    print("\n4. Removing node at position 7...")
    ring.remove_node(7)
    
    ring.display_ring()


def demo_replication():
    """
    Demonstrate data replication for fault tolerance.
    """
    print("\n" + "="*60)
    print("DEMO: Data Replication for Fault Tolerance")
    print("="*60)
    
    ring = ChordRing(bits=4, replication_factor=3)
    
    for pos in [0, 4, 7, 11, 14]:
        ring.add_node(pos)
    
    # Store data
    ring.store("important_data", "This data is replicated!")
    
    print("\nData and replicas:")
    for node in ring.nodes:
        print(f"\nS{node.position}:")
        print(f"  Own data: {node.data}")
        print(f"  Replicas: {node.replicas}")


if __name__ == "__main__":
    # Run all demos
    demo_search_example()
    demo_node_operations()
    demo_replication()
