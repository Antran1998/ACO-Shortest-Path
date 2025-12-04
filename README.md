# ACO_path_planning


## Enhancements (feat/aco-base)

This branch implements two main improvements based on standard ACO research papers:

### 1. Heuristic Function

**Formula:**
```
η_ij = 1 / (d(i, goal) + ε)
```

**Implementation:**
- Added `heuristic()` method in `aco/ant_colony.py` (line ~92-95)
- Uses Euclidean distance from current node to goal node
- ε (EPSILON = 1e-6) prevents division by zero
- Used in `edge_weight()` to calculate: (τ^α) × (η^β)

### 2. Distance-based Pheromone Update

**Formula:**
```
τ_ij(t+1) = (1-ρ) × τ_ij(t) + Δτ_ij

Where:
Δτ_ij = Q / L_k    if edge (i,j) ∈ path_k
Δτ_ij = 0          otherwise
```

**Implementation:**
- `L_k` changed from **number of steps** to **total Euclidean distance** of the path
- Added `calculate_path_distance()` method (line ~82-90) to compute total path distance
- Updated `pheromone_update()` method (line ~123-136) to use Euclidean distance

**Purposw:** More accurate cost representation - diagonal moves cost √2 instead of 1, properly reflecting real distance

### Supporting Changes

**Euclidean Distance Helper:**
- Added `calculate_euclidean_distance(pos1, pos2)` method (line ~79-83)
- Calculates: `d = √[(x1-x2)² + (y1-y2)²]`
- Reused by both heuristic and path distance calculations

