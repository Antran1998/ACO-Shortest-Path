# ACO_path_planning


## Enhancements (feat/aco-base)

This branch implements two main improvements based on standard ACO research papers:

### 1. Heuristic Function

**Formula:**

$$\eta_{ij} = \frac{1}{d(i, j) + \epsilon}$$

**Implementation:**
- Added `heuristic()` method in `aco/ant_colony.py` (line ~92-95)
- Uses Euclidean distance from current node to goal node
- ε (EPSILON = 1e-6) prevents division by zero
- Used in `edge_weight()` to calculate: $(\tau^\alpha) \times (\eta^\beta)$

**Purposw:** Guides ants toward the goal by favoring nodes closer to the destination

### 2. Distance-based Pheromone Update

**Formula:**

$$\tau_{ij}(t+1) = (1-\rho) \times \tau_{ij}(t) + \Delta\tau_{ij}$$

Where:

$$\Delta\tau_{ij} = \begin{cases} 
\dfrac{Q}{L_k} & (i,j) \in \text{path}_k \\
0 & \text{otherwise} 
\end{cases}$$

**Implementation:**
- `L_k` changed from **number of steps** to **total Euclidean distance** of the path
- Added `calculate_path_distance()` method (line ~82-90) to compute total path distance
- Updated `pheromone_update()` method (line ~123-136) to use Euclidean distance

**Advantage:** More accurate cost representation - diagonal moves cost $\sqrt{2}$ instead of 1, properly reflecting real distance

### Supporting Changes

**Euclidean Distance Helper:**
- Added `calculate_euclidean_distance(pos1, pos2)` method (line ~79-83)
- Calculates: $d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$
- Reused by both heuristic and path distance calculations

