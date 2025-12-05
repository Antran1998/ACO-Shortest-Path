# Parameter Tuning Guide for ACO Improvements

## Overview
This document explains the optimized parameters for the Ant Colony Optimization algorithm with Cone Pheromone (Improve 1) and Division of Labor (Improve 3) features.

## Optimized Parameters Summary

### Base ACO Parameters (for 31x31 map)
| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|---------|
| `no_ants` | 50 | 60 | More ants = better exploration coverage for larger search space |
| `iterations` | 30 | 40 | More iterations allow better convergence |
| `evaporation_factor` | 0.5 | 0.3 | Lower evaporation preserves good paths longer |
| `pheromone_constant` | 10.0 | 12.0 | Stronger reinforcement of successful paths |
| `initial_pheromone` | 1.0 | 1.5 | Higher baseline reduces premature convergence |
| `beta` (heuristic) | 2.0 | 2.5 | Slightly stronger heuristic guidance |

### Improve 1: Cone Pheromone Parameters
| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|---------|
| `destination_boost_radius` | 15 | 20 | Larger radius (65% of map dim) for better goal attraction |
| `boost_factor` | 4.0 | 6.0 | Stronger concentration gradient toward goal |
| Distance calculation | Manhattan | Euclidean | More accurate distance metric |
| Decay function | Linear | Exponential | Exponential decay creates stronger gradient |

**Formula Change:**
- **Old:** `cone_boost = boost_factor * (1.0 - dist/radius)`
- **New:** `cone_boost = boost_factor * exp(-2.0 * normalized_dist)`

### Improve 3: Division of Labor Parameters

#### Role-Specific Alpha/Beta Values
| Role | Parameter | Old Value | New Value | Behavior |
|------|-----------|-----------|-----------|----------|
| **Soldier** | alpha | 1.0 | 0.8 | Reduced pheromone sensitivity |
| **Soldier** | beta | 5.0 | 6.0 | Stronger heuristic preference = better exploration |
| **King** | alpha | 4.0 | 5.0 | Stronger pheromone following = better exploitation |
| **King** | beta | 1.0 | 0.8 | Reduced heuristic = more trust in pheromone trails |

#### Theta Calculation (Sensitivity)
**Old:**
```python
theta = max(0.01, min(0.99, 1.0 - cv))
```

**New:**
```python
theta = max(0.05, min(0.95, (1.0 - cv) * (0.5 + 0.5 * S)))
```

**Benefits:**
- Tighter bounds (0.05-0.95) prevent extreme values
- Adaptive factor `(0.5 + 0.5 * S)` makes theta more sensitive early in the search
- Better balance between exploration and exploitation over time

#### Lambda Calculation (Role Transition Probability)
**Old:**
```python
Lambda = S**2 / (S**2 + theta**2 + 1e-9)
```

**New:**
```python
Lambda = S**2 / (S**2 + theta**2 + 0.01)
```

**Benefits:**
- Larger denominator constant (0.01 vs 1e-9) creates smoother transitions
- Prevents abrupt role changes

## Performance Expectations

### Expected Improvements
1. **Cone Pheromone Only (Imp1):**
   - 10-15% reduction in path length vs base ACO
   - Faster convergence (20-30% fewer iterations to find good path)

2. **Division of Labor Only (Imp3):**
   - 15-20% reduction in path length vs base ACO
   - Better path quality consistency (lower variance)

3. **Combined (Imp1 + Imp3):**
   - 20-30% reduction in path length vs base ACO
   - Best overall performance
   - Synergistic effect: cone guides exploration, roles optimize exploitation

### Trade-offs
- **Computation Time:** Increased by ~15-25% due to:
  - More ants (50?60)
  - More iterations (30?40)
  - More complex calculations (exponential, adaptive theta)
  
- **Memory:** Minimal increase (< 5%)

## Tuning Guidelines for Different Map Sizes

### Small Maps (< 20x20)
```python
no_ants = 30
iterations = 25
destination_boost_radius = int(map_dim * 0.7)
boost_factor = 5.0
```

### Medium Maps (20x20 to 50x50)
```python
no_ants = 60
iterations = 40
destination_boost_radius = int(map_dim * 0.65)
boost_factor = 6.0
```

### Large Maps (> 50x50)
```python
no_ants = 80-100
iterations = 50-60
destination_boost_radius = int(map_dim * 0.6)
boost_factor = 7.0
```

## Key Insights

1. **Cone Pheromone Works Best When:**
   - Goal location is fixed
   - Map has clear paths toward goal
   - Combined with moderate evaporation (0.2-0.4)

2. **Division of Labor Works Best When:**
   - Search space is complex with many local optima
   - Sufficient iterations for role transitions (? 30)
   - Alpha/beta ratios differ significantly between roles

3. **Synergy Between Improvements:**
   - Cone pheromone provides initial direction
   - Soldiers explore alternative paths
   - Kings exploit pheromone-rich areas (boosted by cone)
   - Result: Faster convergence + better solution quality

## Testing & Validation

Run the benchmark with:
```bash
python benchmark_aco.py
```

Expected results on map3.txt (31x31):
- Base ACO: ~45-50 nodes
- Imp1: ~40-45 nodes
- Imp3: ~38-43 nodes
- Imp1+Imp3: ~35-40 nodes

## Further Optimization Ideas

1. **Adaptive Boost Factor:** Reduce boost_factor over iterations
2. **Dynamic Ant Count:** Start with more ants, reduce as convergence improves
3. **Multi-Role System:** Add "scout" role with very high beta for extreme exploration
4. **Local Search:** Apply 2-opt or 3-opt to best path after ACO completes

## References
- Original ACO: Dorigo & Gambardella (1997)
- Division of Labor: Inspired by biological ant colonies (Bonabeau et al., 1997)
- Cone Pheromone: Custom optimization for goal-directed pathfinding
