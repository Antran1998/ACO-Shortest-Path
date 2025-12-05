# ACO_path_planning


## Adaptive Heuristic Factors Enhancement (feat/aco-improve2-adaptive-factors)

Based on feat/aco-base, this branch adds adaptive adjustment of PHF (Pheromone Heuristic Factor) and EHF (Expected Heuristic Factor) to improve convergence balance.

### Adaptive PHF & EHF

**Solution:** Dynamic adaptation of heuristic factors based on iteration progress.

**Formula:**

$$\alpha'(n) = \alpha + \xi \int_0^{n/N} t \, dt = \alpha + \xi \left(\frac{n}{N}\right)^2 / 2 \quad \text{(Adaptive PHF)}$$

$$\beta'(n) = \beta + \xi \int_0^{n/N} t \, dt = \beta + \xi \left(\frac{n}{N}\right)^2 / 2 \quad \text{(Adaptive EHF)}$$

Where:
- $n$: Current iteration (0 to N-1)
- $N$: Total iterations
- $\xi$: Adaptive coefficient controlling adaptation speed

**Implementation Details:**
- `calculate_adaptive_factors(current_iteration)`: Computes α' and β' each iteration
- Always uses adaptive factors in `edge_weight()` and `select_next_node()`
- Parameter `xi` in `aco_resolve_path.py` controls sensitivity

### Updated Methods

**`edge_weight(edge, alpha_adaptive, beta_adaptive)`**
- Requires adaptive factors as parameters
- Computes: $(\tau^{\alpha'}) \times (\eta^{\beta'})$

**`select_next_node(node, alpha_adaptive, beta_adaptive)`**
- Uses current iteration's adaptive factors
- Probabilistic selection with normalized weights

### Configuration

```python
# In aco_resolve_path.py
xi = 0.01  # Adaptive coefficient (default)
           # ↑ Increase for faster adaptation from exploration to exploitation
           # ↓ Decrease for slower, more gradual adaptation
```


