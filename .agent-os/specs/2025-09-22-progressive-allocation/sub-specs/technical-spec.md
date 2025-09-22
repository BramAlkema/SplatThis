# Progressive Allocation - Technical Specification

## API Design

### Core Classes

#### ProgressiveConfig
```python
@dataclass
class ProgressiveConfig:
    initial_ratio: float = 0.3              # Start with 30% of target splats
    max_splats: int = 2000                  # Maximum total splats allowed
    add_interval: int = 50                  # Add splats every N iterations
    max_add_per_step: int = 20              # Limit splats added per step
    error_threshold: float = 0.01           # Minimum error for new placement
    convergence_patience: int = 5           # Steps without improvement to stop
    temperature: float = 2.0                # Sampling temperature for error distribution

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0.1 <= self.initial_ratio <= 0.8, "Initial ratio must be 0.1-0.8"
        assert self.max_splats > 0, "Max splats must be positive"
        assert self.add_interval > 0, "Add interval must be positive"
        assert self.temperature > 0, "Temperature must be positive"
```

#### ProgressiveAllocator
```python
class ProgressiveAllocator:
    def __init__(self, config: ProgressiveConfig):
        self.config = config
        self.iteration_count = 0
        self.error_history: List[float] = []
        self.last_addition_iteration = -1

    def should_add_splats(self, current_error: float) -> bool:
        """
        Determine if new splats should be added based on:
        - Current iteration vs. add_interval
        - Error threshold
        - Convergence detection
        """
        # Check if enough iterations have passed
        if self.iteration_count - self.last_addition_iteration < self.config.add_interval:
            return False

        # Check if error is above threshold
        if current_error < self.config.error_threshold:
            return False

        # Check for convergence (error not improving)
        if len(self.error_history) >= self.config.convergence_patience:
            recent_errors = self.error_history[-self.config.convergence_patience:]
            if max(recent_errors) - min(recent_errors) < self.config.error_threshold * 0.1:
                return False  # Converged

        return True

    def get_addition_count(self, current_splat_count: int) -> int:
        """Calculate how many splats to add this step."""
        remaining_budget = self.config.max_splats - current_splat_count
        return min(self.config.max_add_per_step, remaining_budget)

    def record_iteration(self, error: float, added_splats: int = 0) -> None:
        """Record iteration results."""
        self.iteration_count += 1
        self.error_history.append(error)
        if added_splats > 0:
            self.last_addition_iteration = self.iteration_count
```

#### ErrorGuidedPlacement
```python
class ErrorGuidedPlacement:
    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def compute_reconstruction_error(self, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel reconstruction error.

        Args:
            target: Original image (H, W, C)
            rendered: Rendered splat image (H, W, C)

        Returns:
            Error map (H, W) with per-pixel L1 distance
        """
        # Convert to float if needed
        target_f = target.astype(np.float32) / 255.0 if target.dtype == np.uint8 else target
        rendered_f = rendered.astype(np.float32) / 255.0 if rendered.dtype == np.uint8 else rendered

        # Compute L1 error per pixel
        if len(target_f.shape) == 3:
            error_map = np.mean(np.abs(target_f - rendered_f), axis=2)
        else:
            error_map = np.abs(target_f - rendered_f)

        return error_map

    def create_placement_probability(self, error_map: np.ndarray) -> np.ndarray:
        """
        Convert error map to placement probability distribution.

        Args:
            error_map: Per-pixel error (H, W)

        Returns:
            Probability map (H, W) normalized to sum to 1
        """
        # Avoid division by zero
        error_safe = error_map + 1e-8

        # Apply temperature for sampling control
        prob_map = error_safe ** (1.0 / self.temperature)

        # Normalize to probability distribution
        prob_map = prob_map / np.sum(prob_map)

        return prob_map

    def sample_positions(self, prob_map: np.ndarray, count: int) -> List[Tuple[int, int]]:
        """
        Sample new splat positions from probability distribution.

        Args:
            prob_map: Probability distribution (H, W)
            count: Number of positions to sample

        Returns:
            List of (y, x) positions
        """
        if count <= 0:
            return []

        height, width = prob_map.shape
        flat_probs = prob_map.flatten()

        # Sample indices without replacement
        sampled_indices = np.random.choice(
            len(flat_probs),
            size=min(count, len(flat_probs)),
            replace=False,
            p=flat_probs
        )

        # Convert back to 2D coordinates
        positions = []
        for idx in sampled_indices:
            y, x = divmod(idx, width)
            positions.append((y, x))

        return positions
```

### Integration Points

#### Modified AdaptiveSplatExtractor
```python
class AdaptiveSplatExtractor:
    def __init__(self, config: Optional[AdaptiveSplatConfig] = None):
        self.config = config or AdaptiveSplatConfig()
        # Initialize progressive components if enabled
        if self.config.enable_progressive:
            self.progressive_config = self.config.progressive_config or ProgressiveConfig()
            self.allocator = ProgressiveAllocator(self.progressive_config)
            self.placer = ErrorGuidedPlacement(self.progressive_config.temperature)

    def extract_adaptive_splats(self, image: np.ndarray, n_splats: int = 1500, verbose: bool = False) -> List[Gaussian]:
        """
        Main extraction method with optional progressive allocation.
        """
        if self.config.enable_progressive:
            # Override n_splats with progressive config
            return self.extract_progressive_splats(image, verbose)
        else:
            # Use existing static allocation
            return self._extract_static_splats(image, n_splats, verbose)

    def extract_progressive_splats(self, image: np.ndarray, verbose: bool = False) -> List[Gaussian]:
        """
        Progressive splat extraction with error-guided allocation.
        """
        if verbose:
            print(f"Starting progressive allocation, target: {self.progressive_config.max_splats} splats")

        # Phase 1: Initial allocation
        splats = self._initial_allocation(image, verbose)

        # Phase 2: Progressive addition loop
        splats = self._progressive_addition_loop(image, splats, verbose)

        if verbose:
            print(f"Progressive allocation complete: {len(splats)} splats")

        return splats
```

## File Structure

### New Files to Create
```
src/splat_this/core/
├── progressive_allocator.py        # ProgressiveAllocator and ProgressiveConfig
├── error_guided_placement.py       # ErrorGuidedPlacement class
└── reconstruction_error.py         # Error computation utilities

src/splat_this/utils/
└── sampling.py                     # Probability sampling utilities

tests/unit/core/
├── test_progressive_allocator.py
├── test_error_guided_placement.py
└── test_reconstruction_error.py
```

### Modified Files
```
src/splat_this/core/
├── adaptive_extract.py             # Add progressive integration
└── extract.py                      # Update AdaptiveSplatConfig

src/splat_this/cli/
└── main.py                         # Add progressive CLI parameters
```

## Algorithm Implementation

### Initial Allocation Algorithm
```python
def _initial_allocation(self, image: np.ndarray, verbose: bool) -> List[Gaussian]:
    """Phase 1: Initial allocation based on saliency."""
    # Compute saliency map for initial placement
    saliency_map = self.saliency_analyzer.compute_saliency_map(image)

    # Calculate initial splat count
    initial_count = int(self.progressive_config.max_splats * self.progressive_config.initial_ratio)

    if verbose:
        print(f"Initial allocation: {initial_count} splats")

    # Sample positions from saliency distribution
    prob_map = saliency_map / np.sum(saliency_map)
    positions = self.placer.sample_positions(prob_map, initial_count)

    # Create splats at sampled positions
    splats = []
    for y, x in positions:
        splat = self._create_adaptive_splat_at_position(image, (y, x), saliency_map)
        if splat:
            splats.append(splat)

    return splats
```

### Progressive Addition Algorithm
```python
def _progressive_addition_loop(self, image: np.ndarray, splats: List[Gaussian], verbose: bool) -> List[Gaussian]:
    """Phase 2: Progressive addition based on reconstruction error."""
    max_iterations = 1000  # Safety limit

    for iteration in range(max_iterations):
        # Render current splats (simplified for spec)
        rendered = self._render_splats_simple(splats, image.shape[:2])

        # Compute reconstruction error
        error_map = self.placer.compute_reconstruction_error(image, rendered)
        mean_error = np.mean(error_map)

        # Record iteration results
        self.allocator.record_iteration(mean_error)

        # Check if we should add more splats
        if not self.allocator.should_add_splats(mean_error):
            if verbose:
                print(f"Stopping at iteration {iteration}: error converged or threshold reached")
            break

        # Determine how many splats to add
        add_count = self.allocator.get_addition_count(len(splats))

        if add_count > 0:
            # Create probability map from error
            prob_map = self.placer.create_placement_probability(error_map)

            # Sample new positions
            new_positions = self.placer.sample_positions(prob_map, add_count)

            # Create new splats
            new_splats = []
            for y, x in new_positions:
                splat = self._create_adaptive_splat_at_position(image, (y, x), error_map)
                if splat:
                    new_splats.append(splat)

            splats.extend(new_splats)
            self.allocator.record_iteration(mean_error, len(new_splats))

            if verbose:
                print(f"Iteration {iteration}: Added {len(new_splats)} splats, total: {len(splats)}, error: {mean_error:.4f}")

    return splats
```

## Error Metrics

### Reconstruction Error Types
```python
class ReconstructionError:
    @staticmethod
    def l1_error(target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """L1 (Manhattan) distance per pixel."""
        return np.mean(np.abs(target - rendered), axis=2)

    @staticmethod
    def l2_error(target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """L2 (Euclidean) distance per pixel."""
        return np.sqrt(np.mean((target - rendered) ** 2, axis=2))

    @staticmethod
    def perceptual_error(target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
        """Future: SSIM or LPIPS-based perceptual error."""
        # Placeholder for future implementation
        return ReconstructionError.l1_error(target, rendered)
```

## Performance Considerations

### Memory Management
- Error maps are (H, W) single-channel, much smaller than RGB images
- Probability maps can reuse error map memory
- Splat lists grow incrementally, reducing peak memory

### Computational Complexity
- Error computation: O(H × W × C) per iteration
- Position sampling: O(H × W) + O(K log K) for K samples
- Splat creation: O(K) per addition step
- Total: O(I × H × W) for I iterations

### Optimization Opportunities
- Cache rendered output between iterations if no splats change
- Use approximate rendering for error computation (lower resolution)
- Batch splat creation for efficiency
- Early termination based on convergence criteria

## Configuration Examples

### Default Configuration
```python
default_config = ProgressiveConfig(
    initial_ratio=0.3,
    max_splats=2000,
    add_interval=50,
    max_add_per_step=20,
    error_threshold=0.01,
    convergence_patience=5,
    temperature=2.0
)
```

### High-Quality Configuration
```python
high_quality_config = ProgressiveConfig(
    initial_ratio=0.2,        # Start more sparse
    max_splats=4000,          # Allow more splats
    add_interval=25,          # Add more frequently
    max_add_per_step=15,      # Smaller steps
    error_threshold=0.005,    # Lower threshold
    convergence_patience=10,  # More patience
    temperature=1.5           # Sharper sampling
)
```

### Fast Configuration
```python
fast_config = ProgressiveConfig(
    initial_ratio=0.5,        # Start with more splats
    max_splats=1000,          # Lower budget
    add_interval=100,         # Less frequent addition
    max_add_per_step=50,      # Larger steps
    error_threshold=0.02,     # Higher threshold
    convergence_patience=3,   # Less patience
    temperature=3.0           # Softer sampling
)
```