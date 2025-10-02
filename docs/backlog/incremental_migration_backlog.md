# Revised Backlog: Incremental Migration from Current Code

## Assessment & Analysis

*Analysis tickets removed - proceeding with implementation based on existing codebase knowledge*

## Incremental Migration Strategy

### MIGRATE-001: Add Binary Format Alongside HTTP
- Create new binary serialization functions for existing state format
- Add rhomb-to-vector conversion that works with current `get_visible_cells()`
- Keep HTTP endpoints working while adding shared memory capability
- Add feature flag to switch between HTTP and shared memory

### MIGRATE-002: Shared Memory Layer (Non-Breaking)
- Add SharedMemoryManager as optional component
- Create wrapper around existing GameManager
- Add shared memory writes alongside HTTP responses
- Ensure existing HTTP flow continues working

### MIGRATE-003: Add Clock Service (Optional Mode)
- Create Clock service that can work with existing services
- Add event synchronization as opt-in feature
- Allow services to run in both HTTP mode and Clock mode
- Maintain backward compatibility

## Incremental Service Updates

### ENV-MIGRATE-001: Environment Service Gradual Migration
- Add shared memory capability to existing Flask app
- Create new routes that write to both HTTP response AND shared memory
- Add Clock event handling as optional feature
- Maintain existing Flask endpoints for testing/debugging

### INF-MIGRATE-001: Inference Service Gradual Migration
- Add shared memory reading capability alongside HTTP requests
- Create adapter pattern: `HttpClient` vs `SharedMemoryClient`
- Allow inference to switch modes via configuration
- Keep existing HTTP fallback

### INF-MIGRATE-002: Neural Network Integration
- Update existing `snake_agent.py` to handle binary state format
- Modify existing `state_processor.py` for vector input
- Ensure existing model loading/saving continues working
- Add conversion utilities between old and new formats

## Compatibility & Testing

### COMPAT-001: Dual-Mode Operation
- Implement configuration-based mode switching
- Add HTTP→SharedMemory bridge for testing
- Create comparison tools (HTTP vs SharedMemory results)
- Ensure identical behavior in both modes

### TEST-MIGRATE-001: Test Current Code First
- Add unit tests for existing HTTP endpoints
- Test current GameManager and SnakeGame logic
- Create baseline performance tests for HTTP version
- Document current behavior as acceptance criteria

### TEST-MIGRATE-002: Migration Testing
- Test binary format produces same results as JSON
- Test shared memory matches HTTP response data
- Validate Clock-controlled execution vs free-running
- Performance comparison testing

## Deployment Strategy

### DEPLOY-MIGRATE-001: Gradual Rollout
- Update existing docker-compose.yaml to support both modes
- Add feature flags for shared memory vs HTTP
- Create migration scripts for configuration
- Document rollback procedures

### DEPLOY-MIGRATE-002: Production Migration
- Update existing K8s manifests with backward compatibility
- Add Clock service as optional sidecar
- Create blue-green deployment strategy
- Monitor performance during migration

## Code Reuse & Preservation

### REUSE-001: Preserve Game Logic
- Keep existing `game/manager.py` and `game/snake.py` largely unchanged
- Reuse existing collision detection, food spawning, reward calculation
- Preserve existing configuration parsing and validation
- Maintain existing logging formats

### REUSE-002: Preserve Neural Network Code
- Keep existing `snake_agent.py` and neural network architecture
- Reuse existing training logic and batch processing
- Preserve model serialization and loading
- Maintain existing hyperparameter handling

### REUSE-003: Preserve Configuration
- Keep all existing CLI arguments and environment variables
- Preserve existing Docker and K8s configuration structure
- Maintain existing logging and monitoring integration
- Reuse existing error handling patterns

---

## Revised Sprint Plan

### Sprint 1: Foundation & Testing
- TEST-MIGRATE-001 (test current code)
- Start implementation based on existing codebase

### Sprint 2: Add Binary Format (Non-Breaking)
- MIGRATE-001 (binary format alongside HTTP)
- COMPAT-001 (dual-mode operation)
- Keep everything working while adding new capability

### Sprint 3: Add Shared Memory (Optional)
- MIGRATE-002 (shared memory layer)
- ENV-MIGRATE-001 (gradual env migration)
- Still maintaining HTTP as primary

### Sprint 4: Inference Integration
- INF-MIGRATE-001, INF-MIGRATE-002
- TEST-MIGRATE-002 (migration testing)
- Services can run in either mode

### Sprint 5: Clock Service & Full Migration
- MIGRATE-003 (add Clock service)
- DEPLOY-MIGRATE-001, DEPLOY-MIGRATE-002
- Complete migration with rollback capability

This approach respects the existing codebase and provides a safe migration path rather than a complete rewrite.