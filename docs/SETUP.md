# ClickHouse Data Stack Setup Guide

## Prerequisites

- Kubernetes cluster (minikube, kind, or cloud provider)
- Helm 3.x
- kubectl configured

## Initial Setup

1. **Run configuration setup** (one-time):
   ```bash
   ./setup-config.sh
   ```

2. **Initialize working files**:
   ```bash
   make init
   ```

3. **Set credentials** in generated files:
   - `k8s/data-stack/clickhouse/values.yaml`
   - `k8s/data-stack/rabbitmq/values.yaml`

4. **Deploy the stack**:
   ```bash
   make deploy
   ```

## File Structure

```
k8s/
├── namespaces/              # Kubernetes namespaces
├── storage/                 # Storage classes
└── data-stack/              # Service configurations
    ├── clickhouse/
    │   ├── values-template.yaml  # Template (committed)
    │   └── values.yaml          # Working file (not committed)
    └── rabbitmq/
        ├── values-template.yaml  # Template (committed)
        └── values.yaml          # Working file (not committed)

helm/values/                 # Environment-specific overrides
├── dev/
├── staging/
└── production/

scripts/                     # Deployment utilities
├── deploy-template.sh       # Template (committed)
├── deploy.sh               # Working file (not committed)
└── port-forward.sh         # Working file (not committed)
```

## Commands

```bash
make help           # Show all commands
make init           # Create working files from templates
make deploy         # Deploy with default environment (dev)
make dev            # Deploy development environment
make prod           # Deploy production environment
make status         # Check service status
make port-forward   # Access services locally
make destroy        # Remove everything (WARNING: deletes data)
```

## Environments

- **dev**: Minimal resources for local development
- **staging**: Production-like setup for testing
- **production**: Full resources with clustering

Set environment: `ENVIRONMENT=staging make deploy`

## Security Notes

- Templates contain no credentials
- Working files with credentials are gitignored
- Network policies restrict access between namespaces
- Use Kubernetes secrets for production credentials

## Troubleshooting

1. **Services not starting**: Check `make status` and `make logs`
2. **Port conflicts**: Stop existing port forwards
3. **Storage issues**: Verify storage class exists
4. **Permission errors**: Check pod security contexts
