## Cold Setup

Starting from scratch? Here's how to get your ClickHouse data stack running:

### Prerequisites

- kubectl configured for your cluster
- Helm 3 installed
- Access to pull from Docker Hub (bitnamicharts)

### Quick Setup (All-in-One)

If you want to run all steps automatically:

```bash
make quick-deploy
```

This runs init, secrets, deploy, health check, and create-tables in sequence.

### What Gets Deployed

- **ClickHouse**: OLAP database for analytics
- **RabbitMQ**: Message queue for data ingestion
- **Persistent volumes**: For data storage
- **Kubernetes secrets**: For secure credential management

### Getting Connection Details

After deployment, get connection information:

```bash
make how-to-connect
```

This shows URLs, credentials, and port-forwarding instructions for accessing your services.

### Troubleshooting

- Check deployment status: `make status`
- View service logs: `make logs`
- Run health checks: `make health`
- View passwords: `make passwords`

### Cleanup

To completely remove everything (WARNING: destroys all data):

```bash
make destroy
```

You'll need to type 'DELETE' to confirm.
