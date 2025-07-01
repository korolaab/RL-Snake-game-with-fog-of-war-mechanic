# Environment Configuration

## Development Environment

For local development and testing:

```yaml
# helm/values/dev/clickhouse-values.yaml
persistence:
  size: 10Gi
  storageClass: "local-path"

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

shards: 1
replicaCount: 1
```

## Staging Environment

Production-like setup for testing:

```yaml
# helm/values/staging/clickhouse-values.yaml
persistence:
  size: 100Gi
  storageClass: "fast-ssd"

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 1000m
    memory: 4Gi

shards: 2
replicaCount: 2
```

## Production Environment

Full production setup with clustering:

```yaml
# helm/values/production/clickhouse-values.yaml
persistence:
  size: 500Gi
  storageClass: "fast-ssd"

resources:
  limits:
    cpu: 8000m
    memory: 16Gi
  requests:
    cpu: 2000m
    memory: 8Gi

shards: 3
replicaCount: 3

# High availability
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app.kubernetes.io/name: clickhouse
        topologyKey: kubernetes.io/hostname
```

## Cloud Provider Configurations

### AWS EKS

```yaml
# Storage class for AWS
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gp3-encrypted
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
  fsType: ext4
volumeBindingMode: WaitForFirstConsumer
```

### Google GKE

```yaml
# Storage class for GCP
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ssd-encrypted
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  replication-type: regional-pd
volumeBindingMode: WaitForFirstConsumer
```

### Azure AKS

```yaml
# Storage class for Azure
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: premium-ssd
provisioner: kubernetes.io/azure-disk
parameters:
  storageaccounttype: Premium_LRS
  kind: Managed
volumeBindingMode: WaitForFirstConsumer
```
