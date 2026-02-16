#!/bin/bash
set -euo pipefail

SPEEDS=(0 0.1 0.2 0.3 0.5 0.7 1.0)
CHART_DIR="k8s/snake-rl"
VALUES_FILE="$CHART_DIR/values.yaml"
NAMESPACE="default"
TMPDIR=$(mktemp -d)

trap "rm -rf $TMPDIR" EXIT

echo "=== Apple Speed Study: deploying ${#SPEEDS[@]} experiments ==="

for speed in "${SPEEDS[@]}"; do
    release="apple-speed-$(echo $speed | tr '.' '-')"
    override="$TMPDIR/override-${speed}.yaml"

    cat > "$override" <<EOF
experiment:
  name: apple-speed-${speed}
env:
  appleSpeed: ${speed}
EOF

    echo "Installing $release (apple_speed=$speed)..."
    helm install "$release" "$CHART_DIR" \
        -n "$NAMESPACE" \
        -f "$VALUES_FILE" \
        -f "$override"
done

echo ""
echo "=== All ${#SPEEDS[@]} experiments deployed ==="
echo "Waiting for all jobs to complete..."

for speed in "${SPEEDS[@]}"; do
    release="apple-speed-$(echo $speed | tr '.' '-')"
    job_name="$release-snake-rl"
    echo "Waiting for $job_name..."
    kubectl wait --for=condition=complete "job/$job_name" \
        -n "$NAMESPACE" \
        --timeout=7200s || echo "WARNING: $job_name did not complete in time"
done

echo ""
echo "=== All experiments finished ==="
echo "Run: kubectl get jobs -n $NAMESPACE | grep apple-speed"
