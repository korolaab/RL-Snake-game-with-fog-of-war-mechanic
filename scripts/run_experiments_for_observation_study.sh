#!/bin/bash

BASE_VALUES="k8s/snake-rl/values.yaml"
CHART_DIR="k8s/snake-rl"
NAMESPACE="observation-study"

grid_sizes=(11)
beta_values=(0.0001 0.001 0.05)

snake_id="adam"
counter=1
TIMEOUT_SECONDS=$((60*60))

# # Ensure namespace exists
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || {
  echo "Creating namespace: $NAMESPACE"
  kubectl create namespace "$NAMESPACE"
}

for grid in "${grid_sizes[@]}"; do
  for beta in "${beta_values[@]}"; do
    beta_safe=$(echo "$beta" | sed 's/\.//g')  # e.g. 0.001 -> 0001
    suffix="g${grid}-b${beta_safe}"
    experiment_name="obs_study_grid_size_${grid}_beta_${beta_safe}"
    release_name="srl-${suffix}"
    uid="exp${counter}"
    counter=$((counter + 1))

    TMP_VALUES="override-${uid}.yaml"

    cat > "$TMP_VALUES" <<EOF
env:
  args:
    gridWidth: ${grid}
    gridHeight: ${grid}

inference:
  args:
    beta: ${beta}
    snakeId: "${snake_id}"

experiment:
  name: "${experiment_name}"
EOF

    echo "Installing experiment: $experiment_name as release: $release_name in namespace: $NAMESPACE"
    helm install "${release_name}" "${CHART_DIR}" \
    --namespace "$NAMESPACE" \
    -f "$BASE_VALUES" -f "$TMP_VALUES"


    echo "Pods ready (or timeout). Starting (($TIMEOUT_SECONDS/60))-minute timer..."
    sleep $TIMEOUT_SECONDS

    echo "Experiment done. Uninstalling release: $release_name"
    helm uninstall "${release_name}" --namespace "$NAMESPACE"


    rm "$TMP_VALUES"
  done
done
