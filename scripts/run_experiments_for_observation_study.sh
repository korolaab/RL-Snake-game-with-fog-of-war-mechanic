#!/bin/bash
set -euo pipefail

BASE_VALUES="k8s/snake-rl/values.yaml"
CHART_DIR="k8s/snake-rl"
NAMESPACE="observation-study"

grid_sizes=(11)
beta_values=(0.0001 0.001 0.05)

snake_id="adam"
counter=1
TIMEOUT_SECONDS=$((200*60))   # overall cap for each experiment (used by the poller)

# --- ClickHouse connection settings ---
CH_HOST="192.168.88.253"
CH_USER="default"
CH_PASS="uf80Zfvt7XWORQtn"

# --- Monitoring settings ---
TARGET=1000        # stop when max_episode >= TARGET
POLL_SECONDS=5     # poll interval

wait_for_episodes() {
  # Args:
  #   $1 - experiment_name
  local exp_name="$1"
  local start_ts now val rc

  # SQL with a named parameter to safely inject the experiment name
  read -r -d '' SQL <<'SQL'
SELECT max(toInt64(data.episode)) AS max_episode
FROM raw.rl_snake_logs
PREWHERE experiment_name = {exp:String}
  AND deploy_dt = (
    SELECT max(deploy_dt) FROM raw.rl_snake_logs
    PREWHERE experiment_name = {exp:String}
  )
SQL

  echo "Waiting until max_episode >= ${TARGET} for experiment '${exp_name}' (polling every ${POLL_SECONDS}s, max ${TIMEOUT_SECONDS}s)..."
  start_ts=$(date +%s)

  while :; do
    set +e
    val=$(
        kubectl exec -n data-stack -it clickhouse-shard0-0 --  clickhouse-client \
        -h "$CH_HOST" -u "$CH_USER" --password "$CH_PASS" \
        --param_exp="$exp_name" \
        --format=TSVRaw \
        -q "$SQL"
    )
    rc=$?
    set -e
    # Clean any trailing newline/whitespace that made the value look "non-numeric"
    val=$(printf '%s' "$raw" | trim_number)

    if [[ $rc -eq 0 && $val =~ ^[0-9]+$ ]]; then
      echo "Current max_episode: $val"
      if (( val >= TARGET )); then
        echo "Target reached for '${exp_name}' 🎉"
        return 0
      fi
    else
      echo "Query failed or returned non-numeric output (rc=$rc, val='${val:-}'). Will retry..."
    fi

    now=$(date +%s)
    if (( now - start_ts >= TIMEOUT_SECONDS )); then
      echo "Timed out after ${TIMEOUT_SECONDS}s waiting for max_episode >= ${TARGET} for '${exp_name}'."
      return 1
    fi

    sleep "$POLL_SECONDS"
  done
}

# Ensure namespace exists
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
    sleep 60
    # ---- Replaced sleep with ClickHouse monitoring ----
    if wait_for_episodes "$experiment_name"; then
      echo "Monitoring complete: threshold hit for ${experiment_name}"
    else
      echo "Monitoring complete: threshold NOT reached for ${experiment_name}"
    fi

    echo "Experiment done. Uninstalling release: $release_name"
    helm uninstall "${release_name}" --namespace "$NAMESPACE"

    # If threshold not reached, exit non-zero to signal failure in CI
    if ! [[ ${val:-} =~ ^[0-9]+$ ]] || (( ${val:-0} < TARGET )); then
      echo "Final max_episode (${val:-unset}) did not reach target (${TARGET}) for ${experiment_name}." >&2
      rm -f "$TMP_VALUES"
      exit 1
    fi

    rm -f "$TMP_VALUES"
  done
done
