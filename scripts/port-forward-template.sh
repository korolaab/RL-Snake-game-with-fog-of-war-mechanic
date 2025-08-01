#!/bin/bash
# Port Forward Script Template
# Copy to port-forward.sh and customize

NAMESPACE="data-stack"

echo "🔗 Setting up port forwarding..."

# Check if services are running
check_service() {
    if ! kubectl get svc $1 -n ${NAMESPACE} >/dev/null 2>&1; then
        echo "❌ Service $1 not found in namespace ${NAMESPACE}"
        return 1
    fi
    return 0
}

# Port forward function
start_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=$3
    
    if check_service ${service}; then
        kubectl port-forward svc/${service} ${local_port}:${remote_port} -n ${NAMESPACE} &
        echo "✅ ${service}: localhost:${local_port}"
        return $!
    fi
}

# Start port forwards
start_port_forward "clickhouse" "8123" "8123"
CH_PID=$?

start_port_forward "rabbitmq" "15672" "15672"
RMQ_PID=$?

start_port_forward "rabbitmq" "5672" "5672"
RMQ_AMQP_PID=$?

echo ""
echo "🎯 Access services at:"
echo "  - ClickHouse: http://localhost:8123/play"
echo "  - RabbitMQ Management: http://localhost:15672"
echo ""
echo "Press Ctrl+C to stop"

# Cleanup
cleanup() {
    echo "🛑 Stopping port forwards..."
    jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM
wait
