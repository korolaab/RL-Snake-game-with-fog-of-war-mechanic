#!/bin/bash

# Universal protobuf setup script for Docker containers
# This script installs protobuf and generates Python classes

set -e  # Exit on any error

echo "🔧 Setting up Protobuf in Docker container..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install protobuf compiler
echo "📦 Installing protobuf compiler..."
apt-get install -y protobuf-compiler

echo "✅ Protobuf compiler installed: $(protoc --version)"

# Install Python protobuf library
echo "📦 Installing Python protobuf library..."
pip install protobuf

# Check if proto file exists
PROTO_FILE="/app/training.proto"
if [ ! -f "$PROTO_FILE" ]; then
    echo "❌ Error: $PROTO_FILE not found!"
    echo "Make sure to copy training.proto to /app/ in Dockerfile"
    exit 1
fi

# Generate Python classes from proto file
echo "🔧 Generating Python protobuf classes..."
cd /app
protoc --python_out=. training.proto

# Verify generated file
if [ -f "/app/training_pb2.py" ]; then
    echo "✅ Generated rl_communication_pb2.py successfully"
else
    echo "❌ Failed to generate protobuf classes"
    exit 1
fi

# Clean up
echo "🧹 Cleaning up..."
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "🎉 Protobuf setup completed successfully!"
echo "📋 Generated files:"
echo "   - /app/rl_communication_pb2.py"
