#!/usr/bin/env python3
"""
Script to generate gRPC Python code from proto files.
Run this in both Training and Inference directories.
"""

import subprocess
import sys
import os

def generate_grpc_code():
    """Generate Python gRPC code from proto file."""
    proto_file = "training.proto"
    
    if not os.path.exists(proto_file):
        print(f"Error: {proto_file} not found!")
        print("Please copy training.proto to this directory first.")
        return False
    
    try:
        # Generate Python code from proto file
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            "--proto_path=.",
            "--python_out=.",
            "--grpc_python_out=.",
            proto_file
        ]
        
        print(f"Generating gRPC code from {proto_file}...")
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ gRPC code generated successfully!")
            print("Generated files:")
            print("  - training_pb2.py")
            print("  - training_pb2_grpc.py")
            return True
        else:
            print(f"❌ Error generating gRPC code:")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during code generation: {e}")
        return False

if __name__ == "__main__":
    success = generate_grpc_code()
    if not success:
        sys.exit(1)
    
    print("\n📝 Next steps:")
    print("1. Copy training.proto to your Training and Inference directories")
    print("2. Run this script in both directories")
    print("3. Use the generated training_pb2.py and training_pb2_grpc.py in your code")
