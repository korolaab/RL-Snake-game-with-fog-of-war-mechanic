import torch
import numpy as np
from numba import cuda
import time

# Define a simple PyTorch neural network
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define a CUDA kernel for post-processing
@cuda.jit
def custom_processing_kernel(input_array, output_array, scalar):
    """
    A sample CUDA kernel that applies a custom operation:
    - Multiplies each element by a scalar
    - Applies a threshold operation
    """
    i = cuda.grid(1)
    if i < input_array.shape[0]:
        # Custom calculation (example: apply scalar multiplication and threshold)
        val = input_array[i] * scalar
        if val > 0.5:
            output_array[i] = val
        else:
            output_array[i] = 0.0

# Main function demonstrating the workflow
def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Please ensure you have a compatible GPU and CUDA installed.")
        return
    
    # Parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    batch_size = 1000
    
    # Create and move model to GPU
    model = SimpleNN(input_size, hidden_size, output_size).cuda()
    model.eval()  # Set to evaluation mode
    
    # Create random input data
    input_data = torch.randn(batch_size, input_size, device='cuda')
    
    # Forward pass through the neural network
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Neural network output shape: {output.shape}")
    
    # Step 1: Create a Numba-compatible CUDA array from the PyTorch tensor
    # We'll use as_cuda_array for this
    output_numba = cuda.as_cuda_array(output)
    
    # Step 2: Create another CUDA array for the results of our custom processing
    result_numba = cuda.device_array_like(output_numba)
    
    # Step 3: Configure and launch the CUDA kernel
    threads_per_block = 128
    blocks = (output.numel() + threads_per_block - 1) // threads_per_block
    
    # Parameter for our custom processing
    scalar = 2.0
    
    # Start timing
    start_time = time.time()
    
    # Launch the kernel
    custom_processing_kernel[blocks, threads_per_block](output_numba, result_numba, scalar)
    
    # Synchronize to ensure kernel execution is complete
    cuda.synchronize()
    
    # End timing
    end_time = time.time()
    print(f"CUDA kernel execution time: {(end_time - start_time) * 1000:.2f} ms")
    
    # Step 4: Convert the result back to a PyTorch tensor if needed
    # Option 1: Copy to host memory first (slower but sometimes necessary)
    result_host = result_numba.copy_to_host()
    result_tensor_1 = torch.tensor(result_host, device='cuda')
    
    # Option 2: Create a PyTorch tensor that shares the same memory (faster)
    # Get a pointer to the memory
    ptr = cuda.devicearray.device_pointer(result_numba)
    
    # Create a PyTorch tensor from the pointer
    result_tensor_2 = torch.cuda.FloatTensor(output.shape).set_(
        torch.cuda.current_device(),
        ptr.value,
        result_numba.shape,
        result_numba.strides,
    )
    
    # Verify results
    print("First few values of the final result:")
    print(result_tensor_2[:5])
    
    # Further processing in PyTorch if needed
    final_result = result_tensor_2.sum()
    print(f"Sum of all elements after custom processing: {final_result}")

if __name__ == "__main__":
    main()
