SOFIE is experimental implementation for GPU inference using alpaka heterogenous library. This implementation is currently hosted in the gpu/alpaka branch of the standalone SOFIE repository. Alpaka provides a C++ abstraction layer that allows writing a single kernel that can run on various backends: CUDA (NVIDIA), HIP (AMD), SYCL (Intel/generic), and CPU (OpenMP/Threads).

For architecture, I understand that SOFIE will translate a machine learning model ONNX to opitmized C++ code. 

Rmodel_ALPAKA.cxx handles overall strucuture and buffer allocation and session generation

ROperator_*.hxx: Specific operators implement Generate_GPU_ALPAKA and Generate_GPU_Kernel_ALPAKA to emit Alpaka-compatible kernels and call code.
There is also part for session struct which is the template acts as a container:
Accelerator
Tensors
Kernels
Inference solutions
Kernel structure: elemental operations (like ReLU or Add) are C++ structs with templated operator()


Need for improvements:
Operator coverage:
Recurrent layer: LSTM, GRU, RNN
Convolutions: Conv, ConvTranspose
Pooling: MaxPool and AveragePool
Normalization
	
	Memory Management
Buffer Reuse: Intermediate buffers are currently allocated per tensor name. Implementing a memory planner to reuse buffers for non-overlapping lifetimes would significantly reduce GPU memory usage.
Asynchronicity: Improving the overlap of data transfer (H2D/D2H) with kernel execution using multiple streams/queues.
	 Kernel Fusion
Implementing horizontal or vertical fusion at the codegen level. For example, fusing a
Gemm operation followed by an Add (bias) and ReLU into a single kernel call (note: sofieBLAS already provides some support for gemmrelu).
For assignment 4, I was working on Elu and Selu and wrote a test for them

For Assignment 5, I was working on ConV and test with Convwithpadding and Conv


Their responsibilities include:

- emitting GPU kernel definitions
- generating kernel launch code
- connecting input/output tensors

Examples of operators include:

- `ROperator_Tanh`
- `ROperator_Sigmoid`
- `ROperator_Elu`
- `ROperator_Conv`
- `ROperator_BatchNormalization`

---

# Session Structure

The generated inference code includes a template struct:


This struct acts as the **runtime container for model inference**.

It includes:

### Accelerator

Defines the execution backend.

Examples:

alpaka::TagCpu


alpaka::TagGpuCudaRt


---

### Tensor Buffers

Model weights and intermediate tensors are stored using Alpaka buffers:


These buffers can reside on CPU or GPU depending on the accelerator type.

---

### Kernel Instances

Each operator generates a corresponding kernel struct.

Example:


---

### Inference Function

The generated function:


executes the sequence of kernels that correspond to the computational graph.

---

# Kernel Structure

Most element-wise operators are implemented using a **grid-stride kernel pattern**.

Example structure:

```cpp
template<typename TAcc>
ALPAKA_FN_ACC void operator()(TAcc const& acc, ...)
{
    auto idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

    if(idx < numElements)
        output[idx] = operation(input[idx]);
}
```

# Assignment Work
Assignment 4 – Operator Implementation

For this exercise I implemented and tested the following operators:

ELU

SELU

Tasks completed:

implemented Alpaka GPU kernels

added operator integration into SOFIE

wrote unit tests validating operator correctness

Assignment 5 – Convolution Operators

For this exercise I worked on convolution-based operators including:

Conv

ConvWithPadding

The implementation included:

generating Alpaka GPU kernels

integrating the operators into SOFIE

writing tests using ONNX models to validate GPU execution

# Testing

Tests verify correctness by comparing GPU outputs against reference outputs generated from ONNX models.

Operators tested include:

Linear

LinearWithSigmoid

LinearWithLeakyRelu

AddBroadcast

Transpose

Concat

ScatterElements

Split

Tile