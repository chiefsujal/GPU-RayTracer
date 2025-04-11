# GPU-Accelerated Ray Tracer  
A physically based renderer implemented in CUDA, achieving **15x speedup** over CPU.  

## Features  
- Parallel ray-triangle intersection tests using CUDA.  
- BVH acceleration for reduced computation.  
- Path tracing with global illumination (WIP).  

## Benchmark  
| Implementation | Render Time (1024x768) |  
|---------------|-----------------------|  
| CPU (Single-threaded) | 12.4 sec |  
| **GPU (CUDA)** | **0.8 sec** |  

## Setup  
```bash
nvcc raytracer.cu -o raytracer
./raytracer
