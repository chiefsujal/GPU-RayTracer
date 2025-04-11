#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// 3D vector struct
struct Vec3 {
    float x, y, z;
    __device__ Vec3 operator+(Vec3 b) { return {x+b.x, y+b.y, z+b.z}; }
    __device__ Vec3 operator*(float b) { return {x*b, y*b, z*b}; }
    __device__ float dot(Vec3 b) { return x*b.x + y*b.y + z*b.z; }
    __device__ Vec3 cross(Vec3 b) { 
        return {y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x};
    }
    __device__ Vec3 normalized() { 
        float len = sqrtf(x*x + y*y + z*z);
        return {x/len, y/len, z/len};
    }
};

// Sphere object
struct Sphere {
    Vec3 center;
    float radius;
};

// Ray-sphere intersection test (GPU-optimized)
__device__ bool hit_sphere(Sphere s, Vec3 ray_origin, Vec3 ray_dir, float &t) {
    Vec3 oc = ray_origin + s.center * -1.0f;
    float a = ray_dir.dot(ray_dir);
    float b = 2.0f * oc.dot(ray_dir);
    float c = oc.dot(oc) - s.radius*s.radius;
    float discr = b*b - 4*a*c;
    if (discr < 0) return false;
    t = (-b - sqrtf(discr)) / (2.0f*a);
    return true;
}

// CUDA kernel to render scene
__global__ void render_kernel(uchar4 *pixels, int width, int height, Sphere *spheres, int num_spheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    Vec3 ray_origin = {0, 0, 0};
    Vec3 ray_dir = {(float)x/width - 0.5f, (float)y/height - 0.5f, 1};
    ray_dir = ray_dir.normalized();

    // Trace rays against all spheres
    float closest_t = INFINITY;
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (hit_sphere(spheres[i], ray_origin, ray_dir, t) {
            if (t < closest_t) closest_t = t;
        }
    }

    // Shade pixel (red if hit, black otherwise)
    if (closest_t < INFINITY) {
        pixels[y*width + x] = {255, 0, 0, 255};  // RGBA
    } else {
        pixels[y*width + x] = {0, 0, 0, 255};
    }
}

// Main function
int main() {
    // Image setup
    int width = 1024, height = 768;
    uchar4 *pixels;
    cudaMallocManaged(&pixels, width*height*sizeof(uchar4));

    // Scene setup (2 spheres)
    Sphere *spheres;
    int num_spheres = 2;
    cudaMallocManaged(&spheres, num_spheres*sizeof(Sphere));
    spheres[0] = {{0, 0, 5}, 1.0f};
    spheres[1] = {{0, -100.5f, 5}, 100.0f};

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15)/16, (height + 15)/16);
    render_kernel<<<blocksPerGrid, threadsPerBlock>>>(pixels, width, height, spheres, num_spheres);
    cudaDeviceSynchronize();

    // Save image (simplified; use stb_image.h for actual PNG output)
    FILE *f = fopen("output.ppm", "wb");
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width*height; i++) {
        fprintf(f, "%d %d %d ", pixels[i].x, pixels[i].y, pixels[i].z);
    }
    fclose(f);

    cudaFree(pixels);
    cudaFree(spheres);
    return 0;
}
