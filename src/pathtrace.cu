#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/gather.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include <map>
#include <algorithm>

#define ERRORCHECK 0
#define MAX_MESHES 64
#define TILE_SIZE 64  // Fixed t-array size per ray



#define EVALUATION 0

#define NAIVE 1
#define BOUNDING_BOX 1

#define JITTER 1
#define DOF 1

#define STREAM_COMPACT 1
#define COALESCED 0
#define PRINT_RAY_COUNT 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static LinearBVHNode* dev_bvhNodes = NULL;
static TriangleVerts*  dev_bvhTriangles = NULL;

#if EVALUATION
// Performance evaluation timing
static cudaEvent_t start, stop;
static bool events_initialized = false;
static float time_raygen = 0.0f;
static float time_intersection = 0.0f;
static float time_sort = 0.0f;
static float time_shading = 0.0f;
static float time_compaction = 0.0f;
static int perf_iter_count = 0;
#endif

#if !NAIVE
static float* dev_t_vals = NULL;  // Reusable fixed-size buffer
static float* dev_min_t = NULL;   // Per-ray minimum t
static int* dev_min_idx = NULL;   // Per-ray minimum geom index
#endif

#if COALESCED
int* dev_keys = nullptr;
int* dev_indices = nullptr;
ShadeableIntersection* dev_inter_tmp;
PathSegment* dev_paths_tmp;
#endif

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // BVH data
    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(LinearBVHNode));
        cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    }
    if (!scene->bvhTriangles.empty()) {
        cudaMalloc(&dev_bvhTriangles, scene->bvhTriangles.size() * sizeof(TriangleVerts));
        cudaMemcpy(dev_bvhTriangles, scene->bvhTriangles.data(), scene->bvhTriangles.size() * sizeof(TriangleVerts), cudaMemcpyHostToDevice);
    }

#if COALESCED
        cudaMalloc(&dev_keys, pixelcount * sizeof(int));
        cudaMalloc(&dev_indices, pixelcount * sizeof(int));
        cudaMalloc(&dev_inter_tmp, pixelcount * sizeof(ShadeableIntersection));
        cudaMalloc(&dev_paths_tmp, pixelcount * sizeof(PathSegment));
#endif

#if !NAIVE
    // Allocate fixed-size tile buffer
    cudaMalloc(&dev_t_vals, pixelcount * TILE_SIZE * sizeof(float));
    cudaMalloc(&dev_min_t, pixelcount * sizeof(float));
    cudaMalloc(&dev_min_idx, pixelcount * sizeof(int));
#endif
    // cudaMalloc(&dev_mesh_in_scope, num_meshes * sizeof(bool));

#if EVALUATION
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    events_initialized = true;
    time_raygen = 0.0f;
    time_intersection = 0.0f;
    time_sort = 0.0f;
    time_shading = 0.0f;
    time_compaction = 0.0f;
    perf_iter_count = 0;
#endif
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

#if !NAIVE
    cudaFree(dev_t_vals);
    cudaFree(dev_min_t);
    cudaFree(dev_min_idx);
#endif
    cudaFree(dev_bvhNodes);
    cudaFree(dev_bvhTriangles);
#if COALESCED
        cudaFree(dev_keys);
        cudaFree(dev_indices);
        cudaFree(dev_inter_tmp);
        cudaFree(dev_paths_tmp);
#endif

#if EVALUATION
    if (events_initialized) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        events_initialized = false;
    }
#endif

    checkCUDAError("pathtraceFree");
}

__device__ glm::vec2 concentricSampleDisk(float u1, float u2) {
    float sx = 2.0f * u1 - 1.0f;
    float sy = 2.0f * u2 - 1.0f;

    if (sx == 0.0f && sy == 0.0f) return glm::vec2(0.0f);

    float r, theta;
    if (fabsf(sx) > fabsf(sy)) {
        r = sx;
        theta = (PI / 4) * (sy / sx);
    } else {
        r = sy;
        theta = (PI / 2) - (PI / 4) * (sx / sy);
    }
    return glm::vec2(r * cosf(theta), r * sinf(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if JITTER
        // Anti-aliasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::normal_distribution<float> normal(0.0f, 0.005f);
        float jitterX = normal(rng);
        float jitterY = normal(rng);
        jitterX = fminf(fmaxf(jitterX, -0.5f), 0.5f);
        jitterY = fminf(fmaxf(jitterY, -0.5f), 0.5f);
        float px = (float)x + jitterX;
        float py = (float)y + jitterY;
#else
        float px = (float)x;
        float py = (float)y;
#endif
        glm::vec3 dir_pinhole = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (px - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (py - (float)cam.resolution.y * 0.5f)
        );


        glm::vec3 rayOrigin = cam.position;
        glm::vec3 rayDir = dir_pinhole;
#if DOF
        if (cam.lensRadius > 0.0f) {
#if !JITTER
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
#endif
            // Uniform randoms using your thrust RNG
            thrust::uniform_real_distribution<float> uni01(0.0f, 1.0f);
            float r1 = uni01(rng);
            float r2 = uni01(rng);

            // compute focal point along the pinhole ray
            float denom = glm::dot(dir_pinhole, cam.view);
            denom = (fabsf(denom) < 1e-6f) ? 1e-6f * (denom >= 0.0f ? 1.0f : -1.0f) : denom;
            float t_focus = cam.focalDistance / denom;
            glm::vec3 p_focus = cam.position + dir_pinhole * t_focus;

            // sample lens disk and offset origin
            glm::vec2 lensSample = concentricSampleDisk(r1, r2) * cam.lensRadius;
            rayOrigin = cam.position + cam.right * lensSample.x + cam.up * lensSample.y;

            rayDir = glm::normalize(p_focus - rayOrigin);
        }
#endif
        segment.ray.origin = rayOrigin;
        segment.ray.direction = rayDir;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// Tiled intersection - compute TILE_SIZE geometries at a time
#if !NAIVE
__global__ void kernComputeTValsTiled(
    int tile_start,
    int tile_size,
    int geoms_size,
    int num_paths,
    const Geom* geoms,
    const PathSegment* pathSegments,
    float* t_vals)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tile_size * num_paths) return;

    int geom_offset = tid % tile_size;  // Position within tile
    int path_index = tid / tile_size;
    int geom_index = tile_start + geom_offset;

    PathSegment pathSegment = pathSegments[path_index];

    const int t_index = path_index * TILE_SIZE + geom_offset;
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    bool outside = true;

    // Check if this geometry index is out of bounds
    if (geom_index >= geoms_size) {
        t_vals[t_index] = -1.0f;
        return;
    }

    const Geom& geom = geoms[geom_index];

    if (geom.type == CUBE) {
        t_vals[t_index] = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == SPHERE) {
        t_vals[t_index] = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == TRIANGLE) {
        t_vals[t_index] = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
    }
    else {
        t_vals[t_index] = -1.0f;
    }
}

__global__ void kernFindMinT(
    int tile_start,
    int tile_size,
    int num_paths,
    const float* t_vals,
    float* global_min_t,
    int* global_min_idx)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    float min_t = global_min_t[path_index];
    int min_idx = global_min_idx[path_index];

    // Find minimum in this tile
    for (int i = 0; i < tile_size; i++) {
        int t_index = path_index * TILE_SIZE + i;
        float t = t_vals[t_index];
        if (t > 0.0f && t < min_t) {
            min_t = t;
            min_idx = tile_start + i;  // Global geometry index
        }
    }

    global_min_t[path_index] = min_t;
    global_min_idx[path_index] = min_idx;
}

__global__ void kernComputeFinalIntersection(
    int num_paths,
    const PathSegment* pathSegments,
    const Geom* geoms,
    const float* global_min_t,
    const int* global_min_idx,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= num_paths) return;

    int geom_idx = global_min_idx[path_index];
    float min_t = global_min_t[path_index];

    if (geom_idx == -1 || min_t >= FLT_MAX) {
        intersections[path_index].t = -1.0f;
        return;
    }

    PathSegment pathSegment = pathSegments[path_index];
    const Geom& geom = geoms[geom_idx];

    glm::vec3 intersect;
    glm::vec3 normal;
    bool outside = true;
    float t = -1.0f;

    if (geom.type == CUBE) {
        t = boxIntersectionTest(geom, pathSegment.ray, intersect, normal, outside);
    }
    else if (geom.type == SPHERE) {
        t = sphereIntersectionTest(geom, pathSegment.ray, intersect, normal, outside);
    }
    else if (geom.type == TRIANGLE) {
        t = triangleIntersectionTest(geom, pathSegment.ray, intersect, normal, outside);
    }

    intersections[path_index].t = t;
    intersections[path_index].materialId = geom.materialid;
    intersections[path_index].surfaceNormal = normal;
}
#else
__global__ void computeIntersectionsNaive(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    LinearBVHNode* bvhNodes,
    TriangleVerts* bvhTriangles
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // Parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                }
            }
            else if (geom.type == MESH)
            {
                int matId = -1;
                t = meshIntersectionTest(geom, pathSegment.ray, bvhNodes, bvhTriangles,
                                         tmp_intersect, tmp_normal, matId);
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_geom_index = i;
                    intersect_point = tmp_intersect;
                    normal = tmp_normal;
                    intersections[path_index].materialId = matId;
                }
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            intersections[path_index].t = t_min;
            // materialId already set for MESH; set it here for CUBE/SPHERE
            if (geoms[hit_geom_index].type != MESH) {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            }
            intersections[path_index].surfaceNormal = normal;
        }
    }
}
#endif
__global__ void shadeRealMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        PathSegment &pathSegment = pathSegments[idx];

#if !STREAM_COMPACT
        // Skip rays that have already been gathered (marked with -1)
        if (pathSegment.remainingBounces < 0) {
            return;
        }
#endif

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegment.color *= (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
            }

            else {
                if (pathSegment.remainingBounces > 0) {
                    glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                    scatterRay(pathSegment, hitPoint, intersection.surfaceNormal, material, rng);
                }
            }
        }
        else {
            pathSegment.color = BACKGROUND_COLOR;
            pathSegment.remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void gatherImage(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment& iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces == 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
#if !STREAM_COMPACT
            // Mark as gathered to prevent double-accumulation when STREAM_COMPACT is off
            iterationPath.remainingBounces = -1;
#endif
        }
    }
}

// Set the sorting keys to materialId
__global__ void kernSetKeys(int nPaths, int* keys, const ShadeableIntersection* intersections)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        keys[index] = intersections[index].materialId;
    }
}

struct PathIsDead {
  __host__ __device__
  bool operator()(const PathSegment &p) const {
    return p.remainingBounces == 0;
  }
};

int compactPaths_inplace(PathSegment* d_paths, int num_paths) {
  thrust::device_ptr<PathSegment> dev_ptr(d_paths);
  auto new_end = thrust::remove_if(dev_ptr, dev_ptr + num_paths, PathIsDead());
  int new_num_paths = static_cast<int>(new_end - dev_ptr);
  return new_num_paths;
}

// Sort Intersections and Paths according to index
__global__ void kernGatherArrays(int nPaths, int* indices, ShadeableIntersection* inter_out, const ShadeableIntersection* inter_in,
PathSegment* path_out, const PathSegment* path_in
)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < nPaths)
    {
        int src = indices[index];
        inter_out[index] = inter_in[src];
        path_out[index] = path_in[src];
    }
}

void printPerformanceStats()
{
#if EVALUATION
    if (perf_iter_count == 0) {
        printf("No performance data collected yet.\n");
        return;
    }

    float avg_raygen = time_raygen / perf_iter_count;
    float avg_intersection = time_intersection / perf_iter_count;
    float avg_sort = time_sort / perf_iter_count;
    float avg_shading = time_shading / perf_iter_count;
    float avg_compaction = time_compaction / perf_iter_count;
    float total = avg_raygen + avg_intersection + avg_sort + avg_shading;
#if STREAM_COMPACT
    total += avg_compaction;
#endif

    printf("\n========== Performance Stats (Avg over %d iterations) ==========\n", perf_iter_count);
    printf("Ray Generation:   %7.3f ms (%5.1f%%)\n", avg_raygen, (avg_raygen/total)*100.0f);
    printf("Intersection:     %7.3f ms (%5.1f%%)\n", avg_intersection, (avg_intersection/total)*100.0f);
#if COALESCED
    printf("Material Sorting: %7.3f ms (%5.1f%%)\n", avg_sort, (avg_sort/total)*100.0f);
#endif
    printf("Shading:          %7.3f ms (%5.1f%%)\n", avg_shading, (avg_shading/total)*100.0f);
#if STREAM_COMPACT
    printf("Compaction:       %7.3f ms (%5.1f%%)\n", avg_compaction, (avg_compaction/total)*100.0f);
#endif
    printf("----------------------------------------------------------------\n");
    printf("TOTAL:            %7.3f ms\n", total);
    printf("================================================================\n\n");
#endif
}

void resetPerformanceStats()
{
#if EVALUATION
    time_raygen = 0.0f;
    time_intersection = 0.0f;
    time_sort = 0.0f;
    time_shading = 0.0f;
    time_compaction = 0.0f;
    perf_iter_count = 0;
#endif
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

#if EVALUATION
    cudaEventRecord(start);
#endif
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
#if EVALUATION
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    time_raygen += milliseconds;
#endif

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    bool iterationComplete = false;
#if COALESCED
    thrust::device_ptr<int> d_idx(dev_indices);
    thrust::device_ptr<int> d_keys(dev_keys);
#endif

#if PRINT_RAY_COUNT
    if (iter == 10) {  // Print after warmup to avoid spam
        printf("\n[Iter %d] Initial rays: %d\n", iter, num_paths);
    }
#endif

    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        const int geoms_size = hst_scene->geoms.size();
#if EVALUATION
        cudaEventRecord(start);
#endif
#if NAIVE
            computeIntersectionsNaive<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            geoms_size,
            dev_intersections,
            dev_bvhNodes,
            dev_bvhTriangles
        );
#else
            // Tiled intersection: process TILE_SIZE geometries at a time
            // Initialize min arrays
            thrust::device_ptr<float> d_min_t(dev_min_t);
            thrust::device_ptr<int> d_min_idx(dev_min_idx);
            thrust::fill(d_min_t, d_min_t + num_paths, FLT_MAX);
            thrust::fill(d_min_idx, d_min_idx + num_paths, -1);

            // Process geometries in tiles
            int num_tiles = (geoms_size + TILE_SIZE - 1) / TILE_SIZE;
            for (int tile = 0; tile < num_tiles; tile++) {
                int tile_start = tile * TILE_SIZE;
                int tile_size = std::min(TILE_SIZE, geoms_size - tile_start);

                // Compute t-values for this tile
                int totalThreads = tile_size * num_paths;
                int blockSize = 128;
                int numBlocks = (totalThreads + blockSize - 1) / blockSize;
                kernComputeTValsTiled<<<numBlocks, blockSize>>>(
                    tile_start, tile_size, geoms_size, num_paths, dev_geoms, dev_paths, dev_t_vals);

                // Find minimum within this tile
                kernFindMinT<<<numblocksPathSegmentTracing, blockSize1d>>>(
                    tile_start, tile_size, num_paths, dev_t_vals, dev_min_t, dev_min_idx);
            }

            // Compute final intersection from winning geometry
            kernComputeFinalIntersection<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths, dev_paths, dev_geoms, dev_min_t, dev_min_idx, dev_intersections);

            checkCUDAError("compute intersections tiled");
#endif
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#if EVALUATION
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        time_intersection += milliseconds;
#endif
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

# if COALESCED
#if EVALUATION
            cudaEventRecord(start);
#endif
            kernSetKeys<<<numblocksPathSegmentTracing, blockSize1d>>> (num_paths, dev_keys, dev_intersections);
            thrust::sequence(thrust::device, d_idx, d_idx + num_paths);
            thrust::sort_by_key(thrust::device, d_keys, d_keys + num_paths, d_idx);
            kernGatherArrays<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_indices, dev_inter_tmp, dev_intersections, dev_paths_tmp, dev_paths);
            std::swap(dev_inter_tmp, dev_intersections);
            std::swap(dev_paths_tmp, dev_paths);
#if EVALUATION
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            time_sort += milliseconds;
#endif
#endif

#if EVALUATION
        cudaEventRecord(start);
#endif
        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        cudaDeviceSynchronize();
#if EVALUATION
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        time_shading += milliseconds;
#endif

        int n = num_paths;
        int blocksGather = (n + blockSize1d - 1) / blockSize1d;
        gatherImage << <blocksGather, blockSize1d >> > (n, dev_image, dev_paths);
        cudaDeviceSynchronize();
#if STREAM_COMPACT
#if EVALUATION
        cudaEventRecord(start);
#endif
        num_paths = compactPaths_inplace(dev_paths, num_paths);
#if EVALUATION
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        time_compaction += milliseconds;
#endif
#endif
#if PRINT_RAY_COUNT
        if (iter == 10) {  // Print after warmup to avoid spam
            printf("[Iter %d] After bounce %d: %d rays remaining\n", iter, depth, num_paths);
        }
#endif
#if STREAM_COMPACT
        if (num_paths == 0) {
            iterationComplete = true;
        }
#endif
        if (depth >= traceDepth) {
            iterationComplete = true;
        }
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    //// Assemble this iteration and apply it to the image
    //dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    //finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

#if EVALUATION
    perf_iter_count++;
#endif
}
