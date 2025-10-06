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

#define ERRORCHECK 1
const bool COALESCED = 1;

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

int* dev_keys = nullptr;
int* dev_indices = nullptr;
ShadeableIntersection* dev_inter_tmp;
PathSegment* dev_paths_tmp;

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

    // TODO: initialize any extra device memeory you need

    if (COALESCED) {
        cudaMalloc(&dev_keys, pixelcount * sizeof(int));
        cudaMalloc(&dev_indices, pixelcount * sizeof(int));
        cudaMalloc(&dev_inter_tmp, pixelcount * sizeof(ShadeableIntersection));
        cudaMalloc(&dev_paths_tmp, pixelcount * sizeof(PathSegment));
    }
    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
    if (COALESCED) {
        cudaFree(dev_keys);
        cudaFree(dev_indices);
        cudaFree(dev_inter_tmp);
        cudaFree(dev_paths_tmp);
    }

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

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::normal_distribution<float> normal(0.0f, 0.005f);
        float jitterX = normal(rng);
        float jitterY = normal(rng);
        jitterX = fminf(fmaxf(jitterX, -0.5f), 0.5f);
        jitterY = fminf(fmaxf(jitterY, -0.5f), 0.5f);
        float px = (float)x + jitterX;
        float py = (float)y + jitterY;
        glm::vec3 dir_pinhole = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (px - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (py - (float)cam.resolution.y * 0.5f)
        );


        glm::vec3 rayOrigin = cam.position;
        glm::vec3 rayDir = dir_pinhole;

        if (cam.lensRadius > 0.0f) {
            // Uniform randoms using your thrust RNG
            thrust::uniform_real_distribution<float> uni01(0.0f, 1.0f);
            float r1 = uni01(rng);
            float r2 = uni01(rng);

            // compute focal point along the pinhole ray: find t so that camera + t*dir_pinhole lies
            // on plane at distance focalDistance along cam.view.
            // Assumes cam.view is normalized.
            float denom = glm::dot(dir_pinhole, cam.view);
            // avoid /0
            denom = (fabsf(denom) < 1e-6f) ? 1e-6f * (denom >= 0.0f ? 1.0f : -1.0f) : denom;
            float t_focus = cam.focalDistance / denom;
            glm::vec3 p_focus = cam.position + dir_pinhole * t_focus;

            // sample lens disk and offset origin
            glm::vec2 lensSample = concentricSampleDisk(r1, r2) * cam.lensRadius;
            rayOrigin = cam.position + cam.right * lensSample.x + cam.up * lensSample.y;

            // new direction goes from sampled lens point to focal point
            rayDir = glm::normalize(p_focus - rayOrigin);
        }
        segment.ray.origin = rayOrigin;
        segment.ray.direction = rayDir;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
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

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

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
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            PathSegment &pathSegment = pathSegments[idx];

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegment.color *= (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                if (pathSegment.remainingBounces > 0) {
                    glm::vec3 hitPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction;
                    scatterRay(pathSegment, hitPoint, intersection.surfaceNormal, material, rng);
                }
                // float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                // pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                // pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void gatherImage(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.remainingBounces == 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
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

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        if (COALESCED) {

            thrust::device_ptr<int> d_keys(dev_keys);
        
            kernSetKeys<<<numblocksPathSegmentTracing, blockSize1d>>> (num_paths, dev_keys, dev_intersections);
            thrust::device_ptr<int> d_idx(dev_indices);
            thrust::sequence(thrust::device, d_idx, d_idx + num_paths);
            thrust::sort_by_key(thrust::device, d_keys, d_keys + num_paths, d_idx);
            kernGatherArrays<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_indices, dev_inter_tmp, dev_intersections, dev_paths_tmp, dev_paths);
            std::swap(dev_inter_tmp, dev_intersections);
            std::swap(dev_paths_tmp, dev_paths);
        }

        shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        cudaDeviceSynchronize();

        int n = num_paths;
        int blocksGather = (n + blockSize1d - 1) / blockSize1d;
        gatherImage << <blocksGather, blockSize1d >> > (n, dev_image, dev_paths);
        cudaDeviceSynchronize();
        num_paths = compactPaths_inplace(dev_paths, num_paths);
        if (num_paths == 0) {
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
}
