#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        // if (glm::abs(qdxyz) > 0.00001f)
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// AABB slab test — mirrors BVHBounds::intersect but usable on device
__device__ static float aabbIntersect(const BVHBounds& bounds, const Ray& r)
{
    glm::vec3 invDir = glm::vec3(1.0f) / r.direction;
    glm::vec3 tNear  = (bounds.minCorner - r.origin) * invDir;
    glm::vec3 tFar   = (bounds.maxCorner - r.origin) * invDir;
    glm::vec3 tMin   = glm::min(tNear, tFar);
    glm::vec3 tMax   = glm::max(tNear, tFar);
    float t0 = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
    float t1 = glm::min(glm::min(tMax.x, tMax.y), tMax.z);
    if (t0 > t1)  return -1.f;
    if (t0 > 0.f) return t0;
    if (t1 > 0.f) return t1;
    return -1.f;
}

// Möller–Trumbore triangle intersection (world-space vertices, no transform needed)
__device__ static float triangleIntersect(
    const TriangleVerts& tri,
    const Ray& r,
    glm::vec3& normal)
{
    const float EPS = 1e-6f;
    glm::vec3 e1 = tri.v1 - tri.v0;
    glm::vec3 e2 = tri.v2 - tri.v0;

    glm::vec3 h = glm::cross(r.direction, e2);
    float a = glm::dot(e1, h);
    if (glm::abs(a) < EPS) return -1.f;  // ray parallel to triangle

    float f = 1.f / a;
    glm::vec3 s = r.origin - tri.v0;
    float u = f * glm::dot(s, h);
    if (u < 0.f || u > 1.f) return -1.f;

    glm::vec3 q = glm::cross(s, e1);
    float v = f * glm::dot(r.direction, q);
    if (v < 0.f || (u + v) > 1.f) return -1.f;

    float t = f * glm::dot(e2, q);
    if (t < EPS) return -1.f;

    float w = 1.f - u - v;
    normal = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
    if (glm::dot(r.direction, normal) > 0.f)
        normal = -normal;

    return t;
}

#define BVH_STACK_SIZE 64

__device__ float meshIntersectionTest(
    const Geom& mesh,
    const Ray& r,
    const LinearBVHNode* bvhNodes,
    const TriangleVerts* bvhTriangles,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    int& materialId)
{
    float tMin = FLT_MAX;
    glm::vec3 hitNormal(0.f);

    int stack[BVH_STACK_SIZE];
    int stackPtr = 0;
    stack[stackPtr++] = mesh.rootNodeIdx;

    while (stackPtr > 0) {
        int idx = stack[--stackPtr];
        const LinearBVHNode& node = bvhNodes[idx];

        float boxT = aabbIntersect(node.bounds, r);
        if (boxT < 0.f || boxT >= tMin) continue;  // miss or behind current best

        if (node.triangle_idx >= 0) {
            // Leaf: test actual triangle
            glm::vec3 n;
            float t = triangleIntersect(bvhTriangles[node.triangle_idx], r, n);
            if (t > 0.f && t < tMin) {
                tMin      = t;
                hitNormal = n;
                materialId = bvhTriangles[node.triangle_idx].materialId;
            }
        } else {
            // Interior: push both children; left child is always at idx+1
            if (stackPtr + 1 < BVH_STACK_SIZE) {
                stack[stackPtr++] = idx + 1;
                stack[stackPtr++] = node.secondChildOffset;
            }
        }
    }

    if (tMin == FLT_MAX) return -1.f;

    intersectionPoint = r.origin + tMin * r.direction;
    normal = hitNormal;
    return tMin;
}