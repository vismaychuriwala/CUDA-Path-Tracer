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

__host__ __device__ float triangleIntersectionTest(
    Geom tri,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;

    q.origin = multiplyMV(tri.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(tri.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 v0 = tri.v0;
    glm::vec3 v1 = tri.v1;
    glm::vec3 v2 = tri.v2;

    float EPS = 0.000001f;
    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;

    glm::vec3 h = glm::cross(q.direction, e2);
    float a = glm::dot(e1, h);

    if (glm::abs(a) < EPS) {return -1.0f;}  // Ray parallel to triangle

    float f = 1.0f / a;
    glm::vec3 s = q.origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) {return -1.0f;}

    glm::vec3 qvec = glm::cross(s, e1);
    float v = f * glm::dot(q.direction, qvec);
    if (v < 0.0f || (u + v) > 1.0f) {return -1.0f;}

    float t = f * glm::dot(e2, qvec);
    if (t <= EPS) {return -1.0f;}

    glm::vec3 localIntersect = getPointOnRay(q, t);

    glm::vec3 n;
    if (glm::length(tri.n0) > 0.0f && glm::length(tri.n1) > 0.0f &&
        glm::length(tri.n2) > 0.0f)
    {
        float w = 1.0f - u - v;
        n = glm::normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
    }
    else
    {
        n = glm::normalize(glm::cross(e1, e2));
    }

    outside = glm::dot(q.direction, n) < 0.0f;
    if (!outside)
        n = -n;

    intersectionPoint =
        multiplyMV(tri.transform, glm::vec4(localIntersect, 1.0f));
    normal =
        glm::normalize(multiplyMV(tri.invTranspose, glm::vec4(n, 0.0f)));

    return glm::length(r.origin - intersectionPoint);
}