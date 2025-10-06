#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ inline float fresnelSchlick(float cosTheta, float etaI, float etaT)
{
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    float EPS = 0.001f;
    pathSegment.ray.origin = intersect + normal * EPS;

    // Purely Diffuse
    if (m.hasReflective == 0 && m.hasRefractive == 0.0) {
        glm::vec3 diffuseDirection = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.direction = diffuseDirection;
        pathSegment.color *= m.color;
    }

    // Reflective
    else if (m.hasReflective != 0.0f && m.hasRefractive == 0.0f)
    {
        float roughness = 1.0f - m.hasReflective;
        float diffuseLuma  = glm::dot(m.color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
        float specularLuma = glm::dot(m.specular.color, glm::vec3(0.2126f, 0.7152f, 0.0722f));

        // Weighted by roughness
        specularLuma *= (1.0f - roughness);  // stronger reflection when smoother
        diffuseLuma  *= (roughness + 0.2f);  // add small base to avoid pure mirror

        float sum = diffuseLuma + specularLuma + 1e-6f;
        float p_diffuse  = diffuseLuma / sum;

        thrust::uniform_real_distribution<float> u01(0, 1);
        float r = u01(rng);

        // Diffuse branch
        if (r < p_diffuse)
        {
            glm::vec3 diffuseDirection = calculateRandomDirectionInHemisphere(normal, rng);
            pathSegment.ray.direction = diffuseDirection;
            pathSegment.color *= m.color;
        }
        // Reflection Branch
        else {
            glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
            // Set new ray
            pathSegment.ray.direction = glm::normalize(reflectDir);

            // Multiply throughput by specular color
            pathSegment.color *= m.specular.color;
        }
    }

    // Refractive
    if (m.hasRefractive != 0.0f) {

        float iorFrom = 1.0f; // medium - air
        float iorTo   = m.indexOfRefraction;

        glm::vec3 I = pathSegment.ray.direction;
        float cosThetaI = glm::dot(-I, normal);

        bool entering = cosThetaI > 0.0f;
        // If not entering, flip normal and swap iors
        if (!entering) {
            normal = -normal;
            cosThetaI = glm::dot(-I, normal); // recompute (now positive)
            iorFrom = iorTo;
            iorTo = 1.0f;
        }

        float eta = iorFrom / iorTo;

        // Fresnel reflectance
        float reflectProb = fresnelSchlick(cosThetaI, iorFrom, iorTo);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rrand = u01(rng);

        glm::vec3 refractedDir = glm::refract(I, normal, eta);
        bool tir = glm::length(refractedDir) < 1e-8f; // treat near-zero as TIR

        if (tir || rrand < reflectProb) {
            // Reflect
            glm::vec3 reflectDir = glm::reflect(I, normal);
            pathSegment.ray.direction = glm::normalize(reflectDir);
            pathSegment.ray.origin = intersect + normal * EPS;
            pathSegment.color *= m.specular.color;
        } else {
            // Refract
            pathSegment.ray.direction = glm::normalize(refractedDir);
            pathSegment.ray.origin = intersect - normal * EPS;
            pathSegment.color *= m.color; // or m.transmission if available
        }
    }

    pathSegment.remainingBounces--;
}
