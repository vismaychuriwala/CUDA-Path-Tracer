#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromOBJ(
        const std::string obj_filename, 
        const int override_materialid, 
        const glm::vec3 translation,
        const glm::vec3 rotation, const glm::vec3 scale, const int mesh_ID, 
        std::vector<TriangleVerts> &triangleInfo
    );
public:
    Scene(std::string filename);
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
    // std::vector<Geom> boundingBoxes;
    std::vector<LinearBVHNode> bvhNodes;
    std::vector<TriangleVerts>  bvhTriangles;
};
