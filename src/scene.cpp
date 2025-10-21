#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust triangulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"


using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }

    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 0.0;
            newMaterial.hasRefractive = 0.0;
            newMaterial.emittance = 0.0;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"].get<float>();
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            float roughness = p.value("ROUGHNESS", 0.0f);
            roughness = glm::clamp(roughness, 0.0f, 1.0f);
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.hasRefractive = 0.0;
            newMaterial.emittance = 0.0;
            if (p.contains("SPECULAR_COLOR"))
            {
                const auto& scol = p["SPECULAR_COLOR"];
                newMaterial.specular.color = glm::vec3(scol[0], scol[1], scol[2]);
            }
            else
            {
                newMaterial.specular.color = newMaterial.color;
            }
            if (p.contains("SPECULAR_EXPONENT"))
            {
                const auto& s_exp = p["SPECULAR_EXPONENT"];
                newMaterial.specular.exponent = s_exp;
            }
            else {
                newMaterial.specular.exponent = 0.0f;
            }
        }
        else if (p["TYPE"] == "Refractive")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            float transparency = p.value("TRANSPARENCY", 0.0f);
            transparency = glm::clamp(transparency, 0.0f, 1.0f);
            newMaterial.hasRefractive = 1.0f - transparency;
            newMaterial.indexOfRefraction = p.value("IOR", 1.5f);
            float roughness = p.value("ROUGHNESS", 0.0f);
            roughness = glm::clamp(roughness, 0.0f, 1.0f);
            newMaterial.hasReflective = 1.0f - roughness;
            newMaterial.emittance = 0.0f;

            if (p.contains("SPECULAR_COLOR"))
            {
                const auto& scol = p["SPECULAR_COLOR"];
                newMaterial.specular.color = glm::vec3(scol[0], scol[1], scol[2]);
            }
            else
            {
                newMaterial.specular.color = newMaterial.color;
            }
            if (p.contains("SPECULAR_EXPONENT"))
            {
                const auto& s_exp = p["SPECULAR_EXPONENT"];
                newMaterial.specular.exponent = s_exp;
            }
            else {
                newMaterial.specular.exponent = 0.0f;
            }
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    int mesh_ID = 0;
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }

        else if (type == "mesh") {
            string obj_filename;
            if (p.contains("FILE")) {
                obj_filename = p["FILE"];
            }
            else {
                cout << "No filename for mesh" << endl;
                exit(-1);
            }

            int override_materialid = -1;
            if (p.contains("MATERIAL")) {
                override_materialid = MatNameToID[p["MATERIAL"]];
            }
            glm::vec3 translation = glm::vec3(0.f);
            glm::vec3 rotation = glm::vec3(0.f);
            glm::vec3 scale = glm::vec3(1.f);
            if (p.contains("TRANS")) {
                const auto& trans = p["TRANS"];
                translation = glm::vec3(trans[0], trans[1], trans[2]);
            }
            if (p.contains("ROTAT")) {
                const auto& rotat = p["ROTAT"];
                rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            }
            if (p.contains("SCALE")) {
                const auto& scal = p["SCALE"];
                scale = glm::vec3(scal[0], scal[1], scal[2]);
            }
            Geom boundingBox;
            boundingBox.type = CUBE;
            Scene::loadFromOBJ(obj_filename, override_materialid, translation, rotation, scale, mesh_ID, boundingBox);
            boundingBoxes.push_back(boundingBox);
            mesh_ID++;
            continue;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // Depth-of-Field
    camera.focalDistance = cameraData.value("FOCAL_DISTANCE", 10.0f);
    camera.lensRadius = cameraData.value("LENS_RADIUS", 0.0f);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

inline void updateAABB(glm::vec3 &minBound,
                       glm::vec3 &maxBound,
                       const glm::vec3 &p)
{
    minBound.x = glm::min(minBound.x, p.x);
    minBound.y = glm::min(minBound.y, p.y);
    minBound.z = glm::min(minBound.z, p.z);

    maxBound.x = glm::max(maxBound.x, p.x);
    maxBound.y = glm::max(maxBound.y, p.y);
    maxBound.z = glm::max(maxBound.z, p.z);
}
void Scene::loadFromOBJ(const std::string obj_filename, const int override_materialid, const glm::vec3 translation,
    const glm::vec3 rotation, const glm::vec3 scale, const int mesh_ID, Geom &boundingBox) {
    std::filesystem::path objPath(obj_filename);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> obj_shapes;
    std::vector<tinyobj::material_t> obj_materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &obj_shapes, &obj_materials, &warn, &err, objPath.string().c_str());

    if (!warn.empty()) {
    std::cout << "Load Obj warn: " << warn << std::endl;
    }

    if (!err.empty()) {
    std::cerr << "Load Obj err: " << err << std::endl;
    }

    if (!ret) {
    exit(1);
    }

    std::unordered_map<int, int> objMatIDtoGlobal;

    // Loop over materials
    for (size_t i = 0; i < obj_materials.size(); i++) {
        const auto& obj_m = obj_materials[i];
        Material m{};
        m.color = glm::vec3(obj_m.diffuse[0], obj_m.diffuse[1], obj_m.diffuse[2]);

        // Emission
        if (obj_m.emission[0] > 0 || obj_m.emission[1] > 0 || obj_m.emission[2] > 0) {
            m.emittance = glm::length(glm::vec3(obj_m.emission[0], obj_m.emission[1], obj_m.emission[2]));
        }

        // Specular
        if (glm::length(glm::vec3(obj_m.specular[0], obj_m.specular[1], obj_m.specular[2])) > 0.0f) {
            m.hasReflective = 1.0f;
            m.specular.color = glm::vec3(obj_m.specular[0], obj_m.specular[1], obj_m.specular[2]);
            m.specular.exponent = obj_m.shininess > 0 ? obj_m.shininess : 50.0f;
        }

        // Refraction
        if (obj_m.ior > 1.01f) {
            m.hasRefractive = 1.0f;
            m.indexOfRefraction = obj_m.ior;
        }

        objMatIDtoGlobal[i] = (int)materials.size();
        materials.push_back(m);
    }

    glm::vec3 minBound(FLT_MAX);
    glm::vec3 maxBound(-FLT_MAX);
    const glm::mat4 modelMatrix = utilityCore::buildTransformationMatrix(
    translation, rotation, scale);
    const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(modelMatrix)));
    const bool invertWinding = glm::determinant(glm::mat3(modelMatrix)) < 0.0f;

    auto transformPos = [&](const glm::vec3& p) {
        glm::vec4 hp = modelMatrix * glm::vec4(p, 1.0f);
        return glm::vec3(hp);
    };

    auto transformNorm = [&](const glm::vec3& n) {
        return glm::normalize(normalMatrix * n);
    };
    std::cout << "Number of shapes in " + obj_filename + ": " << obj_shapes.size() << endl;
    // Loop over shapes
    for (size_t s = 0; s < obj_shapes.size(); s++) {
        std::cout << "Number of triangles: " << obj_shapes[s].mesh.num_face_vertices.size() << endl;

        // Loop over faces(polygon) 
        size_t index_offset = 0;
        for (size_t f = 0; f < obj_shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = obj_shapes[s].mesh.num_face_vertices[f];
            if (fv != 3) {
                index_offset += fv;
                continue;
            }

            tinyobj::index_t idx0 = obj_shapes[s].mesh.indices[index_offset + 0];
            tinyobj::index_t idx1 = obj_shapes[s].mesh.indices[index_offset + 1];
            tinyobj::index_t idx2 = obj_shapes[s].mesh.indices[index_offset + 2];

            if (invertWinding)
                std::swap(idx1, idx2);

            glm::vec3 p0(attrib.vertices[3 * idx0.vertex_index + 0],
                        attrib.vertices[3 * idx0.vertex_index + 1],
                        attrib.vertices[3 * idx0.vertex_index + 2]);
            glm::vec3 p1(attrib.vertices[3 * idx1.vertex_index + 0],
                        attrib.vertices[3 * idx1.vertex_index + 1],
                        attrib.vertices[3 * idx1.vertex_index + 2]);
            glm::vec3 p2(attrib.vertices[3 * idx2.vertex_index + 0],
                        attrib.vertices[3 * idx2.vertex_index + 1],
                        attrib.vertices[3 * idx2.vertex_index + 2]);

            Geom g{};
            g.type = TRIANGLE;
            g.v0 = transformPos(p0);
            g.v1 = transformPos(p1);
            g.v2 = transformPos(p2);

            updateAABB(minBound, maxBound, g.v0);
            updateAABB(minBound, maxBound, g.v1);
            updateAABB(minBound, maxBound, g.v2);

            // normals
            if (!attrib.normals.empty() &&
                idx0.normal_index >= 0 &&
                idx1.normal_index >= 0 &&
                idx2.normal_index >= 0) {
                g.n0 = transformNorm(glm::vec3(attrib.normals[3 * idx0.normal_index + 0],
                                            attrib.normals[3 * idx0.normal_index + 1],
                                            attrib.normals[3 * idx0.normal_index + 2]));
                g.n1 = transformNorm(glm::vec3(attrib.normals[3 * idx1.normal_index + 0],
                                            attrib.normals[3 * idx1.normal_index + 1],
                                            attrib.normals[3 * idx1.normal_index + 2]));
                g.n2 = transformNorm(glm::vec3(attrib.normals[3 * idx2.normal_index + 0],
                                            attrib.normals[3 * idx2.normal_index + 1],
                                            attrib.normals[3 * idx2.normal_index + 2]));
            } else {
                glm::vec3 faceN = glm::normalize(glm::cross(g.v1 - g.v0, g.v2 - g.v0));
                g.n0 = g.n1 = g.n2 = faceN;
            }

            // material assignment
            int matID = (override_materialid == -1)
                            ? ((obj_shapes[s].mesh.material_ids[f] >= 0)
                                ? objMatIDtoGlobal[obj_shapes[s].mesh.material_ids[f]]
                                : -1)
                            : override_materialid;

            if (matID < 0) {
                Material defaultM{};
                defaultM.color = glm::vec3(0.5f);
                defaultM.emittance = 0.0f;
                matID = (int)materials.size();
                materials.push_back(defaultM);
            }
            g.materialid = matID;

            // identity transforms
            g.translation = glm::vec3(0);
            g.rotation = glm::vec3(0);
            g.scale = glm::vec3(1);
            g.transform = glm::mat4(1.0f);
            g.inverseTransform = glm::mat4(1.0f);
            g.invTranspose = glm::mat4(1.0f);
            g.meshID = mesh_ID;

            geoms.push_back(g);

            index_offset += fv;
        }
    }

    float EPS = 0.0001;
    minBound -= EPS;
    maxBound += EPS;
    boundingBox.translation = (minBound + maxBound) * 0.5f;
    boundingBox.scale = maxBound - minBound;
    boundingBox.rotation = glm::vec3(0.0f);

    boundingBox.transform = utilityCore::buildTransformationMatrix(
    boundingBox.translation, boundingBox.rotation, boundingBox.scale);
    boundingBox.inverseTransform = glm::inverse(boundingBox.transform);
    boundingBox.invTranspose = glm::inverseTranspose(boundingBox.transform);
    boundingBox.meshID = mesh_ID;
    boundingBox.materialid = 0;
    // geoms.push_back(boundingBox);
}