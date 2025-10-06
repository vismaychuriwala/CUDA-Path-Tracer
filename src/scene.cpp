#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

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
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
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

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
