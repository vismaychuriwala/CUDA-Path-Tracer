#include "sceneStructs.h"
#include <memory>

template<typename T>
using uPtr = std::unique_ptr<T>;

template<typename T, typename... Args>
uPtr<T> mkU(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

class BVHNode {
private:
    uPtr<BVHNode> child_L;
    uPtr<BVHNode> child_R;
    TriangleVerts *triangle = nullptr;
    // A bounding box that tightly encompasses
    // the bounding boxes of all children of this node,
    // or of the shape in the node if it is a leaf node.
    BVHBounds bbox;

public:

    BVHNode();
    BVHNode(const BVHNode &b);
    uPtr<BVHNode> clone() const;

    void setupLeafNode(TriangleVerts *t, const BVHBounds &bounds);
    void setupInteriorNode(uPtr<BVHNode> cL, uPtr<BVHNode> cR);
    int flattenBVHTree(int *offset, std::vector<LinearBVHNode>& nodes, std::vector<TriangleVerts> &t) const;

};

// std::optional<Intersection> BVHNode::intersect(const Ray &ray) const {
//     // TODO: Implement a recursive depth-first search intersection test
//     // between the ray and the current node.

//     // Base case: The current node is a leaf node (it has no children)
//     //            Return the intersection with the node's Shape, if there
//     //            is such an intersection.

//     // Recursive cases: The current node is an inner node, and the ray
//     //                  intersects the bounding box of one or more of its children.
//     //                  For efficiency, first traverse the child node with
//     //                  the nearer bounding-box intersection.

//     if (this->shape) {
//         std::optional<Intersection> isect = this->shape->intersect(ray);
//         return isect;
//     }
//     if (!this->bbox.intersect(ray)) {
//         return std::nullopt;
//     }

//     std::optional<Intersection> rIsect = this->child_R->intersect(ray);
//     std::optional<Intersection> lIsect = this->child_L->intersect(ray);

//     if(rIsect) {
//         if (lIsect) {
//             if (lIsect->t < rIsect->t) {
//                 return lIsect;
//             }
//             else {
//                 return rIsect;
//             }
//         }
//         else {
//             return rIsect;
//         }
//     }
//     else {
//         return lIsect;
//     }

//     return std::nullopt;
// }

// Ray-Box intersection test re-implemented for a bounding box
float BVHBounds::intersect(const Ray &ray) const {
    glm::vec3 invDir = glm::vec3(1.0) / ray.direction;
    glm::vec3 near = (this->minCorner - ray.origin) * invDir;
    glm::vec3 far  = (this->maxCorner - ray.origin) * invDir;

    glm::vec3 tmin = glm::min(near, far);
    glm::vec3 tmax = glm::max(near, far);

    float t0 = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
    float t1 = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

    if(t0 > t1) return -1.f;
    if(t0 > 0.f) { // We're outside the box looking at it
        return t0;
    }
    if(t1 > 0.f) { // We're inside the box looking at one of its sides
        return t1;
    }
    return -1.f;
}


// A helper function you can use to determine what dimension
// of a bounding box is longest.
int BVHBounds::maximumExtent() const {
    float maxLen = glm::abs(maxCorner.x - minCorner.x);
    int axis = 0;

    for(int i = 1; i < 3; ++i) {
        float len = glm::abs(maxCorner[i] - minCorner[i]);
        if(len > maxLen) {
            maxLen = len;
            axis = i;
        }
    }
    return axis;
}

glm::vec3 BVHBounds::Offset(const glm::vec3 &p) const {
    glm::vec3 o = p - minCorner;
    glm::vec3 d = maxCorner - minCorner;
    if (d.x > 0) o.x /= d.x;
    if (d.y > 0) o.y /= d.y;
    if (d.z > 0) o.z /= d.z;
    return o;
}


BVHBounds Union(const BVHBounds &a, const BVHBounds &b) {
    glm::vec3 minimum = glm::min(a.minCorner, b.minCorner);
    glm::vec3 maximum = glm::max(a.maxCorner, b.maxCorner);
    return BVHBounds(minimum, maximum);
}

glm::vec3 getCentroid(const TriangleVerts& t) {
    return ((t.v0 + t.v1 + t.v2) / 3.f);
}

BVHBounds getTriangleBounds(const TriangleVerts& t) {
    std::vector<glm::vec3> tPoints({t.v0, t.v1, t.v2});
    return BVHBounds(tPoints);
}

uPtr<BVHNode> recursiveBVHBuild(std::vector<TriangleVerts> &triangleInfo,
                                int start, int end, int *numLeafNodes) {

    uPtr<BVHNode> node = mkU<BVHNode>();
    BVHBounds currentLayerBounds;

    if (start == end) {
        return nullptr;
    }

    for (int t = start; t < end; t++) {
        currentLayerBounds = Union(currentLayerBounds, getTriangleBounds(triangleInfo.at(t)));
    }

    if (start == end - 1) {
        TriangleVerts* tri = &triangleInfo.at(start);
        node->setupLeafNode(tri, currentLayerBounds);
        *numLeafNodes += 1;
    }

    else {
        int axisToExpand = currentLayerBounds.maximumExtent();
        int mid;

            std::sort(triangleInfo.begin() + start, triangleInfo.begin() + end,
                        [axisToExpand](const TriangleVerts& a, const TriangleVerts& b) {
                return getCentroid(a)[axisToExpand] < getCentroid(b)[axisToExpand];
            });
            mid = (start + end) / 2;

        node->setupInteriorNode(recursiveBVHBuild(triangleInfo, start, mid, numLeafNodes),
                                recursiveBVHBuild(triangleInfo, mid, end, numLeafNodes));
    }

    return node;
}

void BVHNode::setupInteriorNode(uPtr<BVHNode> cL, uPtr<BVHNode> cR) {
    child_L = std::move(cL);
    child_R = std::move(cR);
    bbox = Union(child_L->bbox, child_R->bbox);
}


void BVHNode::setupLeafNode(TriangleVerts *t, const BVHBounds &bounds) {
    this->triangle = t;
    this->bbox = bounds;
}

BVHNode::BVHNode()
    : child_L(nullptr), child_R(nullptr),
      triangle(nullptr), bbox()
{}

BVHNode::BVHNode(const BVHNode &b)
    : child_L(b.child_L->clone()), child_R(b.child_R->clone()),
      triangle(b.triangle), bbox(b.bbox)
{}

uPtr<BVHNode> BVHNode::clone() const {
    return mkU<BVHNode>(*this);
}


BVHBounds::BVHBounds()
    : BVHBounds(glm::vec3(std::numeric_limits<float>::max()),
                glm::vec3(std::numeric_limits<float>::lowest()))
{}

BVHBounds::BVHBounds(const glm::vec3 &min, const glm::vec3 &max)
    : minCorner(min), maxCorner(max)
{}


BVHBounds::BVHBounds(const std::vector<glm::vec3> &verts)
    : BVHBounds()
{
    for(const auto &p : verts) {
        minCorner = glm::min(minCorner, p);
        maxCorner = glm::max(maxCorner, p);
    }
}

int BVHNode::flattenBVHTree(int *offset, std::vector<LinearBVHNode> &nodes, std::vector<TriangleVerts> &t) const {
    LinearBVHNode& linearNode = nodes[*offset];
    linearNode.bounds = this->bbox;
    int myOffset = (*offset)++;
    if (this->triangle) {
        linearNode.triangle_idx = t.size();
        t.push_back(*(this->triangle));
    } else {
        this->child_L->flattenBVHTree(offset, nodes, t);
        linearNode.secondChildOffset = this->child_R->flattenBVHTree(offset, nodes, t);
    }
    return myOffset;
}

std::pair<std::vector<LinearBVHNode>, std::vector<TriangleVerts>> recursiveFlattenBVHTree(BVHNode* root, int totalNodes) {
    std::vector<LinearBVHNode> nodes;
    std::vector<TriangleVerts> t;
    nodes.reserve(totalNodes);
    for (int i = 0; i < totalNodes; i++) {
        nodes.push_back(LinearBVHNode());
    }
    int offset = 0;
    root->flattenBVHTree(&offset, nodes, t);
    std::pair<std::vector<LinearBVHNode>, std::vector<TriangleVerts>> result(nodes, t);
    return result;
}

// std::optional<Intersection> intersectLinearBVHTree(const Ray &ray, const std::vector<uPtr<LinearBVHNode>>& nodes) {
//     std::optional<Intersection> isect = std::nullopt;
//     std::queue<int> nodesToVisit;
//     nodesToVisit.push(0);
//     while (!nodesToVisit.empty()) {
//         int currentNodeIndex = nodesToVisit.front();
//         nodesToVisit.pop();
//         const LinearBVHNode* node = nodes[currentNodeIndex].get();

//         std::optional<float> boxDist = node->bounds.intersect(ray);
//         if (boxDist && (!isect || *boxDist < isect->t)) {
//             if (node->primitive) {
//                 std::optional<Intersection> currIsect = node->primitive->intersect(ray);
//                 if (currIsect && (!isect || currIsect->t < isect->t)) {
//                     isect = currIsect;
//                 }
//             } else {
//                 nodesToVisit.push(currentNodeIndex + 1);
//                 nodesToVisit.push(node->secondChildOffset);
//             }
//         }
//     }
//     return isect;
// }
