#include "sceneStructs.h"
#include <memory>

#define USE_SAH 1

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

float BVHBounds::surfaceArea() const {
    glm::vec3 d = maxCorner - minCorner;
    return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

#if USE_SAH

BVHBounds computeCentroidBounds(std::vector<TriangleVerts> &triangles, int start, int end) {
    BVHBounds centroidBounds;
    for (int i = start; i < end; i++) {
        glm::vec3 c = getCentroid(triangles[i]);
        centroidBounds = Union(centroidBounds, BVHBounds(c, c));
    }
    return centroidBounds;
}

int bucketSAHSplit(std::vector<TriangleVerts> &triangles, int start, int end,
                   int axis, const BVHBounds &bounds, const BVHBounds &centroidBounds) {
    constexpr int nBuckets = 12;
    struct BucketInfo {
        int count = 0;
        BVHBounds bounds;
    };
    BucketInfo buckets[nBuckets];

    for (int i = start; i < end; i++) {
        int b = nBuckets * centroidBounds.Offset(getCentroid(triangles[i]))[axis];
        if (b == nBuckets) b = nBuckets - 1;
        buckets[b].count++;
        buckets[b].bounds = Union(buckets[b].bounds, getTriangleBounds(triangles[i]));
    }

    float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; i++) {
        BVHBounds b0, b1;
        int count0 = 0, count1 = 0;
        for (int j = 0; j <= i; j++) {
            b0 = Union(b0, buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for (int j = i + 1; j < nBuckets; j++) {
            b1 = Union(b1, buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] = 0.125f + (count0 * b0.surfaceArea() + count1 * b1.surfaceArea()) / bounds.surfaceArea();
    }

    int minCostSplitBucket = 0;
    float minCost = cost[0];
    for (int i = 1; i < nBuckets - 1; i++) {
        if (cost[i] < minCost) {
            minCost = cost[i];
            minCostSplitBucket = i;
        }
    }

    auto pmid = std::partition(triangles.begin() + start, triangles.begin() + end,
                               [&](const TriangleVerts& tri) {
                                   int b = nBuckets * centroidBounds.Offset(getCentroid(tri))[axis];
                                   if (b == nBuckets) b = nBuckets - 1;
                                   return b <= minCostSplitBucket;
                               });
    int mid = pmid - triangles.begin();

    if (mid == start || mid == end) {
        mid = (start + end) / 2;
    }

    return mid;
}

#endif


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
#if USE_SAH
            BVHBounds centroidBounds = computeCentroidBounds(triangleInfo, start, end);
            glm::vec3 centroidExtent = centroidBounds.maxCorner - centroidBounds.minCorner;

            if (centroidExtent[axisToExpand] == 0) {
                mid = (start + end) / 2;
            } else {
                mid = bucketSAHSplit(triangleInfo, start, end, axisToExpand, currentLayerBounds, centroidBounds);
            }
#else
            std::sort(triangleInfo.begin() + start, triangleInfo.begin() + end,
                        [axisToExpand](const TriangleVerts& a, const TriangleVerts& b) {
                return getCentroid(a)[axisToExpand] < getCentroid(b)[axisToExpand];
            });
            mid = (start + end) / 2;
#endif
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
