/**
 * @file GeometryLoader.cpp
 * @brief Geometry loading implementation
 */

#include "GeometryLoader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <unordered_map>

// OpenCASCADE includes for CAD loading (optional)
#ifdef HAS_OPENCASCADE
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>
#endif

namespace edgepredict {

GeometryLoader::GeometryLoader() = default;
GeometryLoader::~GeometryLoader() = default;

bool GeometryLoader::detectFormat(const std::string& path, std::string& format) {
    size_t dotPos = path.rfind('.');
    if (dotPos == std::string::npos) {
        m_lastError = "No file extension found";
        return false;
    }
    
    format = path.substr(dotPos + 1);
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    return true;
}

bool GeometryLoader::load(const std::string& path, Mesh& mesh) {
    std::string format;
    if (!detectFormat(path, format)) {
        return false;
    }
    
    std::cout << "[GeometryLoader] Loading: " << path << " (format: " << format << ")" << std::endl;
    
    if (format == "stl") {
        return loadSTL(path, mesh);
    } else if (format == "step" || format == "stp") {
#ifdef HAS_OPENCASCADE
        return loadSTEP(path, mesh);
#else
        m_lastError = "STEP loading requires OpenCASCADE (not installed). Use STL format instead.";
        std::cerr << "[GeometryLoader] " << m_lastError << std::endl;
        return false;
#endif
    } else if (format == "iges" || format == "igs") {
#ifdef HAS_OPENCASCADE
        return loadIGES(path, mesh);
#else
        m_lastError = "IGES loading requires OpenCASCADE (not installed). Use STL format instead.";
        std::cerr << "[GeometryLoader] " << m_lastError << std::endl;
        return false;
#endif
    } else {
        m_lastError = "Unsupported format: " + format;
        return false;
    }
}

bool GeometryLoader::loadSTL(const std::string& path, Mesh& mesh) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        m_lastError = "Cannot open file: " + path;
        return false;
    }
    
    mesh.clear();
    
    // Check if binary or ASCII STL
    char header[80];
    file.read(header, 80);
    
    // Read number of triangles (binary STL)
    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), sizeof(uint32_t));
    
    // Check if this looks like binary (file size should match)
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    size_t expectedBinarySize = 84 + numTriangles * 50;
    
    bool isBinary = (fileSize == expectedBinarySize);
    file.seekg(0, std::ios::beg);
    
    if (isBinary) {
        // Binary STL
        file.seekg(84);  // Skip header + triangle count
        
        std::unordered_map<size_t, int> vertexMap;
        
        for (uint32_t i = 0; i < numTriangles; ++i) {
            // Read normal (not used, we compute our own)
            float normal[3];
            file.read(reinterpret_cast<char*>(normal), 12);
            
            Triangle tri;
            
            // Read 3 vertices
            for (int v = 0; v < 3; ++v) {
                float vertex[3];
                file.read(reinterpret_cast<char*>(vertex), 12);
                
                // Create vertex hash for deduplication
                Vec3 pos(vertex[0], vertex[1], vertex[2]);
                size_t hash = std::hash<double>{}(pos.x) ^ 
                             (std::hash<double>{}(pos.y) << 1) ^
                             (std::hash<double>{}(pos.z) << 2);
                
                auto it = vertexMap.find(hash);
                if (it != vertexMap.end()) {
                    tri.indices[v] = it->second;
                } else {
                    int idx = static_cast<int>(mesh.nodes.size());
                    FEMNode node;
                    node.position = pos;
                    node.originalPosition = pos;
                    mesh.nodes.push_back(node);
                    vertexMap[hash] = idx;
                    tri.indices[v] = idx;
                }
            }
            
            mesh.triangles.push_back(tri);
            
            // Skip attribute byte count
            uint16_t attr;
            file.read(reinterpret_cast<char*>(&attr), 2);
        }
    } else {
        // ASCII STL - simple parser
        file.seekg(0);
        std::string line;
        
        std::unordered_map<size_t, int> vertexMap;
        Triangle currentTri;
        int vertexIdx = 0;
        
        while (std::getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            
            if (line.rfind("vertex", 0) == 0) {
                float x, y, z;
                if (sscanf(line.c_str(), "vertex %f %f %f", &x, &y, &z) == 3) {
                    Vec3 pos(x, y, z);
                    size_t hash = std::hash<double>{}(pos.x) ^ 
                                 (std::hash<double>{}(pos.y) << 1) ^
                                 (std::hash<double>{}(pos.z) << 2);
                    
                    auto it = vertexMap.find(hash);
                    if (it != vertexMap.end()) {
                        currentTri.indices[vertexIdx] = it->second;
                    } else {
                        int idx = static_cast<int>(mesh.nodes.size());
                        FEMNode node;
                        node.position = pos;
                        node.originalPosition = pos;
                        mesh.nodes.push_back(node);
                        vertexMap[hash] = idx;
                        currentTri.indices[vertexIdx] = idx;
                    }
                    
                    vertexIdx++;
                    if (vertexIdx == 3) {
                        mesh.triangles.push_back(currentTri);
                        vertexIdx = 0;
                    }
                }
            }
        }
    }
    
    // Compute normals
    computeNormals(mesh);
    
    std::cout << "[GeometryLoader] Loaded STL: " << mesh.nodeCount() << " nodes, " 
              << mesh.triangleCount() << " triangles" << std::endl;
    
    return true;
}

bool GeometryLoader::loadSTEP(const std::string& path, Mesh& mesh) {
#ifdef HAS_OPENCASCADE
    STEPControl_Reader reader;
    
    IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
        m_lastError = "Failed to read STEP file: " + path;
        return false;
    }
    
    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    
    if (shape.IsNull()) {
        m_lastError = "STEP file produced null shape";
        return false;
    }
    
    return meshFromCADShape(shape, mesh, m_defaultDeflection);
#else
    (void)path; (void)mesh;
    m_lastError = "STEP loading requires OpenCASCADE";
    return false;
#endif
}

bool GeometryLoader::loadIGES(const std::string& path, Mesh& mesh) {
#ifdef HAS_OPENCASCADE
    IGESControl_Reader reader;
    
    IFSelect_ReturnStatus status = reader.ReadFile(path.c_str());
    if (status != IFSelect_RetDone) {
        m_lastError = "Failed to read IGES file: " + path;
        return false;
    }
    
    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    
    if (shape.IsNull()) {
        m_lastError = "IGES file produced null shape";
        return false;
    }
    
    return meshFromCADShape(shape, mesh, m_defaultDeflection);
#else
    (void)path; (void)mesh;
    m_lastError = "IGES loading requires OpenCASCADE";
    return false;
#endif
}

#ifdef HAS_OPENCASCADE
bool GeometryLoader::meshFromCADShape(const TopoDS_Shape& shape, Mesh& mesh, double deflection) {
    mesh.clear();
    
    // Tessellate the shape
    BRepMesh_IncrementalMesh(shape, deflection);
    
    // Map to deduplicate vertices
    std::unordered_map<size_t, int> vertexMap;
    
    // Extract triangles from all faces
    for (TopExp_Explorer explorer(shape, TopAbs_FACE); explorer.More(); explorer.Next()) {
        TopLoc_Location location;
        Handle(Poly_Triangulation) triangulation = 
            BRep_Tool::Triangulation(TopoDS::Face(explorer.Current()), location);
        
        if (triangulation.IsNull()) continue;
        
        for (int i = 1; i <= triangulation->NbTriangles(); ++i) {
            Standard_Integer n1, n2, n3;
            triangulation->Triangle(i).Get(n1, n2, n3);
            
            Triangle tri;
            Standard_Integer nodeIndices[3] = {n1, n2, n3};
            
            for (int v = 0; v < 3; ++v) {
                gp_Pnt point = triangulation->Node(nodeIndices[v]).Transformed(location);
                Vec3 pos(point.X(), point.Y(), point.Z());
                
                // Hash for deduplication
                size_t hash = std::hash<double>{}(pos.x * 1000) ^ 
                             (std::hash<double>{}(pos.y * 1000) << 1) ^
                             (std::hash<double>{}(pos.z * 1000) << 2);
                
                auto it = vertexMap.find(hash);
                if (it != vertexMap.end()) {
                    tri.indices[v] = it->second;
                } else {
                    int idx = static_cast<int>(mesh.nodes.size());
                    FEMNode node;
                    node.position = pos;
                    node.originalPosition = pos;
                    mesh.nodes.push_back(node);
                    vertexMap[hash] = idx;
                    tri.indices[v] = idx;
                }
            }
            
            mesh.triangles.push_back(tri);
        }
    }
    
    computeNormals(mesh);
    
    std::cout << "[GeometryLoader] Loaded CAD: " << mesh.nodeCount() << " nodes, " 
              << mesh.triangleCount() << " triangles" << std::endl;
    
    return true;
}
#endif

void GeometryLoader::scaleMesh(Mesh& mesh, double scale) {
    for (auto& node : mesh.nodes) {
        node.position = node.position * scale;
        node.originalPosition = node.originalPosition * scale;
    }
}

void GeometryLoader::computeNormals(Mesh& mesh) {
    for (auto& tri : mesh.triangles) {
        if (tri.indices[0] >= 0 && tri.indices[0] < static_cast<int>(mesh.nodes.size()) &&
            tri.indices[1] >= 0 && tri.indices[1] < static_cast<int>(mesh.nodes.size()) &&
            tri.indices[2] >= 0 && tri.indices[2] < static_cast<int>(mesh.nodes.size())) {
            
            const Vec3& v0 = mesh.nodes[tri.indices[0]].position;
            const Vec3& v1 = mesh.nodes[tri.indices[1]].position;
            const Vec3& v2 = mesh.nodes[tri.indices[2]].position;
            
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            tri.normal = edge1.cross(edge2).normalized();
        }
    }
}

void GeometryLoader::getBoundingBox(const Mesh& mesh, Vec3& minCorner, Vec3& maxCorner) {
    if (mesh.nodes.empty()) {
        minCorner = Vec3::zero();
        maxCorner = Vec3::zero();
        return;
    }
    
    minCorner = mesh.nodes[0].position;
    maxCorner = mesh.nodes[0].position;
    
    for (const auto& node : mesh.nodes) {
        minCorner.x = std::min(minCorner.x, node.position.x);
        minCorner.y = std::min(minCorner.y, node.position.y);
        minCorner.z = std::min(minCorner.z, node.position.z);
        maxCorner.x = std::max(maxCorner.x, node.position.x);
        maxCorner.y = std::max(maxCorner.y, node.position.y);
        maxCorner.z = std::max(maxCorner.z, node.position.z);
    }
}

} // namespace edgepredict
