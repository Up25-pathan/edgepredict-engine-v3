#pragma once
/**
 * @file GeometryLoader.h
 * @brief Geometry loading for STL and CAD files (STEP/IGES)
 */

#include "Types.h"
#include "Config.h"
#include <string>
#include <memory>

// Forward declare OpenCASCADE types to avoid heavy includes
class TopoDS_Shape;

namespace edgepredict {

/**
 * @brief Loads geometry from various file formats into Mesh structure
 */
class GeometryLoader {
public:
    GeometryLoader();
    ~GeometryLoader();
    
    /**
     * @brief Load mesh from file (auto-detect format)
     * @param path Path to geometry file
     * @param mesh Output mesh
     * @return true if loading succeeded
     */
    bool load(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from STL file
     */
    bool loadSTL(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from STEP file
     */
    bool loadSTEP(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Load mesh from IGES file
     */
    bool loadIGES(const std::string& path, Mesh& mesh);
    
    /**
     * @brief Apply scale factor to mesh
     * @param mesh Mesh to scale
     * @param scale Scale factor (e.g., 0.001 for mm to m)
     */
    static void scaleMesh(Mesh& mesh, double scale);
    
    /**
     * @brief Compute mesh normals
     */
    static void computeNormals(Mesh& mesh);
    
    /**
     * @brief Compute mesh bounding box
     */
    static void getBoundingBox(const Mesh& mesh, Vec3& minCorner, Vec3& maxCorner);
    
    /**
     * @brief Get last error message
     */
    const std::string& getLastError() const { return m_lastError; }

private:
    bool detectFormat(const std::string& path, std::string& format);
    bool meshFromCADShape(const TopoDS_Shape& shape, Mesh& mesh, double deflection);
    
    std::string m_lastError;
    double m_defaultDeflection = 0.05;  // Tessellation tolerance (mm)
};

} // namespace edgepredict
