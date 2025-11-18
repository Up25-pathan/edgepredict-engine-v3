#pragma once

#include "simulation.h" // We need Node and Mesh definitions
#include <vector>
#include <map>

// --- OpenCASCADE Headers ---
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopAbs_ShapeEnum.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Edge.hxx>
#include <opencascade/TopoDS_Face.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepAdaptor_Surface.hxx>
#include <opencascade/gp_Pnt.hxx>
#include <opencascade/gp_Dir.hxx>
#include <opencascade/GCPnts_AbscissaPoint.hxx>
#include <opencascade/BRepAdaptor_Curve.hxx>
#include <opencascade/Geom_Surface.hxx>

// --- Data structure to hold the measured geometry ---
// This is the "smart" data that replaces the "dumb" config values.

struct CuttingEdgeMetrics {
    double rake_angle_deg;     // Measured rake angle
    double clearance_angle_deg; // Measured clearance angle
    double helix_angle_deg;    // Measured helix angle
    TopoDS_Edge edge;           // The actual CAD edge
    TopoDS_Face rake_face;
    TopoDS_Face flank_face;
};

struct ToolGeometryData {
    std::vector<CuttingEdgeMetrics> cutting_edges;
    Eigen::Vector3d tool_axis;   // Assumed axis of rotation (e.g., Z-axis)
    double tool_diameter;
    double tool_length;
    // ... other global geometry properties
};


class GeometricAnalyzer {
public:
    // Constructor takes the 3D model
    GeometricAnalyzer(const TopoDS_Shape& shape);

    // The main function that does all the R&D work
    ToolGeometryData analyze();

private:
    // --- Internal Helper Functions ---

    // 1. Find all "sharp" edges in the model
    std::vector<TopoDS_Edge> find_potential_cutting_edges();

    // 2. For a sharp edge, find its two adjacent faces
    bool get_adjacent_faces(const TopoDS_Edge& edge, TopoDS_Face& face1, TopoDS_Face& face2);

    // 3. The core logic: measure the angles between the faces
    CuttingEdgeMetrics measure_edge_properties(const TopoDS_Edge& edge);

    // 4. Helper to get the angle between two surfaces at a specific point
    double get_angle_between_faces(const TopoDS_Face& face1, const TopoDS_Face& face2, const gp_Pnt& point_on_edge);

    // 5. Helper to measure the helix angle of a 3D edge curve
    double measure_helix_angle(const TopoDS_Edge& edge, const gp_Dir& tool_axis);


    // --- Member Variables ---
    const TopoDS_Shape& m_shape;
    ToolGeometryData m_results;
};