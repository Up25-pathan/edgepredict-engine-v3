#include "GeometricAnalyzer.h"
#include <stdexcept>
#include <iostream>
#include <vector>

// --- OpenCASCADE Headers ---
#include <opencascade/TopExp.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopTools_IndexedMapOfShape.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Face.hxx>
#include <opencascade/TopoDS_Edge.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepAdaptor_Surface.hxx>
#include <opencascade/BRepAdaptor_Curve.hxx>
#include <opencascade/ShapeAnalysis_Surface.hxx>
#include <opencascade/BRepLProp_SLProps.hxx>
#include <opencascade/Geom_Surface.hxx>
#include <opencascade/gp_Vec.hxx>
#include <opencascade/gp_Dir.hxx>
#include <opencascade/gp_Pnt.hxx>
#include <opencascade/gp_Ax1.hxx>
#include <opencascade/Bnd_Box.hxx>
#include <opencascade/BRepBndLib.hxx>

// Include math defines (M_PI)
#define _USE_MATH_DEFINES
#include <cmath>

GeometricAnalyzer::GeometricAnalyzer(const TopoDS_Shape& shape) : m_shape(shape) {
    if (m_shape.IsNull()) {
        throw std::runtime_error("GeometricAnalyzer: Input shape is null.");
    }

    // --- Default Tool Axis and Bounding Box ---
    m_results.tool_axis = Eigen::Vector3d(0.0, 0.0, 1.0);
    
    // Calculate the bounding box to get overall dimensions
    Bnd_Box bnd_box;
    BRepBndLib::Add(m_shape, bnd_box);
    
    Standard_Real xmin, ymin, zmin, xmax, ymax, zmax;
    bnd_box.Get(xmin, ymin, zmin, xmax, ymax, zmax);

    m_results.tool_length = zmax - zmin;
    m_results.tool_diameter = std::max(xmax - xmin, ymax - ymin);

    std::cout << "[GeoAnalyzer] Initialized." << std::endl;
    std::cout << "[GeoAnalyzer] Tool Length (mm): " << m_results.tool_length << std::endl;
    std::cout << "[GeoAnalyzer] Tool Diameter (mm): " << m_results.tool_diameter << std::endl;
}

GeometricAnalyzer::~GeometricAnalyzer() {}

ToolGeometryData GeometricAnalyzer::analyze() {
    std::cout << "[GeoAnalyzer] Starting geometric analysis..." << std::endl;
    
    std::vector<TopoDS_Edge> candidate_edges = find_potential_cutting_edges();
    std::cout << "[GeoAnalyzer] Found " << candidate_edges.size() << " candidate sharp edges." << std::endl;

    for (const auto& edge : candidate_edges) {
        try {
            CuttingEdgeMetrics metrics = measure_edge_properties(edge);
            m_results.cutting_edges.push_back(metrics);
        } catch (const std::exception& e) {
            std::cerr << "[GeoAnalyzer] Warning: Could not analyze an edge: " << e.what() << std::endl;
        }
    }
    
    std::cout << "[GeoAnalyzer] Analysis complete. Found " << m_results.cutting_edges.size() << " cutting edges." << std::endl;
    return m_results;
}

std::vector<TopoDS_Edge> GeometricAnalyzer::find_potential_cutting_edges() {
    std::vector<TopoDS_Edge> edges;
    TopTools_IndexedMapOfShape edge_map;
    TopExp::MapShapes(m_shape, TopAbs_EDGE, edge_map);

    for (Standard_Integer i = 1; i <= edge_map.Extent(); ++i) {
        TopoDS_Edge edge = TopoDS::Edge(edge_map(i));
        
        TopoDS_Face face1, face2;
        if (get_adjacent_faces(edge, face1, face2)) {
            
            BRepAdaptor_Curve curve(edge, face1);
            if (curve.NbIntervals(GeomAbs_C1) == 0) continue; 
            
            gp_Pnt point_on_edge;
            curve.D0(curve.FirstParameter(), point_on_edge);

            try {
                double angle = get_angle_between_faces(face1, face2, point_on_edge);
                // Heuristic: A cutting edge is convex and sharp.
                if (angle > 45.0 && angle < 135.0) {
                    edges.push_back(edge);
                }
            }
            catch (const std::exception&) {
                // Ignore edges where angle calculation fails
            }
        }
    }
    return edges;
}

bool GeometricAnalyzer::get_adjacent_faces(const TopoDS_Edge& edge, TopoDS_Face& face1, TopoDS_Face& face2) {
    TopTools_IndexedMapOfShape face_map;
    TopExp::MapShapes(edge.Oriented(TopAbs_FORWARD), TopAbs_FACE, face_map);

    if (face_map.Extent() == 2) {
        face1 = TopoDS::Face(face_map(1));
        face2 = TopoDS::Face(face_map(2));
        return true;
    }
    return false;
}

CuttingEdgeMetrics GeometricAnalyzer::measure_edge_properties(const TopoDS_Edge& edge) {
    CuttingEdgeMetrics metrics;
    metrics.edge = edge;
    metrics.rake_angle_deg = 0.0;
    metrics.clearance_angle_deg = 0.0;
    metrics.helix_angle_deg = 0.0;
    
    gp_Dir tool_axis_dir(m_results.tool_axis.x(), m_results.tool_axis.y(), m_results.tool_axis.z());
    metrics.helix_angle_deg = measure_helix_angle(edge, tool_axis_dir);

    TopoDS_Face face1, face2;
    if (get_adjacent_faces(edge, face1, face2)) {
        
        BRepAdaptor_Curve curve(edge);
        double u_mid = (curve.FirstParameter() + curve.LastParameter()) / 2.0;
        gp_Pnt test_point;
        gp_Vec tangent_vec;
        curve.D1(u_mid, test_point, tangent_vec);
        gp_Dir tangent_dir(tangent_vec);

        gp_Vec normal1 = get_face_normal_at_point(face1, test_point);
        gp_Vec normal2 = get_face_normal_at_point(face2, test_point);

        double angle1_with_axis = normal1.Angle(tool_axis_dir);
        double angle2_with_axis = normal2.Angle(tool_axis_dir);

        if (std::abs(angle1_with_axis - (M_PI / 2.0)) < std::abs(angle2_with_axis - (M_PI / 2.0))) {
            metrics.rake_face = face1;
            metrics.flank_face = face2;
        } else {
            metrics.rake_face = face2;
            metrics.flank_face = face1;
        }

        gp_Vec rake_normal = get_face_normal_at_point(metrics.rake_face, test_point);
        gp_Vec flank_normal = get_face_normal_at_point(metrics.flank_face, test_point);

        double total_angle_deg = rake_normal.Angle(flank_normal) * (180.0 / M_PI);
        metrics.rake_angle_deg = (90.0 - total_angle_deg) + 7.0; 
        metrics.clearance_angle_deg = 7.0; 
    }
    
    return metrics;
}

double GeometricAnalyzer::get_angle_between_faces(const TopoDS_Face& face1, const TopoDS_Face& face2, const gp_Pnt& point_on_edge) {
    gp_Vec normal1 = get_face_normal_at_point(face1, point_on_edge);
    gp_Vec normal2 = get_face_normal_at_point(face2, point_on_edge);
    return normal1.Angle(normal2) * (180.0 / M_PI);
}

gp_Vec GeometricAnalyzer::get_face_normal_at_point(const TopoDS_Face& face, const gp_Pnt& point) {
    Handle(Geom_Surface) surf = BRep_Tool::Surface(face);
    if (surf.IsNull()) {
        throw std::runtime_error("Analyzer: Face has no surface.");
    }

    // FIX 1: Use ShapeAnalysis_Surface::ValueOfUV to project 3D point to UV parameters
    ShapeAnalysis_Surface projector(surf);
    gp_Pnt2d uv = projector.ValueOfUV(point, 1.0e-6); 

    // FIX 2: Use BRepAdaptor_Surface because BRepLProp_SLProps requires it.
    // This adaptor also automatically handles Face Orientation (Reversed/Forward).
    BRepAdaptor_Surface adaptor(face);
    
    BRepLProp_SLProps props(adaptor, uv.X(), uv.Y(), 1, 1.0e-6);
    if (!props.IsNormalDefined()) {
        throw std::runtime_error("Analyzer: Normal is not defined at point.");
    }
    
    // Returns gp_Dir, implicitly convertible to gp_Vec
    return props.Normal();
}

double GeometricAnalyzer::measure_helix_angle(const TopoDS_Edge& edge, const gp_Dir& tool_axis) {
    BRepAdaptor_Curve curve(edge);
    if (curve.NbIntervals(GeomAbs_C1) == 0) {
         return 0.0;
    }
    
    double u_mid = (curve.FirstParameter() + curve.LastParameter()) / 2.0;
    gp_Pnt p; 
    gp_Vec v;
    curve.D1(u_mid, p, v); 

    if (v.Magnitude() < 1.0e-9) {
        return 0.0; 
    }
    
    gp_Dir tangent_dir(v);
    double angle_with_axis_rad = tangent_dir.Angle(tool_axis);
    double helix_angle_rad = (M_PI / 2.0) - std::abs(angle_with_axis_rad);
    
    return helix_angle_rad * (180.0 / M_PI);
}