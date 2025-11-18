#define _USE_MATH_DEFINES
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip> // For std::setw, std::setprecision
#include <string>
#include <map>
#include <random>
#include <memory> // For std::unique_ptr, std::make_unique

// --- NEW INCLUDES FOR THE STRATEGY PATTERN ---
#include "IPhysicsStrategy.h"
#include "TurningStrategy.h"   // Our "legacy" engine
#include "MillingStrategy.h"   // Our "R&D" engine
// --- END NEW INCLUDES ---

// OpenCASCADE includes for loading
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Face.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>
#include <opencascade/TColgp_Array1OfPnt.hxx>
#include <opencascade/Poly_Array1OfTriangle.hxx>
#include <opencascade/gp_Pnt.hxx>
#include <opencascade/TopLoc_Location.hxx>

#include "stl_reader.h" //

// ═══════════════════════════════════════════════════════════
// SIMULATION CLASS (NOW A STRATEGY FACTORY)
// ═══════════════════════════════════════════════════════════

Simulation::Simulation() : predicted_tool_life_hours(0.0), wear_threshold_mm(0.3) {
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        load_config("input.json"); //
        
        // LoadGeometry is now split:
        // 1. Load the raw CAD shape
        // 2. Load the render mesh
        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR in Simulation Constructor: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

Simulation::~Simulation() = default;

void Simulation::run() {
    // --- THIS IS THE NEW "SMART" ENGINE LOGIC ---
    std::cout << "EdgePredict Engine v4.0 - Virtual Lab" << std::endl;

    // 1. Get simulation type from config
    //    Default to "turning" if not specified, to maintain backward compatibility
    std::string sim_type = config_data.value("machining_type", "turning");
    
    // 2. The "Strategy Factory"
    //    We create the correct "machine" for the job.
    if (sim_type == "turning") {
        std::cout << "Loading 'Turning' physics strategy (Legacy Linear Engine)." << std::endl;
        m_strategy = std::make_unique<TurningStrategy>(this);
    } else if (sim_type == "milling" || sim_type == "drilling") {
        std::cout << "Loading 'Milling/Drilling' physics strategy (R&D Rotational Engine)." << std::endl;
        m_strategy = std::make_unique<MillingStrategy>(this);
    } else {
        throw std::runtime_error("Unknown 'machining_type' in input.json: " + sim_type);
    }

    // 3. Initialize the chosen strategy
    //    (This is where the R&D engine will use GeometricAnalyzer)
    m_strategy->initialize(mesh, cad_shape, config_data);

    // 4. Run the simulation loop
    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    double dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    
    std::cout << "Simulation starting (" << num_steps << " steps)..." << std::endl;
    for (int step = 1; step <= num_steps; ++step) {
        // Delegate all physics work to the strategy
        json step_metrics = m_strategy->run_time_step(dt, step);
        
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        
        // Store the metrics from the strategy
        time_series_output.push_back(step_metrics);
        
        // Report progress (this is unchanged from your original)
        if (step % 10 == 0 || step == 1) {
             write_progress(step, num_steps, step_metrics);
        }
    }
    std::cout << "Simulation loop finished." << std::endl;

    // 5. Get final results from the strategy
    json final_results = m_strategy->get_final_results();
    cfd_visualization_data = m_strategy->get_visualization_data();

    // 6. Post-process and write output
    calculate_tool_life_prediction(final_results);
    write_output(final_results);
    
    std::cout << "Simulation complete. Output written to " 
              << config_data["file_paths"]["output_results"].get<std::string>() << std::endl;
}

void Simulation::write_progress(int step, int total_steps, const json& current_metrics) {
    // This function is for the worker/backend to read stdout
    std::cout << "  Step " << std::setw(4) << step << "/" << total_steps
              << " | T_max: " << std::fixed << std::setprecision(0) << std::setw(4) 
              << current_metrics.value("max_temperature_C", 0.0) << "°C"
              << " | Wear: " << std::scientific << std::setprecision(2) 
              << current_metrics.value("total_accumulated_wear_m", 0.0) << " m" << std::endl;
}

void Simulation::load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open config file: " + path);
    config_data = json::parse(f);
    std::cout << "Configuration file loaded from " << path << std::endl;
}

std::string Simulation::get_file_ext(const std::string& s) {
    size_t i = s.rfind('.', s.length());
    return (i != std::string::npos) ? s.substr(i + 1) : "";
}

void Simulation::load_geometry() {
    std::string path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::string ext = get_file_ext(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    std::cout << "Loading geometry: " << path << std::endl;
    
    if (ext == "stl") {
        load_geometry_stl(path);
        // We don't have a CAD shape for STL, so we create an empty one
        cad_shape = TopoDS_Shape();
    } else if (ext == "step" || ext == "stp" || ext == "iges" || ext == "igs") {
        // 1. Load the raw CAD shape
        cad_shape = load_geometry_cad(path, ext.substr(0, 4));
        
        // 2. Create the visual mesh from the CAD shape for rendering/FEA
        create_mesh_from_cad(cad_shape);
    } else {
        throw std::runtime_error("Unknown geometry format: " + ext);
    }

    // Scale the *visual mesh* to meters
    // (We assume CAD files are in mm)
    for (auto& n : mesh.nodes) n.position *= 0.001;
    
    std::cout << "Geometry loaded: " << mesh.nodes.size() << " nodes and " 
              << mesh.triangle_indices.size() / 3 << " triangles." << std::endl;
}

void Simulation::load_geometry_stl(const std::string& path) {
    // This logic is unchanged from your original file
    stl_reader::StlMesh<float, unsigned int> stl_mesh;
    try {
         stl_mesh.read_file(path.c_str());
    } catch(const std::exception& e) {
        throw std::runtime_error("Failed to read STL file: " + std::string(e.what()));
    }

    double ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    mesh.nodes.clear();
    mesh.triangle_indices.clear();

    for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
        const float* p = stl_mesh.vrt_coords(i);
        mesh.nodes.push_back({Eigen::Vector3d(p[0], p[1], p[2]), ambient_temp, 0, 0, 1e-6, 0, 0, NodeStatus::OK});
    }

    for (size_t i = 0; i < stl_mesh.num_tris(); ++i) {
        const unsigned int* t = stl_mesh.tri_corner_inds(i);
        mesh.triangle_indices.push_back(t[0]);
        mesh.triangle_indices.push_back(t[1]);
        mesh.triangle_indices.push_back(t[2]);
    }
}

TopoDS_Shape Simulation::load_geometry_cad(const std::string& path, const std::string& format) {
    TopoDS_Shape shape;
    
    if (format.find("step") != std::string::npos || format.find("stp") != std::string::npos) {
        STEPControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone) {
            throw std::runtime_error("Failed to read STEP file: " + path);
        }
        r.TransferRoots(); 
        shape = r.OneShape();
    } else if (format.find("iges") != std::string::npos || format.find("igs") != std::string::npos) {
        IGESControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone) {
            throw std::runtime_error("Failed to read IGES file: " + path);
        }
        r.TransferRoots(); 
        shape = r.OneShape();
    } else {
        throw std::runtime_error("Unsupported CAD format: " + format);
    }
    
    if (shape.IsNull()) {
        throw std::runtime_error("CAD file loaded but shape is null.");
    }
    
    return shape; // Return the raw 3D shape
}

void Simulation::create_mesh_from_cad(const TopoDS_Shape& shape) {
    // This is the meshing logic moved from your old load_geometry_cad
    // 0.05 is the mesh deflection in mm. We can make this a config parameter.
    double mesh_deflection = config_data.value("mesh_deflection_mm", 0.05);
    BRepMesh_IncrementalMesh(shape, mesh_deflection);
    
    mesh.nodes.clear();
    mesh.triangle_indices.clear();

    std::map<std::array<double,3>, int> node_map;
    double ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();

    // Iterate over all faces in the shape
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopLoc_Location L;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), L);
        
        if (triangulation.IsNull()) continue;

        const TColgp_Array1OfPnt& nodes = triangulation->Nodes();
        const Poly_Array1OfTriangle& triangles = triangulation->Triangles();

        // Add nodes to our mesh structure, using a map to avoid duplicates
        for (int i = 1; i <= triangulation->NbNodes(); ++i) {
            gp_Pnt p = nodes(i).Transformed(L);
            std::array<double,3> coords = {p.X(), p.Y(), p.Z()};
            if (node_map.find(coords) == node_map.end()) {
                node_map[coords] = (int)mesh.nodes.size();
                mesh.nodes.push_back({Eigen::Vector3d(coords[0], coords[1], coords[2]), ambient_temp, 0, 0, 1e-6, 0, 0, NodeStatus::OK});
            }
        }

        // Add triangle indices
        for (int i = 1; i <= triangulation->NbTriangles(); ++i) {
            int n1, n2, n3;
            triangulation->Triangle(i).Get(n1, n2, n3);
            
            // Get the 3D points for these node indices
            gp_Pnt p1 = nodes(n1).Transformed(L);
            gp_Pnt p2 = nodes(n2).Transformed(L);
            gp_Pnt p3 = nodes(n3).Transformed(L);

            // Find the corresponding index in our global mesh.nodes vector
            mesh.triangle_indices.push_back(node_map[{p1.X(), p1.Y(), p1.Z()}]);
            mesh.triangle_indices.push_back(node_map[{p2.X(), p2.Y(), p2.Z()}]);
            mesh.triangle_indices.push_back(node_map[{p3.X(), p3.Y(), p3.Z()}]);
        }
    }
}

void Simulation::calculate_tool_life_prediction(const json& final_results) {
    if (!final_results.contains("total_accumulated_wear_m") || !final_results.contains("total_time_s")) {
        predicted_tool_life_hours = -1.0; // Indicate error
        std::cerr << "Warning: Could not calculate tool life. Final results missing." << std::endl;
        return;
    }
    
    double wear_m = final_results["total_accumulated_wear_m"].get<double>();
    double time_s = final_results["total_time_s"].get<double>();
    
    // Use the wear threshold from config, default to 0.3mm
    wear_threshold_mm = config_data.value("wear_threshold_mm", 0.3);
    double wear_threshold_m = wear_threshold_mm / 1000.0;

    if (wear_m > 1e-12 && time_s > 1e-9) { // Avoid division by zero
        double wear_rate_m_per_s = wear_m / time_s;
        double time_to_failure_s = wear_threshold_m / wear_rate_m_per_s;
        predicted_tool_life_hours = time_to_failure_s / 3600.0;
    } else {
        predicted_tool_life_hours = 500.0; // Default high value if no wear
    }
    
    // Clamp to reasonable values
    predicted_tool_life_hours = std::min(std::max(predicted_tool_life_hours, 0.01), 500.0);
}

void Simulation::write_output(const json& final_results) {
    json out;
    out["metadata"] = { 
        {"engine_version", "4.0-R&D-STRATEGY"}, 
        {"simulation_type", config_data.value("machining_type", "turning")},
        {"mesh_nodes", mesh.nodes.size()},
        {"mesh_triangles", mesh.triangle_indices.size() / 3}
    };
    
    out["tool_life_prediction"] = { 
        {"predicted_hours", predicted_tool_life_hours}, 
        {"wear_threshold_mm", wear_threshold_mm} 
    };
    
    out["final_metrics"] = final_results;
    out["time_series_data"] = time_series_output;
    
    if (!cfd_visualization_data.empty()) {
        out["visualization_data"] = cfd_visualization_data;
    }

    // Write final node states for visualization
    out["final_node_states"] = json::array();
    for (const auto& n : mesh.nodes) {
        out["final_node_states"].push_back({
            {"position", {n.position.x(), n.position.y(), n.position.z()}}, // Already in meters
            {"stress_MPa", n.stress}, 
            {"temperature_C", n.temperature},
            {"wear_m", n.accumulated_wear},
            {"status", (n.status == NodeStatus::OK ? "OK" : "FRACTURED")}
        });
    }
    
    out["mesh_connectivity"] = mesh.triangle_indices;
    
    std::string output_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::ofstream o(output_path);
    if (!o.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    o << std::setw(4) << out << std::endl;
}