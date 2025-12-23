#define _USE_MATH_DEFINES
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <algorithm>

// Strategy implementations
#include "IPhysicsStrategy.h"
#include "TurningStrategy.h"
#include "MillingStrategy.h"

// STL Reader
#include "stl_reader.h"

// OpenCASCADE Headers (CAD Support)
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>

// --- Constructor & Destructor ---
Simulation::Simulation() 
    : predicted_tool_life_hours(0.0)
    , wear_threshold_mm(0.3)
    , m_strategy(nullptr) 
{
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        
        load_config("input.json");
        
        // --- Initialize New Managers (Tier 1 & 3 Features) ---
        // These act as the "Brain" (Optimizer) and "Conscience" (Energy) of the engine
        m_optimizer = std::make_unique<OptimizationManager>(config_data);
        m_energy_monitor = std::make_unique<EnergyMonitor>(config_data);
        
        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] Simulation Init Failed: " << e.what() << "\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Simulation::~Simulation() {
    if (m_strategy) delete m_strategy;
}

// --- Main Simulation Loop ---
void Simulation::run() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v3.1 - High Performance     " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string sim_type = config_data.value("machining_type", "turning");
    
    // Select Strategy Factory
    if (sim_type == "turning") {
        m_strategy = new TurningStrategy(this);
    } else if (sim_type == "milling" || sim_type == "drilling") {
        m_strategy = new MillingStrategy(this);
    } else {
        throw std::runtime_error("Unknown machining_type: " + sim_type);
    }

    // Initialize Physics Strategy (This sets up SPH, FEA, CFD)
    m_strategy->initialize(mesh, cad_shape, config_data);

    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    double dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    int output_interval = config_data["simulation_parameters"].value("output_interval_steps", 10);
    
    std::cout << "Simulation starting (" << num_steps << " frames)..." << std::endl;
    
    for (int step = 1; step <= num_steps; ++step) {
        // Run Physics Step (Adaptive & FSI logic handles internally inside strategy)
        json step_metrics = m_strategy->run_time_step(dt, step);
        
        // Enrich metrics
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        
        // Store output based on interval to save memory
        if (step % output_interval == 0 || step == 1) {
            time_series_output.push_back(step_metrics);
            
            // Retrieve heavy particle data for viz only when needed
            // This prevents stalling the GPU every single frame
            json viz_frame = m_strategy->get_visualization_data();
            if (!viz_frame.empty()) {
                cfd_visualization_data.push_back(viz_frame[0]); 
            }
            
            write_progress(step, num_steps, step_metrics);
        }
    }
    
    std::cout << "\nSimulation finished. Finalizing results..." << std::endl;
    
    // Get Final Analysis (Distortion, Tool Life, etc.)
    json final_results = m_strategy->get_final_results();
    calculate_tool_life_prediction(final_results);
    
    // Save to Disk
    write_output(final_results);
    
    std::cout << "Done. Output saved." << std::endl;
}

// --- Helper Functions ---

void Simulation::write_progress(int step, int total_steps, const json& current_metrics) {
    double percent = (double)step / total_steps * 100.0;
    
    // Safe access to metrics with defaults
    double stress = current_metrics.value("max_stress_MPa", 0.0);
    double temp = current_metrics.value("max_temp_C", 0.0);
    double vel = current_metrics.value("max_velocity_m_s", 0.0);
    
    // Carriage return (\r) to update line in place
    std::cout << "  Frame " << std::setw(6) << step 
              << " | " << std::fixed << std::setprecision(1) << percent << "%"
              << " | S: " << std::scientific << std::setprecision(2) << stress << " MPa"
              << " | T: " << std::fixed << std::setprecision(1) << temp << " C"
              << " | V: " << std::fixed << std::setprecision(1) << vel << " m/s"
              << "   \r" << std::flush;
}

void Simulation::load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open config file: " + path);
    config_data = json::parse(f);
    std::cout << "Config loaded: " << path << std::endl;
}

std::string Simulation::get_file_ext(const std::string& s) {
    size_t i = s.rfind('.', s.length());
    return (i != std::string::npos) ? s.substr(i + 1) : "";
}

// --- Geometry Loading (Tool & Mesh) ---

void Simulation::load_geometry() {
    std::string path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::string ext = get_file_ext(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    std::cout << "Loading tool geometry: " << path << std::endl;
    
    if (ext == "stl") { 
        load_geometry_stl(path); 
        cad_shape = TopoDS_Shape(); // Empty shape for STL
    } else { 
        // STEP or IGES
        load_geometry_cad(path, ext.substr(0, 4)); 
        create_mesh_from_cad(cad_shape); 
    }
    
    if (mesh.nodes.empty()) {
        throw std::runtime_error("Geometry loaded but yielded 0 nodes. Check file units/scale.");
    }
    
    // Scale Check: Convert to meters if assumption is mm
    // (Standardizing on Meters for Physics)
    for (auto& n : mesh.nodes) { 
        n.position *= 0.001; 
        n.original_position *= 0.001; 
    }
    
    std::cout << "Geometry loaded: " << mesh.nodes.size() << " tool nodes." << std::endl;
}

void Simulation::load_geometry_stl(const std::string& path) {
    stl_reader::StlMesh<float, unsigned int> stl_mesh;
    try { 
        stl_mesh.read_file(path.c_str()); 
    } catch (const std::exception& e) { 
        throw std::runtime_error("Failed to read STL: " + std::string(e.what())); 
    }

    double ambient = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    // Convert STL to Engine Mesh Format
    for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
        const float* p = stl_mesh.vrt_coords(i);
        Node n; 
        n.position = Eigen::Vector3d(p[0], p[1], p[2]); 
        n.original_position = n.position;
        n.velocity = Eigen::Vector3d::Zero();
        n.acceleration = Eigen::Vector3d::Zero();
        n.force = Eigen::Vector3d::Zero(); 
        n.mass = 1e-9; // Will be recalculated by FEA
        n.temperature = ambient; 
        n.stress = 0; 
        n.status = (int)NodeStatus::OK;
        n.accumulated_wear = 0.0;
        n.is_active_contact = false;
        
        mesh.nodes.push_back(n);
    }
    
    for (size_t i = 0; i < stl_mesh.num_tris(); ++i) {
        const unsigned int* t = stl_mesh.tri_corner_inds(i);
        mesh.triangle_indices.push_back(t[0]); 
        mesh.triangle_indices.push_back(t[1]); 
        mesh.triangle_indices.push_back(t[2]);
    }
}

void Simulation::load_geometry_cad(const std::string& path, const std::string& format) {
    if (format.find("step") != std::string::npos || format.find("stp") != std::string::npos) { 
        STEPControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone) 
            throw std::runtime_error("Could not read STEP file: " + path);
        r.TransferRoots(); 
        cad_shape = r.OneShape(); 
    }
    else { 
        IGESControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone)
            throw std::runtime_error("Could not read IGES file (Did you mean STEP?): " + path);
        r.TransferRoots(); 
        cad_shape = r.OneShape(); 
    }
    
    if (cad_shape.IsNull()) throw std::runtime_error("CAD file loaded but shape is null.");
}

void Simulation::create_mesh_from_cad(const TopoDS_Shape& shape) {
    // Mesh the CAD shape using OpenCASCADE BRepMesh
    double deflection = config_data.value("mesh_deflection_mm", 0.05);
    BRepMesh_IncrementalMesh(shape, deflection);
    
    std::map<std::array<double,3>, int> map;
    double ambient = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopLoc_Location L;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), L);
        if (tri.IsNull()) continue;

        // Process Nodes
        for (int i = 1; i <= tri->NbNodes(); ++i) {
            gp_Pnt p = tri->Node(i).Transformed(L);
            std::array<double,3> c = {p.X(), p.Y(), p.Z()};
            
            // Deduplicate nodes
            if (map.find(c) == map.end()) {
                map[c] = mesh.nodes.size();
                Node n; 
                n.position = Eigen::Vector3d(c[0], c[1], c[2]); 
                n.original_position = n.position;
                n.velocity = Eigen::Vector3d::Zero();
                n.acceleration = Eigen::Vector3d::Zero();
                n.force = Eigen::Vector3d::Zero(); 
                n.mass = 1e-9;
                n.temperature = ambient; 
                n.stress = 0; 
                n.status = (int)NodeStatus::OK;
                n.accumulated_wear = 0.0;
                n.is_active_contact = false;
                
                mesh.nodes.push_back(n);
            }
        }
        
        // Process Triangles
        for (int i = 1; i <= tri->NbTriangles(); ++i) {
            int n1, n2, n3; 
            tri->Triangle(i).Get(n1, n2, n3);
            
            gp_Pnt p1 = tri->Node(n1).Transformed(L); 
            gp_Pnt p2 = tri->Node(n2).Transformed(L); 
            gp_Pnt p3 = tri->Node(n3).Transformed(L);
            
            mesh.triangle_indices.push_back(map[{p1.X(), p1.Y(), p1.Z()}]);
            mesh.triangle_indices.push_back(map[{p2.X(), p2.Y(), p2.Z()}]);
            mesh.triangle_indices.push_back(map[{p3.X(), p3.Y(), p3.Z()}]);
        }
    }
}

void Simulation::calculate_tool_life_prediction(const json& final_results) { 
    // Simplified placeholder logic - Real prediction happens inside OptimizationManager or specialized module
    // This function could analyze cumulative wear data from nodes
    predicted_tool_life_hours = 500.0; 
}

void Simulation::write_output(const json& final_results) {
    json out; 
    out["final_metrics"] = final_results; 
    
    // Add Visualization Data (Particles, Tool Position)
    out["visualization_data"] = cfd_visualization_data;
    
    // --- NEW: Add Sustainability Report (Tier 3 Feature) ---
    if (m_energy_monitor) {
        out["sustainability_report"] = m_energy_monitor->get_report();
    }
    
    std::string out_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::ofstream o(out_path);
    if (!o.is_open()) {
        std::cerr << "Failed to write output to: " << out_path << std::endl;
        return;
    }
    
    // Pretty print JSON
    o << std::setw(4) << out << std::endl;
}