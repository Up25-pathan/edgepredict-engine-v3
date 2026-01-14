#define _USE_MATH_DEFINES
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <map>
#include <algorithm>
#include <vector>

// Strategies
#include "IPhysicsStrategy.h"
#include "TurningStrategy.h"
#include "MillingStrategy.h"

// Helpers
#include "stl_reader.h"

// OpenCASCADE Headers
#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>

// --- Constructor ---
// FIX: Accepts config_path to allow running from command line (docker)
Simulation::Simulation(const std::string& config_path) 
    : predicted_tool_life_hours(0.0)
    , wear_threshold_mm(0.3)
    , m_strategy(nullptr) 
{
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        
        // Load Configuration
        load_config(config_path);
        
        // Initialize Managers (AI Optimization & Energy)
        m_optimizer = std::make_unique<OptimizationManager>(config_data);
        m_energy_monitor = std::make_unique<EnergyMonitor>(config_data);
        
        // Initialize G-Code Interpreter (if enabled)
        if (config_data.contains("gcode_file")) {
            std::string gpath = config_data["gcode_file"].get<std::string>();
            if (!gpath.empty()) {
                m_gcode_interpreter = std::make_unique<GCodeInterpreter>();
                m_gcode_interpreter->load_file(gpath);
            }
        }

        // Load 3D Assets
        load_geometry();

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] Simulation Init Failed: " << e.what() << "\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Simulation::~Simulation() {
    if (m_strategy) delete m_strategy;
}

// --- Main Loop ---
void Simulation::run() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v3.2 - G-Code Enabled       " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string sim_type = config_data.value("machining_type", "turning");
    
    // Strategy Factory
    if (sim_type == "turning") {
        m_strategy = new TurningStrategy(this);
    } else if (sim_type == "milling" || sim_type == "drilling") {
        m_strategy = new MillingStrategy(this);
    } else {
        throw std::runtime_error("Unknown machining_type: " + sim_type);
    }

    // Initialize Physics
    m_strategy->initialize(mesh, cad_shape, config_data);

    // Simulation Parameters
    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    double dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    int output_interval = config_data["simulation_parameters"].value("output_interval_steps", 10);
    
    std::cout << "Simulation starting (" << num_steps << " frames)..." << std::endl;
    
    for (int step = 1; step <= num_steps; ++step) {
        // --- NEW: G-Code / Machine State Update ---
        if (m_gcode_interpreter) {
            double time = step * dt;
            MachineState state = m_gcode_interpreter->get_state_at_time(time);
            
            // Update Strategy with new machine coordinates/speeds
            if(state.active) {
                m_strategy->update_machine_state(state.target_pos, state.feed_rate_mm_min, state.spindle_speed_rpm);
            }
        }

        // Run Physics Step
        json step_metrics = m_strategy->run_time_step(dt, step);
        
        // Record Metrics
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        
        // Output Handling
        if (step % output_interval == 0 || step == 1) {
            time_series_output.push_back(step_metrics);
            
            // Heavy data (Particles/CFD)
            json viz_frame = m_strategy->get_visualization_data();
            if (!viz_frame.empty()) {
                cfd_visualization_data.push_back(viz_frame[0]); 
            }
            
            write_progress(step, num_steps, step_metrics);
        }
    }
    
    std::cout << "\nSimulation finished. Finalizing results..." << std::endl;
    
    // Post-Process
    json final_results = m_strategy->get_final_results();
    calculate_tool_life_prediction(final_results);
    
    // Write Results
    write_output(final_results);
    
    std::cout << "Done. Output saved." << std::endl;
}

// --- Helpers ---

void Simulation::write_progress(int step, int total_steps, const json& current_metrics) {
    double percent = (double)step / total_steps * 100.0;
    
    double stress = current_metrics.value("max_stress_MPa", 0.0);
    double temp = current_metrics.value("max_temp_C", 0.0);
    
    std::cout << "  Frame " << std::setw(6) << step 
              << " | " << std::fixed << std::setprecision(1) << percent << "%"
              << " | S: " << std::scientific << std::setprecision(2) << stress << " MPa"
              << " | T: " << std::fixed << std::setprecision(1) << temp << " C"
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

// --- Geometry ---

void Simulation::load_geometry() {
    std::string path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::string ext = get_file_ext(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    std::cout << "Loading tool geometry: " << path << std::endl;
    
    // Load mesh data
    if (ext == "stl") { 
        load_geometry_stl(path); 
        cad_shape = TopoDS_Shape(); 
    } else { 
        load_geometry_cad(path, ext.substr(0, 4)); 
        create_mesh_from_cad(cad_shape); 
    }
    
    if (mesh.nodes.empty()) {
        throw std::runtime_error("Geometry loaded but yielded 0 nodes. Check file units/scale.");
    }
    
    // --- FIX: Unit Scaling (mm to meters is typical) ---
    double scale_factor = config_data.value("unit_scale_factor", 0.001); 
    if(scale_factor != 1.0) {
        std::cout << "Applying geometry scale factor: " << scale_factor << std::endl;
        for (auto& n : mesh.nodes) { 
            n.position *= scale_factor; 
            n.original_position *= scale_factor; 
        }
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
    
    for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
        const float* p = stl_mesh.vrt_coords(i);
        Node n; 
        n.position = Eigen::Vector3d(p[0], p[1], p[2]); 
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
    } else { 
        IGESControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone)
            throw std::runtime_error("Could not read IGES file: " + path);
        r.TransferRoots(); 
        cad_shape = r.OneShape(); 
    }
    
    if (cad_shape.IsNull()) throw std::runtime_error("CAD file loaded but shape is null.");
}

void Simulation::create_mesh_from_cad(const TopoDS_Shape& shape) {
    double deflection = config_data.value("mesh_deflection_mm", 0.05);
    BRepMesh_IncrementalMesh(shape, deflection);
    
    std::map<std::array<double,3>, int> map;
    double ambient = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopLoc_Location L;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), L);
        if (tri.IsNull()) continue;

        for (int i = 1; i <= tri->NbNodes(); ++i) {
            gp_Pnt p = tri->Node(i).Transformed(L);
            std::array<double,3> c = {p.X(), p.Y(), p.Z()};
            
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
    predicted_tool_life_hours = 500.0; 
}

void Simulation::write_output(const json& final_results) {
    json out; 
    out["final_metrics"] = final_results; 
    out["visualization_data"] = cfd_visualization_data;
    
    if (m_energy_monitor) {
        out["sustainability_report"] = m_energy_monitor->get_report();
    }
    
    std::string out_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::ofstream o(out_path);
    if (!o.is_open()) {
        std::cerr << "Failed to write output to: " << out_path << std::endl;
        return;
    }
    o << std::setw(4) << out << std::endl;
}