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

#include "IPhysicsStrategy.h"
#include "TurningStrategy.h"
#include "MillingStrategy.h"
#include "stl_reader.h"

#include <opencascade/STEPControl_Reader.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/TopoDS_Shape.hxx>
#include <opencascade/TopExp_Explorer.hxx>
#include <opencascade/BRep_Tool.hxx>
#include <opencascade/BRepMesh_IncrementalMesh.hxx>
#include <opencascade/Poly_Triangulation.hxx>

Simulation::Simulation() : predicted_tool_life_hours(0.0), wear_threshold_mm(0.3), m_strategy(nullptr) {
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        load_config("input.json");
        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << "\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Simulation::~Simulation() {
    if (m_strategy) delete m_strategy;
}

void Simulation::run() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v14.0 - FIXED LOADER        " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string sim_type = config_data.value("machining_type", "turning");
    
    if (sim_type == "turning") m_strategy = new TurningStrategy(this);
    else if (sim_type == "milling" || sim_type == "drilling") m_strategy = new MillingStrategy(this);
    else throw std::runtime_error("Unknown machining_type: " + sim_type);

    m_strategy->initialize(mesh, cad_shape, config_data);

    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    double dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    
    std::cout << "Simulation starting (" << num_steps << " frames)..." << std::endl;
    
    for (int step = 1; step <= num_steps; ++step) {
        json step_metrics = m_strategy->run_time_step(dt, step);
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        time_series_output.push_back(step_metrics);
        
        if (step % 10 == 0 || step == 1) write_progress(step, num_steps, step_metrics);
    }
    
    std::cout << "\nSimulation finished. Finalizing results..." << std::endl;
    
    json final_results = m_strategy->get_final_results();
    cfd_visualization_data = m_strategy->get_visualization_data();
    calculate_tool_life_prediction(final_results);
    write_output(final_results);
    
    std::cout << "Done. Output saved." << std::endl;
}

void Simulation::write_progress(int step, int total_steps, const json& current_metrics) {
    double percent = (double)step / total_steps * 100.0;
    std::cout << "  Frame " << std::setw(5) << step 
              << " | " << std::fixed << std::setprecision(1) << percent << "%"
              << " | S: " << std::scientific << std::setprecision(2) << current_metrics.value("max_stress_MPa", 0.0) << " MPa"
              << " | T: " << std::fixed << std::setprecision(1) << current_metrics.value("max_temp_C", 0.0) << " C"
              << " | V: " << std::fixed << std::setprecision(1) << current_metrics.value("max_velocity_m_s", 0.0) << " m/s"
              << "   \r" << std::flush;
}

void Simulation::load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open config file: " + path);
    config_data = json::parse(f);
    std::cout << "Config loaded." << std::endl;
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
        cad_shape = TopoDS_Shape(); 
    } else { 
        load_geometry_cad(path, ext.substr(0, 4)); 
        create_mesh_from_cad(cad_shape); 
    }
    
    if (mesh.nodes.empty()) {
        throw std::runtime_error("Geometry loaded but yielded 0 nodes. Check file units/scale.");
    }

    // Convert mm to meters for physics engine
    for (auto& n : mesh.nodes) { n.position *= 0.001; n.original_position *= 0.001; }
    std::cout << "Geometry loaded: " << mesh.nodes.size() << " nodes." << std::endl;
}

void Simulation::load_geometry_stl(const std::string& path) {
    stl_reader::StlMesh<float, unsigned int> stl_mesh;
    try { stl_mesh.read_file(path.c_str()); } 
    catch (const std::exception& e) { throw std::runtime_error("Failed to read STL: " + std::string(e.what())); }
    double ambient = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
        const float* p = stl_mesh.vrt_coords(i);
        Node n; n.position = Eigen::Vector3d(p[0], p[1], p[2]); n.original_position = n.position;
        n.velocity = n.acceleration = n.force = Eigen::Vector3d::Zero(); n.mass = 1e-9;
        n.temperature = ambient; n.stress=0; n.status=NodeStatus::OK;
        mesh.nodes.push_back(n);
    }
    for (size_t i = 0; i < stl_mesh.num_tris(); ++i) {
        const unsigned int* t = stl_mesh.tri_corner_inds(i);
        mesh.triangle_indices.push_back(t[0]); mesh.triangle_indices.push_back(t[1]); mesh.triangle_indices.push_back(t[2]);
    }
}

void Simulation::load_geometry_cad(const std::string& path, const std::string& format) {
    // FIX: Added check for "stp" (short format)
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

    if (cad_shape.IsNull()) {
        throw std::runtime_error("CAD file loaded but shape is null.");
    }
}

void Simulation::create_mesh_from_cad(const TopoDS_Shape& shape) {
    BRepMesh_IncrementalMesh(shape, config_data.value("mesh_deflection_mm", 0.05));
    std::map<std::array<double,3>, int> map;
    double ambient = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopLoc_Location L;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), L);
        if (tri.IsNull()) continue;
        const TColgp_Array1OfPnt& nodes = tri->Nodes();
        for (int i = 1; i <= tri->NbNodes(); ++i) {
            gp_Pnt p = nodes(i).Transformed(L);
            std::array<double,3> c = {p.X(), p.Y(), p.Z()};
            if (map.find(c) == map.end()) {
                map[c] = mesh.nodes.size();
                Node n; n.position = Eigen::Vector3d(c[0], c[1], c[2]); n.original_position = n.position;
                n.velocity=n.acceleration=n.force=Eigen::Vector3d::Zero(); n.mass=1e-9;
                n.temperature=ambient; n.stress=0; n.status=NodeStatus::OK;
                mesh.nodes.push_back(n);
            }
        }
        for (int i = 1; i <= tri->NbTriangles(); ++i) {
            int n1, n2, n3; tri->Triangle(i).Get(n1, n2, n3);
            gp_Pnt p1 = nodes(n1).Transformed(L); gp_Pnt p2 = nodes(n2).Transformed(L); gp_Pnt p3 = nodes(n3).Transformed(L);
            mesh.triangle_indices.push_back(map[{p1.X(), p1.Y(), p1.Z()}]);
            mesh.triangle_indices.push_back(map[{p2.X(), p2.Y(), p2.Z()}]);
            mesh.triangle_indices.push_back(map[{p3.X(), p3.Y(), p3.Z()}]);
        }
    }
}

void Simulation::calculate_tool_life_prediction(const json& final_results) { predicted_tool_life_hours = 500.0; }
void Simulation::write_output(const json& final_results) {
    json out; out["final_metrics"] = final_results; out["visualization_data"] = cfd_visualization_data;
    std::ofstream o(config_data["file_paths"]["output_results"].get<std::string>());
    o << std::setw(4) << out << std::endl;
}