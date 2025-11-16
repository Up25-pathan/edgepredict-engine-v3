#define _USE_MATH_DEFINES
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <omp.h>
#include "stl_reader.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <map>
#include <random>

#include "physics_models.h" 
#include "navier_stokes.h"
#include "particle_system.h" 

FEASolver::FEASolver(const json& config, Simulation* sim) 
    : config_data(config), parent_sim(sim), total_fractured_nodes(0)
{
    dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    
    material_model = std::make_unique<JohnsonCookModel>(config_data);
    wear_model = std::make_unique<UsuiWearModel>(config_data);
    thermal_model = std::make_unique<ThermalModel>(config_data);
    failure_model = std::make_unique<FailureCriterion>(config_data);
    
    ambient_temperature = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    const auto& physics = config_data["physics_parameters"];
    strain_rate = physics["strain_rate"].get<double>();
    strain_increment = physics["strain_increment_per_step"].get<double>();
    sliding_velocity = physics["sliding_velocity_m_s"].get<double>();

    // --- REMOVED MANUAL CONTACT ZONE PARSING ---
    // We now use auto-detection, so we don't need to throw errors about contact_zone.
    
    cfd_enabled = false;
    if (config_data.contains("cfd_parameters")) {
        cfd_enabled = config_data["cfd_parameters"].value("enable_cfd", false);
        if (cfd_enabled) {
            chip_model = std::make_unique<ChipFormationModel>(config_data);
            fluid_solver = std::make_unique<NavierStokesSolver>(config_data); 
            fluid_solver->initialize_grid(20, 20, 20, 0.0005);
            
            double chip_velocity_calc = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
            double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
            Eigen::Vector3d chip_dir(cos(rake_angle), 0, sin(rake_angle));
            fluid_solver->set_inlet_velocity(chip_dir * chip_velocity_calc);
            
            particle_system = std::make_unique<ParticleSystem>(2000); 
        }
    }
}

FEASolver::~FEASolver() = default;

void FEASolver::solve(Mesh& mesh, int num_steps) {
    active_mesh = &mesh;

    // --- NEW: Run Auto-Detection ---
    auto_detect_cutting_edge(mesh); 

    std::cout << "  EDGEPREDICT ENGINE v3.5 - SMART TURNING PHYSICS" << std::endl;
    
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Node& node = mesh.nodes[i];
        node.temperature = ambient_temperature;
        node.plastic_strain = 1e-6; 
        node.strain = 1e-6;         
        node.stress = 0.0;
        node.accumulated_wear = 0.0;
        node.status = NodeStatus::OK;
    }

    for (int step = 1; step <= num_steps; ++step) {
        json step_metrics = solve_time_step(step);
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        parent_sim->time_series_output.push_back(step_metrics);
    }
}

json FEASolver::solve_time_step(int step_num) {
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        // --- NEW SMART LOGIC ---
        // Instead of checking a box coordinates, we check the flag set by auto_detect
        bool is_in_contact_zone = node.is_active_contact;
        
        if (node.status == NodeStatus::OK) {
            update_node_mechanics(node, dt, is_in_contact_zone);
            double chip_heat_flux = 0.0;
            if (cfd_enabled && fluid_solver && thermal_model && is_in_contact_zone) {
                double chip_temp = fluid_solver->get_temperature_at(node.position);
                chip_heat_flux = thermal_model->calculate_heat_transfer_from_chip(chip_temp, node.temperature, 1e-7);
            }
            update_node_thermal(node, dt, chip_heat_flux, is_in_contact_zone);
            if (is_in_contact_zone && check_node_failure(node)) {
                fractured_this_step++;
            }
        }
        if (node.temperature > max_temp) max_temp = node.temperature;
        if (node.stress > max_stress) max_stress = node.stress;
        total_wear += node.accumulated_wear;
    }
    total_fractured_nodes += fractured_this_step;
    
    json cfd_metrics;
    if (cfd_enabled && fluid_solver && particle_system && thermal_model && chip_model) { 
        fluid_solver->update_boundaries_and_sources(*active_mesh, ambient_temperature, strain_rate, *thermal_model);
        fluid_solver->update(dt);
        
        // Use actual material melting point
        double T_melt_limit = config_data["material_properties"]["melting_point_C"].get<double>();

        if (step_num % 2 == 0) {
             Eigen::Vector3d shear_zone(1e-4, 0.0, 1e-4); 
             double chip_vel = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
             double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
             Eigen::Vector3d emission_vel(cos(rake_angle - 0.2) * chip_vel, 0, sin(rake_angle - 0.2) * chip_vel);
             
             double avg_tool_temp = std::min(max_temp, T_melt_limit - 50.0);
             particle_system->emit_particles(shear_zone, emission_vel, avg_tool_temp, 50); 
        }
        particle_system->update(dt, *fluid_solver);

        double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
        auto rake_pressures = fluid_solver->get_rake_face_pressure(Eigen::Vector3d(0.2, 0, 0.8).normalized());
        double avg_rake_pressure = 0.0;
        double max_rake_pressure = 0.0;
        for (double p : rake_pressures) {
             avg_rake_pressure += p;
             if (p > max_rake_pressure) max_rake_pressure = p;
        }
        avg_rake_pressure /= (rake_pressures.empty() ? 1.0 : rake_pressures.size());
        
        // Analytical Pressure Fallback (Material Agnostic)
        if (max_rake_pressure < 1e-6) {
             double specific_energy = config_data["material_properties"].value("specific_cutting_energy_MPa", 3000.0) * 1e6;
             max_rake_pressure = specific_energy * (std::min(1.0, strain_rate / 10000.0));
        }

        double reported_chip_temp = std::min(particle_system->get_average_temperature(), T_melt_limit);

        cfd_metrics["chip_flow"] = {
            {"avg_chip_temperature_C", reported_chip_temp},
            {"chip_velocity_m_s", chip_model->calculate_chip_velocity(sliding_velocity, 2.5)},
            {"shear_angle_degrees", 45.0 + (rake_angle * 180.0/M_PI)/2.0}
        };
        cfd_metrics["rake_face"] = { {"avg_pressure_MPa", max_rake_pressure / 1e6} };

        if (step_num % 5 == 0) {
            json frame;
            frame["step"] = step_num;
            frame["particles"] = particle_system->export_particles();
            parent_sim->cfd_visualization_data.push_back(frame);
        }
    }

    if (step_num % 10 == 0 || step_num == 1) {
        std::cout << "  Step " << std::setw(4) << step_num 
                  << " │ T_max: " << std::fixed << std::setprecision(0) << std::setw(4) << max_temp << "°C"
                  << " │ Wear: " << std::scientific << std::setprecision(2) << total_wear << " m" << std::endl;
    }

    json metrics;
    metrics["max_temperature_C"] = max_temp;
    metrics["max_stress_MPa"] = max_stress;
    metrics["total_accumulated_wear_m"] = total_wear;
    metrics["cumulative_fractured_nodes"] = total_fractured_nodes;
    if (!cfd_metrics.empty()) metrics["cfd"] = cfd_metrics;
    return metrics;
}

void FEASolver::update_node_mechanics(Node& node, double dt, bool is_in_contact_zone) {
    if (!material_model || !wear_model || !failure_model) return;

    if (is_in_contact_zone) {
        node.plastic_strain += strain_increment;
        double effective_strain_rate = std::max(strain_increment / dt, strain_rate);

        node.stress = material_model->calculate_stress(node.temperature, node.plastic_strain, effective_strain_rate);
        
        double wear_rate = wear_model->calculate_wear_rate(node.temperature, node.stress, sliding_velocity);
        node.accumulated_wear += wear_rate * dt;
        
        double damage_inc = failure_model->calculate_damage_increment(node.stress, effective_strain_rate/strain_rate, node.accumulated_damage);
        
        node.accumulated_damage += damage_inc * 0.1;
        
        if (node.accumulated_damage >= 1.0) check_node_failure(node);
    } else {
        node.stress *= 0.95; 
    }
}

void FEASolver::update_node_thermal(Node& node, double dt, double chip_heat_flux, bool is_in_contact_zone) {
    if (!thermal_model) return;
    double heat_gen = is_in_contact_zone ? thermal_model->calculate_heat_generation(node.stress, strain_rate) : 0.0;
    double heat_loss = thermal_model->calculate_heat_dissipation(node.temperature, ambient_temperature, 1e-7);
    
    node.temperature += (heat_gen - heat_loss + (is_in_contact_zone ? chip_heat_flux : 0.0)) * dt;

    // Use actual material melting point from Config
    double T_melt = config_data["material_properties"]["melting_point_C"].get<double>();

    if (node.temperature > T_melt) node.temperature = T_melt;
    if (node.temperature < ambient_temperature) node.temperature = ambient_temperature;
}

bool FEASolver::check_node_failure(Node& node) {
    if (node.accumulated_damage >= 1.0) {
        node.status = NodeStatus::FRACTURED;
        node.stress = 0.0;
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════
// AUTO-DETECTION LOGIC (NEW)
// ═══════════════════════════════════════════════════════════

void FEASolver::auto_detect_cutting_edge(Mesh& mesh) {
    std::cout << "  [Auto-Detect] Scanning tool geometry for cutting tip..." << std::endl;

    if (mesh.nodes.empty()) return;

    // 1. Find the "Tip"
    size_t tip_index = 0;
    double max_score = -1e9;

    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const auto& pos = mesh.nodes[i].position;
        // Heuristic: For standard ISO turning (Z is up, X is radial), 
        // the tip is the most positive X and Z.
        double score = (pos.x() * 0.7) + (pos.z() * 0.3); 

        if (score > max_score) {
            max_score = score;
            tip_index = i;
        }
    }

    Eigen::Vector3d tip_position = mesh.nodes[tip_index].position;
    std::cout << "  [Auto-Detect] Tip found at: " 
              << tip_position.x() << ", " << tip_position.y() << ", " << tip_position.z() << std::endl;

    // 2. Activate nodes near the tip (2mm radius)
    double detection_radius = 0.002; 
    int active_count = 0;

    for (auto& node : mesh.nodes) {
        double dist = (node.position - tip_position).norm();
        
        if (dist <= detection_radius) {
            node.is_active_contact = true;
            active_count++;
        } else {
            node.is_active_contact = false;
        }
    }

    std::cout << "  [Auto-Detect] Activated " << active_count << " nodes for simulation." << std::endl;
}

// ═══════════════════════════════════════════════════════════
// SIMULATION CLASS
// ═══════════════════════════════════════════════════════════
Simulation::Simulation() : predicted_tool_life_hours(0.0), wear_threshold_mm(0.3) {
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        load_config("input.json");
        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::run() {
    FEASolver solver(config_data, this);
    solver.solve(mesh, config_data["simulation_parameters"]["num_steps"].get<int>());
    calculate_tool_life_prediction();
    write_output();
}

void Simulation::load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
    config_data = json::parse(f);
}

std::string get_file_ext(const std::string& s) {
    size_t i = s.rfind('.', s.length());
    return (i != std::string::npos) ? s.substr(i + 1) : "";
}

void Simulation::load_geometry() {
    std::string path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::string ext = get_file_ext(path);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "stl") load_geometry_stl(path);
    else if (ext == "step" || ext == "stp" || ext == "iges" || ext == "igs") load_geometry_cad(path, ext.substr(0,4));
    else throw std::runtime_error("Unknown geometry format: " + ext);

    for (auto& n : mesh.nodes) n.position *= 0.001; // Scale to meters
}

void Simulation::load_geometry_stl(const std::string& path) {
    stl_reader::StlMesh<float, unsigned int> stl_mesh(path);
    for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
        const float* p = stl_mesh.vrt_coords(i);
        mesh.nodes.push_back({Eigen::Vector3d(p[0], p[1], p[2]), config_data["physics_parameters"]["ambient_temperature_C"].get<double>(), 0, 0, 0, 0, 0, NodeStatus::OK});
    }
}

void Simulation::load_geometry_cad(const std::string& path, const std::string& format) {
    TopoDS_Shape shape;
    
    // --- CRITICAL FIX: Check for "stp" as well as "step" ---
    if (format.find("step") != std::string::npos || format.find("stp") != std::string::npos) {
        STEPControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone) throw std::runtime_error("Bad STEP file");
        r.TransferRoots(); shape = r.OneShape();
    } else {
        // Otherwise assume IGES
        IGESControl_Reader r; 
        if (r.ReadFile(path.c_str()) != IFSelect_RetDone) throw std::runtime_error("Bad IGES file");
        r.TransferRoots(); shape = r.OneShape();
    }
    // -------------------------------------------------------

    BRepMesh_IncrementalMesh(shape, 0.05);
    std::map<std::array<double,3>, int> nmap;
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopLoc_Location L;
        auto tri = BRep_Tool::Triangulation(TopoDS::Face(ex.Current()), L);
        if (tri.IsNull()) continue;
        for (int i=1; i<=tri->NbNodes(); ++i) {
            gp_Pnt p = tri->Node(i).Transformed(L);
            std::array<double,3> c = {p.X(), p.Y(), p.Z()};
            if (nmap.find(c) == nmap.end()) {
                mesh.nodes.push_back({Eigen::Vector3d(c[0],c[1],c[2]), config_data["physics_parameters"]["ambient_temperature_C"].get<double>(), 0,0,0,0,0,NodeStatus::OK});
                nmap[c] = mesh.nodes.size()-1;
            }
        }
        for (int i=1; i<=tri->NbTriangles(); ++i) {
            int n1,n2,n3; tri->Triangle(i).Get(n1,n2,n3);
            mesh.triangle_indices.push_back(nmap[{tri->Node(n1).Transformed(L).X(), tri->Node(n1).Transformed(L).Y(), tri->Node(n1).Transformed(L).Z()}]);
            mesh.triangle_indices.push_back(nmap[{tri->Node(n2).Transformed(L).X(), tri->Node(n2).Transformed(L).Y(), tri->Node(n2).Transformed(L).Z()}]);
            mesh.triangle_indices.push_back(nmap[{tri->Node(n3).Transformed(L).X(), tri->Node(n3).Transformed(L).Y(), tri->Node(n3).Transformed(L).Z()}]);
        }
    }
}

void Simulation::calculate_tool_life_prediction() {
    double wear = time_series_output.back()["total_accumulated_wear_m"].get<double>();
    double time = time_series_output.back()["time_s"].get<double>();
    predicted_tool_life_hours = (wear > 1e-12) ? (0.0003 / (wear/time)) / 3600.0 : 100.0;
    predicted_tool_life_hours = std::min(std::max(predicted_tool_life_hours, 0.1), 500.0);
}

void Simulation::write_output() {
    json out;
    out["metadata"] = { {"engine_version", "3.5-SMART-TURNING"}, {"nodes", mesh.nodes.size()} };
    out["tool_life_prediction"] = { {"predicted_hours", predicted_tool_life_hours}, {"wear_threshold_mm", 0.3} };
    out["time_series_data"] = time_series_output;
    if (!cfd_visualization_data.empty()) out["cfd_particle_animation"] = cfd_visualization_data;
    out["final_node_states"] = json::array();
    for (const auto& n : mesh.nodes) {
        out["final_node_states"].push_back({
            {"position", {n.position.x(), n.position.y(), n.position.z()}},
            {"stress_MPa", n.stress}, {"temperature_C", n.temperature}
        });
    }
    out["mesh_connectivity"] = mesh.triangle_indices;
    std::ofstream(config_data["file_paths"]["output_results"].get<std::string>()) << out;
}
