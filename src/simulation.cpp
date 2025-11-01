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
#include <string> // For string operations
#include <map>    // For node de-duplication
#include <random> // For deterministic PRNG

// Include all physics modules
#include "physics_models.h" 
#include "navier_stokes.h"
#include "particle_system.h" 

// ═══════════════════════════════════════════════════════════
// FEA SOLVER IMPLEMENTATION (Unchanged)
// ═══════════════════════════════════════════════════════════

FEASolver::FEASolver(const json& config, Simulation* sim) 
    : config_data(config), parent_sim(sim), total_fractured_nodes(0)
{
    dt = config_data["simulation_parameters"]["time_step_duration_s"].get<double>();
    
    // Initialize physics models
    material_model = std::make_unique<JohnsonCookModel>(config_data);
    wear_model = std::make_unique<UsuiWearModel>(config_data);
    thermal_model = std::make_unique<ThermalModel>(config_data);
    failure_model = std::make_unique<FailureCriterion>(config_data);
    
    // Get initial temperature
    ambient_temperature = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    // Get simulation parameters
    const auto& physics = config_data["physics_parameters"];
    strain_rate = physics["strain_rate"].get<double>();
    strain_increment = physics["strain_increment_per_step"].get<double>();
    sliding_velocity = physics["sliding_velocity_m_s"].get<double>();

    // Read Contact Zone
    if (!config_data["physics_parameters"].contains("contact_zone")) {
        throw std::runtime_error("Configuration is missing 'contact_zone' definition in 'physics_parameters'.");
    }
    const auto& zone = config_data["physics_parameters"]["contact_zone"];
    contact_zone_min = {zone["x"][0].get<double>(), zone["y"][0].get<double>(), zone["z"][0].get<double>()};
    contact_zone_max = {zone["x"][1].get<double>(), zone["y"][1].get<double>(), zone["z"][1].get<double>()};
    std::cout << "  ✓ Contact zone loaded." << std::endl;
    
    // Initialize CFD if enabled
    cfd_enabled = false;
    if (config_data.contains("cfd_parameters")) {
        cfd_enabled = config_data["cfd_parameters"].value("enable_cfd", false); // Robust read
        
        if (cfd_enabled) {
            std::cout << "  ✓ CFD Analysis ENABLED" << std::endl;
            
            chip_model = std::make_unique<ChipFormationModel>(config_data);
            fluid_solver = std::make_unique<NavierStokesSolver>(config_data); 
            fluid_solver->initialize_grid(20, 20, 20, 0.0005);
            
            double chip_velocity_calc = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
            double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
            Eigen::Vector3d chip_dir(cos(rake_angle), 0, sin(rake_angle));
            fluid_solver->set_inlet_velocity(chip_dir * chip_velocity_calc);
            
            particle_system = std::make_unique<ParticleSystem>(2000); 
            
            std::cout << "  ✓ Navier-Stokes solver initialized (20³ grid)" << std::endl;
            std::cout << "  ✓ Particle system ready (max 2000 particles)" << std::endl;
        }
    }
}

FEASolver::~FEASolver() = default;

void FEASolver::solve(Mesh& mesh, int num_steps) {
    active_mesh = &mesh;
    std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  EDGEPREDICT ENGINE v3.0 - PRODUCTION SIMULATION" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Total nodes: " << mesh.nodes.size() << std::endl;
    std::cout << "  Time steps: " << num_steps << std::endl;
    std::cout << "  Time step size: " << dt << " s" << std::endl;
    
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Node& node = mesh.nodes[i];
        node.temperature = ambient_temperature;
        node.plastic_strain = 1e-6; 
        node.strain = 1e-6;         
        if(material_model) {
            node.stress = material_model->calculate_stress(ambient_temperature, node.plastic_strain, strain_rate);
        } else {
             node.stress = 0.0;
        }
        node.accumulated_wear = 0.0;
        node.status = NodeStatus::OK;
    }
    
    int thread_count = 0;
    #pragma omp parallel
    {
        #pragma omp single
        thread_count = omp_get_num_threads();
    }
    std::cout << "  OpenMP threads: " << thread_count << std::endl;
    std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;

    for (int step = 1; step <= num_steps; ++step) {
        json step_metrics = solve_time_step(step);
        step_metrics["step"] = step;
        step_metrics["time_s"] = step * dt;
        parent_sim->time_series_output.push_back(step_metrics);
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  SIMULATION COMPLETE" << std::endl;
    std::cout << "  Total fractured nodes: " << total_fractured_nodes << std::endl;
    std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;
}

json FEASolver::solve_time_step(int step_num) {
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        const auto& pos = node.position;
        bool is_in_contact_zone = (pos.x() >= contact_zone_min.x() && pos.x() <= contact_zone_max.x() &&
                                   pos.y() >= contact_zone_min.y() && pos.y() <= contact_zone_max.y() &&
                                   pos.z() >= contact_zone_min.z() && pos.z() <= contact_zone_max.z());
        
        if (node.status == NodeStatus::OK) {
            update_node_mechanics(node, dt, is_in_contact_zone);

            double chip_heat_flux = 0.0;
            if (cfd_enabled && fluid_solver && thermal_model && is_in_contact_zone) {
                double chip_temp = fluid_solver->get_temperature_at(node.position);
                chip_heat_flux = thermal_model->calculate_heat_transfer_from_chip(
                    chip_temp, node.temperature, 1e-7);
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

        if (step_num % 2 == 0) {
             Eigen::Vector3d shear_zone(0.0, 0.0, 0.0);
             double chip_ratio = config_data["cfd_parameters"]["chip_thickness_ratio"].get<double>();
             double chip_vel = chip_model->calculate_chip_velocity(sliding_velocity, chip_ratio);
             double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
             Eigen::Vector3d emission_vel(cos(rake_angle) * chip_vel, 0, sin(rake_angle) * chip_vel);
             double avg_tool_temp = 0.0;
             if (!active_mesh->nodes.empty()) {
                for(const auto& node : active_mesh->nodes) { avg_tool_temp += node.temperature; }
                avg_tool_temp /= active_mesh->nodes.size();
             } else {
                 avg_tool_temp = ambient_temperature;
             }
             particle_system->emit_particles(shear_zone, emission_vel, avg_tool_temp, 15);
        }

        particle_system->update(dt, *fluid_solver);

        double friction_angle = atan(config_data["cfd_parameters"]["friction_coefficient"].get<double>());
        double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
        double shear_angle = chip_model->calculate_shear_angle(rake_angle, friction_angle);
        double chip_velocity_calc = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
        auto rake_pressures = fluid_solver->get_rake_face_pressure(Eigen::Vector3d(0,0,1));
        double avg_rake_pressure = 0.0;
        double max_rake_pressure = 0.0;
        for (double p : rake_pressures) {
            avg_rake_pressure += p;
            if (p > max_rake_pressure) max_rake_pressure = p;
        }
        avg_rake_pressure /= (rake_pressures.empty() ? 1.0 : rake_pressures.size());
        
        cfd_metrics["chip_flow"] = {
            {"active_particles", particle_system->get_active_count()},
            {"avg_chip_temperature_C", particle_system->get_average_temperature()},
            {"max_chip_pressure_MPa", particle_system->get_max_pressure()},
            {"chip_velocity_m_s", chip_velocity_calc},
            {"shear_angle_degrees", shear_angle * 180.0 / M_PI}
        };
        
        cfd_metrics["rake_face"] = {
            {"avg_pressure_MPa", avg_rake_pressure / 1e6},
            {"max_pressure_MPa", max_rake_pressure / 1e6},
            {"contact_length_mm", chip_model->calculate_contact_length(0.0001, rake_angle) * 1000.0}
        };

        if (step_num % 5 == 0) {
            json frame;
            frame["step"] = step_num;
            frame["time_s"] = step_num * dt;
            frame["particles"] = particle_system->export_particles();
            if(parent_sim) {
                parent_sim->cfd_visualization_data.push_back(frame);
            }
        }
    }

    double max_contact_stress = 0.0;
    for (const auto& node : active_mesh->nodes) {
        const auto& pos = node.position;
        bool is_in_zone = (pos.x() >= contact_zone_min.x() && pos.x() <= contact_zone_max.x() &&
                           pos.y() >= contact_zone_min.y() && pos.y() <= contact_zone_max.y() &&
                           pos.z() >= contact_zone_min.z() && pos.z() <= contact_zone_max.z());
        if (is_in_zone && node.stress > max_contact_stress) max_contact_stress = node.stress;
    }
    
     std::cout << "  Step " << std::setw(4) << step_num << "/" << config_data["simulation_parameters"]["num_steps"].get<int>()
               << " │ T_max: " << std::fixed << std::setprecision(1) << std::setw(6) << max_temp << "°C"
               << " │ σ_contact: " << std::fixed << std::setprecision(1) << std::setw(6) << max_contact_stress << " MPa"
               << " │ Wear: " << std::scientific << std::setprecision(2) << total_wear << " m"
               << " │ Failed: " << std::setw(3) << fractured_this_step;
    
    if (cfd_enabled && cfd_metrics.contains("chip_flow") && cfd_metrics["chip_flow"].contains("active_particles")) {
        std::cout << " │ Particles: " << std::setw(4) << cfd_metrics["chip_flow"]["active_particles"];
    }
    std::cout << std::endl;


    json metrics;
    metrics["max_temperature_C"] = max_temp;
    metrics["max_stress_MPa"] = max_stress;
    metrics["total_accumulated_wear_m"] = total_wear;
    metrics["fractured_nodes_count"] = fractured_this_step;
    metrics["cumulative_fractured_nodes"] = total_fractured_nodes;
    
    if (!cfd_metrics.empty()) {
        metrics["cfd"] = cfd_metrics;
    }
    
    return metrics;
}

void FEASolver::update_node_mechanics(Node& node, double dt, bool is_in_contact_zone) {
    if (!material_model || !wear_model || !failure_model) {
        return; 
    }

    if (is_in_contact_zone) {
        double actual_strain_rate = strain_increment / dt;
        node.plastic_strain += strain_increment;
        double effective_strain_rate = std::max(actual_strain_rate, strain_rate);

        node.stress = material_model->calculate_stress(
            node.temperature, 
            node.plastic_strain,
            effective_strain_rate
        );
        
        double wear_rate = wear_model->calculate_wear_rate(
            node.temperature,
            node.stress,
            sliding_velocity
        );
        node.accumulated_wear += wear_rate * dt;
        
        double stress_factor = std::min(1.0, node.stress / 2000.0); 
        double temp_factor = std::min(1.0, node.temperature / 1200.0);
        double strain_rate_factor = std::min(1.0, effective_strain_rate / (100.0 * strain_rate));
        
        double damage_increment = failure_model->calculate_damage_increment(
            node.stress,
            effective_strain_rate / strain_rate,
            node.accumulated_damage
        ) * (0.7 + 0.3 * (stress_factor + temp_factor + strain_rate_factor) / 3.0);
        
        double thermal_sensitivity = std::min(1.0, std::max(0.0, (node.temperature - 200.0) / 800.0));
        double accumulation_chance = 0.20 + 0.30 * thermal_sensitivity; // CALIBRATED: Increased from 0.05+0.15 for faster damage
        
        if ((static_cast<double>(rand()) / RAND_MAX) < accumulation_chance) {
            node.accumulated_damage += damage_increment;
        }
        
        double base_threshold = 0.95 + 0.1 * (static_cast<double>(rand()) / RAND_MAX);
        double temp_adjusted_threshold = base_threshold * (1.0 + 0.2 * (1.0 - thermal_sensitivity));
        
        if (node.accumulated_damage >= temp_adjusted_threshold) {
            check_node_failure(node);
        }
    } else {
        node.stress = material_model->calculate_stress(
            node.temperature, 
            node.plastic_strain,
            1e-3 // Near-zero strain rate
        );
        node.accumulated_wear += 0.0;
        node.accumulated_damage += 0.0;
    }
}

void FEASolver::update_node_thermal(Node& node, double dt, double chip_heat_flux, bool is_in_contact_zone) {
    if (!thermal_model) return; 

    double heat_gen = 0.0;
    double chip_heat = 0.0;

    if (is_in_contact_zone) {
        heat_gen = thermal_model->calculate_heat_generation(node.stress, strain_rate);
        chip_heat = chip_heat_flux;
    }
    
    double heat_loss_rate = thermal_model->calculate_heat_dissipation(node.temperature, 
        ambient_temperature, 
        1e-7);
    
    double dT = (heat_gen - heat_loss_rate + chip_heat) * dt;
    node.temperature += dT;
    
    double T_melt = config_data["material_properties"]["melting_point_C"].get<double>();
    if (node.temperature > T_melt) {
        node.temperature = T_melt;
    }
    
    if (node.temperature < ambient_temperature) {
        node.temperature = ambient_temperature;
    }
}

bool FEASolver::check_node_failure(Node& node) {
    if (!failure_model) return false; 

    bool failed = failure_model->check_failure(node.stress, node.temperature, node.accumulated_damage);
    
    if (failed) {
        node.status = NodeStatus::FRACTURED;
        node.stress = 0.0;
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════
// SIMULATION CLASS IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

Simulation::Simulation() : predicted_tool_life_hours(0.0), wear_threshold_mm(0.3) {
    try {
        time_series_output = json::array();
        cfd_visualization_data = json::array();
        load_config("input.json");

        load_geometry();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR during initialization: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::run() {
    std::cout << "\n" << std::endl;
    std::cout << "███████╗██████╗  ██████╗ ███████╗██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗" << std::endl;
    std::cout << "██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝" << std::endl;
    std::cout << "█████╗  ██║  ██║██║  ███╗█████╗  ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   " << std::endl;
    std::cout << "██╔══╝  ██║  ██║██║   ██║██╔══╝  ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   " << std::endl;
    std::cout << "███████╗██████╔╝╚██████╔╝███████╗██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   " << std::endl;
    std::cout << "╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   " << std::endl;
    std::cout << "                         SIMULATION ENGINE v3.0 (PRODUCTION)                           \n" << std::endl;
    
    int num_steps = config_data["simulation_parameters"]["num_steps"].get<int>();
    
    FEASolver solver(config_data, this);
    solver.solve(mesh, num_steps);
    
    calculate_tool_life_prediction();
    write_output();
    
    std::cout << "\n✓ Engine finished successfully!" << std::endl;
}

void Simulation::load_config(const std::string& path) {
    std::cout << "  Loading configuration: " << path << std::endl;
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open input file: " + path);
    }
    
    try {
        config_data = json::parse(f);
        std::cout << "  ✓ Configuration loaded successfully" << std::endl;
    } catch (json::parse_error& e) {
        throw std::runtime_error("Failed to parse input.json: " + std::string(e.what()));
    }
}

// --- NEW: File extension helper ---
std::string get_file_extension(const std::string& filename) {
    size_t dot_pos = filename.rfind('.');
    if (dot_pos == std::string::npos) {
        return "";
    }
    std::string ext = filename.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return ext;
}

// --- UPDATED: Geometry loading dispatcher ---
void Simulation::load_geometry() {
    std::string geometry_path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::string ext = get_file_extension(geometry_path);

    // Determine geometry units (default mm for CAD/STL)
    std::string units = "mm";
    try {
        if (config_data.contains("file_paths") && config_data["file_paths"].contains("geometry_units")) {
            units = config_data["file_paths"]["geometry_units"].get<std::string>();
        }
    } catch (...) { /* keep default */ }
    std::transform(units.begin(), units.end(), units.begin(), [](unsigned char c){ return std::tolower(c); });
    double scale_to_m = (units == "mm") ? 0.001 : 1.0; // convert to meters

    std::cout << "  Loading tool geometry: " << geometry_path << " (Format: " << ext << ", Units: " << units << ")" << std::endl;

    mesh.nodes.clear();
    if (ext == "stl") {
        load_geometry_stl(geometry_path);
    } else if (ext == "step" || ext == "stp") {
        load_geometry_cad(geometry_path, "step");
    } else if (ext == "iges" || ext == "igs") {
        load_geometry_cad(geometry_path, "iges");
    } else {
        throw std::runtime_error("Unsupported geometry file format: " + ext);
    }

    // Scale positions to meters
    if (std::abs(scale_to_m - 1.0) > 1e-12) {
        for (auto& node : mesh.nodes) {
            node.position *= scale_to_m;
        }
    }

    // Compute bounding box
    if (mesh.nodes.empty()) {
        throw std::runtime_error("No nodes in mesh after loading geometry.");
    }
    Eigen::Vector3d bb_min = mesh.nodes.front().position;
    Eigen::Vector3d bb_max = mesh.nodes.front().position;
    for (const auto& n : mesh.nodes) {
        bb_min = bb_min.cwiseMin(n.position);
        bb_max = bb_max.cwiseMax(n.position);
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Tool bbox (m): min=(" << bb_min.x() << ", " << bb_min.y() << ", " << bb_min.z() << ")"
              << " max=(" << bb_max.x() << ", " << bb_max.y() << ", " << bb_max.z() << ")" << std::endl;

    // Optionally auto-derive contact zone from bbox with 1mm pad if enabled
    bool auto_contact = false;
    try {
        if (config_data.contains("physics_parameters") && config_data["physics_parameters"].contains("auto_contact_zone")) {
            auto_contact = config_data["physics_parameters"]["auto_contact_zone"].get<bool>();
        }
    } catch (...) { /* keep default false */ }

    if (auto_contact) {
        auto pad = 0.001; // 1 mm pad
        json cz;
        cz["x"] = {bb_min.x() - pad, bb_max.x() + pad};
        cz["y"] = {bb_min.y() - pad, bb_max.y() + pad};
        cz["z"] = {bb_min.z() - pad, bb_max.z() + pad};
        config_data["physics_parameters"]["contact_zone"] = cz;
        std::cout << "  ✓ auto_contact_zone applied from bbox with 1mm padding" << std::endl;
    }

    // Log contact zone and compute coverage
    if (!config_data["physics_parameters"].contains("contact_zone")) {
        throw std::runtime_error("Configuration is missing 'contact_zone' after geometry load.");
    }
    const auto& zone = config_data["physics_parameters"]["contact_zone"];
    Eigen::Vector2d zx(zone["x"][0].get<double>(), zone["x"][1].get<double>());
    Eigen::Vector2d zy(zone["y"][0].get<double>(), zone["y"][1].get<double>());
    Eigen::Vector2d zz(zone["z"][0].get<double>(), zone["z"][1].get<double>());
    std::cout << "  Contact zone (m): x=[" << zx.x() << ", " << zx.y() << "]"
              << " y=[" << zy.x() << ", " << zy.y() << "]"
              << " z=[" << zz.x() << ", " << zz.y() << "]" << std::endl;

    size_t in_contact = 0;
    for (const auto& n : mesh.nodes) {
        const auto& p = n.position;
        if (p.x() >= zx.x() && p.x() <= zx.y() &&
            p.y() >= zy.x() && p.y() <= zy.y() &&
            p.z() >= zz.x() && p.z() <= zz.y()) {
            ++in_contact;
        }
    }
    double coverage = 100.0 * (mesh.nodes.empty() ? 0.0 : static_cast<double>(in_contact) / static_cast<double>(mesh.nodes.size()));
    std::cout << std::setprecision(2) << "  Contact coverage: " << coverage << "% (" << in_contact << "/" << mesh.nodes.size() << ")" << std::endl;
    if (coverage < 1.0) {
        std::cout << "  WARNING: Contact zone covers <1% of nodes. Check geometry_units and contact_zone bounds." << std::endl;
    }

    std::cout << std::setprecision(6);
    std::cout << "  ✓ Loaded " << mesh.nodes.size() << " nodes from " << geometry_path << "\n" << std::endl;
}

// --- NEW: Original STL loading logic, now in its own function ---
void Simulation::load_geometry_stl(const std::string& path) {
    try {
        stl_reader::StlMesh<float, unsigned int> stl_mesh(path);
        mesh.nodes.reserve(stl_mesh.num_vrts());
        
        double initial_ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
        
        for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
            const float* pos = stl_mesh.vrt_coords(i);
            Node node;
            node.position = Eigen::Vector3d(pos[0], pos[1], pos[2]);
            node.temperature = initial_ambient_temp;
            node.plastic_strain = 0.0;
            node.strain = 0.0;
            node.stress = 0.0;
            node.accumulated_wear = 0.0;
            node.accumulated_damage = 0.0;
            node.status = NodeStatus::OK;

            mesh.nodes.push_back(node);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read STL file '" + path + "': " + e.what());
    }
}

// --- NEW: OpenCASCADE CAD loading logic ---
void Simulation::load_geometry_cad(const std::string& path, const std::string& format) {
    
    // --- 1. Read the CAD file ---
    TopoDS_Shape shape;
    if (format == "step") {
        STEPControl_Reader reader;
        IFSelect_ReturnStatus stat = reader.ReadFile(path.c_str());
        if (stat != IFSelect_RetDone) {
            throw std::runtime_error("STEP file could not be read.");
        }
        reader.TransferRoots();
        shape = reader.OneShape();
    } else if (format == "iges") {
        IGESControl_Reader reader;
        IFSelect_ReturnStatus stat = reader.ReadFile(path.c_str());
        if (stat != IFSelect_RetDone) {
            throw std::runtime_error("IGES file could not be read.");
        }
        reader.TransferRoots();
        shape = reader.OneShape();
    }

    if (shape.IsNull()) {
        throw std::runtime_error("CAD file is empty or failed to transfer shape.");
    }
    
    // --- 2. Mesh the CAD shape into triangles ---
    // This generates a high-quality mesh from the NURBS geometry
    // 0.1 is the linear deflection limit (meshing precision).
    BRepMesh_IncrementalMesh(shape, 0.1);

    // --- 3. Extract nodes and de-duplicate them ---
    // We use a map to ensure each node is added only once, just like the STL reader does.
    // This map stores the (x,y,z) coordinate and maps it to the index in our final mesh.nodes vector.
    std::map<std::array<double, 3>, int> node_map;
    double initial_ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
    
    // TopExp_Explorer iterates over all sub-shapes (e.g., faces) of the main shape
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next()) {
        TopoDS_Face face = TopoDS::Face(ex.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, loc);

        if (triangulation.IsNull()) {
            continue; // Skip faces that couldn't be meshed
        }

        const TColgp_Array1OfPnt& nodes = triangulation->Nodes();
        
        // Iterate over all nodes in this face's mesh
        for (Standard_Integer i = nodes.Lower(); i <= nodes.Upper(); ++i) {
            gp_Pnt pnt = nodes.Value(i).Transformed(loc);
            std::array<double, 3> coord = {pnt.X(), pnt.Y(), pnt.Z()};

            // Check if we've already added this node
            if (node_map.find(coord) == node_map.end()) {
                // This is a new node. Add it to our mesh.
                Node node;
                node.position = Eigen::Vector3d(pnt.X(), pnt.Y(), pnt.Z());
                node.temperature = initial_ambient_temp;
                node.plastic_strain = 0.0;
                node.strain = 0.0;
                node.stress = 0.0;
                node.accumulated_wear = 0.0;
                node.accumulated_damage = 0.0;
                node.status = NodeStatus::OK;
                
                mesh.nodes.push_back(node);
                node_map[coord] = static_cast<int>(mesh.nodes.size() - 1);
            }
        }
        
        // Note: The FEA solver only needs the node cloud (`mesh.nodes`).
        // It does not need the triangle connectivity (indices) because it's
        // a nodal simulation, not a full finite-element (matrix) simulation.
        // Therefore, we do not need to process `triangulation->Triangles()`.
    }

    if (mesh.nodes.empty()) {
        throw std::runtime_error("No nodes were generated from the CAD file. The file might be empty or meshing failed.");
    }
}


void Simulation::calculate_tool_life_prediction() {
    // Calculate tool life based on wear rate progression
    
    if (time_series_output.empty() || time_series_output.back()["time_s"].get<double>() <= 0) {
        predicted_tool_life_hours = 0.0; // Cannot predict if no time passed or no data
        return;
    }
    
    // Get final total wear
    double final_wear_m = time_series_output.back()["total_accumulated_wear_m"].get<double>();
    
    // Read threshold from config, with a default
    wear_threshold_mm = config_data.value("tool_life_prediction", json::object()).value("wear_threshold_mm", 0.3);
    double wear_threshold_m = wear_threshold_mm * 0.001; // Convert mm to m
    
    // Get simulation duration
    double total_time_s = time_series_output.back()["time_s"].get<double>();
    
    // Calculate average wear rate over the simulation
    double avg_wear_rate_per_s = final_wear_m / total_time_s;

    // Use average rate for linear extrapolation as a baseline
    if (avg_wear_rate_per_s <= 1e-20) { // Avoid division by zero if wear is negligible
        predicted_tool_life_hours = 200.0; // Set to max if wear rate is effectively zero
    } else {
         predicted_tool_life_hours = (wear_threshold_m / avg_wear_rate_per_s) / 3600.0; // Time = Distance / Rate
    }
    
    // Apply realistic bounds
    double min_life = 0.5;  // 30 minutes minimum
    double max_life = 200.0; // 200 hours maximum
    // Ensure prediction is finite and within bounds
    if (!std::isfinite(predicted_tool_life_hours)) {
         predicted_tool_life_hours = max_life;
    }
    predicted_tool_life_hours = std::max(min_life, std::min(max_life, predicted_tool_life_hours));

    // Adjust prediction based on fractured nodes (early fractures reduce life)
    int cumulative_fractured = 0;
    try {
        // Ensure the key exists before accessing
        if(time_series_output.back().contains("cumulative_fractured_nodes")) {
             cumulative_fractured = time_series_output.back().value("cumulative_fractured_nodes", 0);
        }
    } catch (...) {
        cumulative_fractured = 0; // Default if access fails
    }

    double node_count = 1.0;
    if (mesh.nodes.size() > 0) node_count = static_cast<double>(mesh.nodes.size());
    double frac_fractured = (node_count > 0.0) ? (static_cast<double>(cumulative_fractured) / node_count) : 0.0;

    // Reduce predicted life proportionally to the fraction fractured, with a soft cap
    if (frac_fractured > 0.0) {
        double reduction = std::min(0.9, frac_fractured * 1.5); // scale factor
        predicted_tool_life_hours *= (1.0 - reduction);
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  TOOL LIFE PREDICTION" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Wear threshold: " << std::fixed << std::setprecision(2) << wear_threshold_mm << " mm" << std::endl;
    std::cout << "  Final total wear: " << std::scientific << std::setprecision(3) << final_wear_m << " m" << std::endl;
    std::cout << "  📊 PREDICTED TOOL LIFE: " << std::fixed << std::setprecision(2) << predicted_tool_life_hours << " hours" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;
}

void Simulation::write_output() {
    std::string output_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::cout << "  Writing results to: " << output_path << std::endl;

    json output_data;
    int thread_count_meta = 0;
    #pragma omp parallel
    {
        #pragma omp single
        thread_count_meta = omp_get_num_threads();
    }
    output_data["metadata"] = {
        {"engine_version", "3.2-CAD"},
        {"calibration_version", "v1"},
        {"num_nodes", mesh.nodes.size()},
        {"openmp_threads", thread_count_meta},
        {"simulation_complete", true}
    };
    
    output_data["tool_life_prediction"] = {
        {"predicted_hours", predicted_tool_life_hours},
        {"wear_threshold_mm", wear_threshold_mm},
        {"confidence", "average_rate_extrapolation"}
    };
    
    output_data["time_series_data"] = time_series_output;
    
    if (!cfd_visualization_data.empty()) {
        output_data["cfd_particle_animation"] = cfd_visualization_data;
        std::cout << "  ✓ CFD animation data included (" << cfd_visualization_data.size() << " frames)" << std::endl;
    }

    // Final node states
    output_data["final_node_states"] = json::array();
    for (const auto& node : mesh.nodes) {
        output_data["final_node_states"].push_back({
            {"position", {node.position.x(), node.position.y(), node.position.z()}},
            {"temperature_C", node.temperature},
            {"stress_MPa", node.stress},
            {"strain", node.strain},
            {"plastic_strain", node.plastic_strain},
            {"accumulated_wear_m", node.accumulated_wear},
            {"accumulated_damage", node.accumulated_damage},
            {"status", (node.status == NodeStatus::OK) ? "OK" : "FRACTURED"},
        });
    }

    std::ofstream o(output_path);
    o << std::setw(2) << output_data << std::endl; 
    
    std::cout << "  ✓ Results written successfully\n" << std::endl;
}