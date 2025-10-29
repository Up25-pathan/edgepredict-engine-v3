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

// ═══════════════════════════════════════════════════════════
// FEA SOLVER IMPLEMENTATION
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
    
    // Initialize CFD if enabled
    cfd_enabled = false;
    if (config_data.contains("cfd_parameters")) {
        cfd_enabled = config_data["cfd_parameters"]["enable_cfd"].get<bool>();
        
        if (cfd_enabled) {
            std::cout << "  ✓ CFD Analysis ENABLED" << std::endl;
            
            // Initialize chip formation model
            chip_model = std::make_unique<ChipFormationModel>(config_data);
            
            // Initialize fluid solver (20x20x20 grid, 0.0005m cells = 10mm domain)
            fluid_solver = std::make_unique<NavierStokesSolver>(config_data);
            fluid_solver->initialize_grid(20, 20, 20, 0.0005);
            
            // Set inlet velocity (chip flow)
            double chip_velocity = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
            double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
            Eigen::Vector3d chip_dir(cos(rake_angle), 0, sin(rake_angle));
            fluid_solver->set_inlet_velocity(chip_dir * chip_velocity);
            
            // Initialize particle system (max 2000 particles)
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
    std::cout << "  EDGEPREDICT ENGINE v2.0 - PRODUCTION SIMULATION" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Total nodes: " << mesh.nodes.size() << std::endl;
    std::cout << "  Time steps: " << num_steps << std::endl;
    std::cout << "  Time step size: " << dt << " s" << std::endl;
    
    // Initialize all nodes with proper temperature and state
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Node& node = mesh.nodes[i];
        node.temperature = ambient_temperature;
        node.plastic_strain = 1e-6; // Small initial strain to prevent zero-division
        node.strain = 1e-6;         // Initial elastic strain
        node.stress = material_model->calculate_stress(ambient_temperature, node.plastic_strain, strain_rate);
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
    // ───────────────────────────────────────────────────────
    // PHASE 1: FEA Update (Solid Mechanics + Thermal)
    // ───────────────────────────────────────────────────────
    
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    // Collect node data for CFD coupling
    std::vector<double> node_temps;
    std::vector<double> node_stresses;
    std::vector<Eigen::Vector3d> node_positions;
    
    node_temps.reserve(active_mesh->nodes.size());
    node_stresses.reserve(active_mesh->nodes.size());
    node_positions.reserve(active_mesh->nodes.size());
    
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        if (node.status == NodeStatus::OK) {
            // Mechanics update
            update_node_mechanics(node, dt);
            
            // Thermal update (chip heat flux from CFD if available)
            double chip_heat_flux = 0.0;
            if (cfd_enabled && fluid_solver) {
                double chip_temp = fluid_solver->get_temperature_at(node.position);
                chip_heat_flux = thermal_model->calculate_heat_transfer_from_chip(
                    chip_temp, node.temperature, 1e-7);
            }
            update_node_thermal(node, dt, chip_heat_flux);
            
            // Failure check
            if (check_node_failure(node)) {
                fractured_this_step++;
            }
        }
        
        // Track maxima
        if (node.temperature > max_temp) max_temp = node.temperature;
        if (node.stress > max_stress) max_stress = node.stress;
        total_wear += node.accumulated_wear;
        
        // Collect for CFD
        #pragma omp critical
        {
            node_temps.push_back(node.temperature);
            node_stresses.push_back(node.stress);
            node_positions.push_back(node.position);
        }
    }
    
    total_fractured_nodes += fractured_this_step;
    
    // ───────────────────────────────────────────────────────
    // PHASE 2: CFD Update (Chip Flow Simulation)
    // ───────────────────────────────────────────────────────
    
    json cfd_metrics;
    if (cfd_enabled && fluid_solver && particle_system) {
        // Set tool boundary from mesh nodes
        fluid_solver->set_tool_boundary(node_positions);
        
        // Apply heat sources from hot tool nodes
        for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
            // Consider nodes significantly above ambient temperature
            if (node_temps[i] > ambient_temperature + 50.0) {
                double delta_T = node_temps[i] - ambient_temperature;
                double heat_flux = thermal_model->calculate_heat_generation(node_stresses[i], strain_rate);
                fluid_solver->apply_heat_source(node_positions[i], heat_flux);
            }
        }
        
        // Update fluid solver
        fluid_solver->update(dt);
        
        // Emit new particles at shear zone (tool tip)
        if (step_num % 2 == 0) { // Every 2nd step to control particle count
            Eigen::Vector3d shear_zone(0.0, 0.0, 0.0); // Tool tip at origin
            double chip_ratio = config_data["cfd_parameters"]["chip_thickness_ratio"].get<double>();
            double chip_vel = chip_model->calculate_chip_velocity(sliding_velocity, chip_ratio);
            double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
            Eigen::Vector3d emission_vel(cos(rake_angle) * chip_vel, 0, sin(rake_angle) * chip_vel);
            
            // Use actual temperature from shear zone for particles
            double shear_zone_temp = std::accumulate(node_temps.begin(), node_temps.end(), 0.0) / node_temps.size();
            particle_system->emit_particles(shear_zone, emission_vel, shear_zone_temp, 15);
        }
        
        // Update particles using fluid field
        particle_system->update(dt, *fluid_solver);
        
        // Calculate CFD metrics
        double friction_angle = atan(config_data["cfd_parameters"]["friction_coefficient"].get<double>());
        double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
        double shear_angle = chip_model->calculate_shear_angle(rake_angle, friction_angle);
        double chip_velocity = chip_model->calculate_chip_velocity(sliding_velocity, 2.5);
        
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
            {"chip_velocity_m_s", chip_velocity},
            {"shear_angle_degrees", shear_angle * 180.0 / M_PI}
        };
        
        cfd_metrics["rake_face"] = {
            {"avg_pressure_MPa", avg_rake_pressure / 1e6},
            {"max_pressure_MPa", max_rake_pressure / 1e6},
            {"contact_length_mm", chip_model->calculate_contact_length(0.0001, rake_angle) * 1000.0}
        };
        
        // Store particle snapshot (every 5 steps to reduce data size)
        if (step_num % 5 == 0) {
            json frame;
            frame["step"] = step_num;
            frame["time_s"] = step_num * dt;
            frame["particles"] = particle_system->export_particles();
            parent_sim->cfd_visualization_data.push_back(frame);
        }
    }
    
    // ───────────────────────────────────────────────────────
    // PHASE 3: Report Progress
    // ───────────────────────────────────────────────────────
    
    std::cout << "  Step " << std::setw(4) << step_num << "/" << config_data["simulation_parameters"]["num_steps"].get<int>()
              << " │ T_max: " << std::fixed << std::setprecision(1) << std::setw(6) << max_temp << "°C"
              << " │ σ_max: " << std::setw(7) << max_stress << " MPa"
              << " │ Wear: " << std::scientific << std::setprecision(2) << total_wear << " m"
              << " │ Failed: " << std::setw(3) << fractured_this_step;
    
    if (cfd_enabled && cfd_metrics.contains("chip_flow")) {
        std::cout << " │ Particles: " << std::setw(4) << cfd_metrics["chip_flow"]["active_particles"];
    }
    std::cout << std::endl;
    
    // ───────────────────────────────────────────────────────
    // PHASE 4: Return Metrics
    // ───────────────────────────────────────────────────────
    
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

void FEASolver::update_node_mechanics(Node& node, double dt) {
    // Update plastic strain using actual time step
    double actual_strain_rate = strain_increment / dt;  // Convert increment to rate
    node.plastic_strain += strain_increment;
    
    // Calculate effective strain rate (combine base rate with increment)
    double effective_strain_rate = std::max(actual_strain_rate, strain_rate);
    
    // Calculate stress using Johnson-Cook model with effective values
    node.stress = material_model->calculate_stress(
        node.temperature, 
        node.plastic_strain,
        effective_strain_rate
    );
    
    // Calculate wear using Usui model with actual sliding conditions
    double wear_rate = wear_model->calculate_wear_rate(
        node.temperature,
        node.stress,
        sliding_velocity
    );
    node.accumulated_wear += wear_rate * dt;
    
    // Calculate progressive damage using actual stress state
    // Calculate damage factors
    double stress_factor = std::min(1.0, node.stress / 2000.0); // Assume 2000 MPa UTS
    double temp_factor = std::min(1.0, node.temperature / 1200.0); // 1200°C max temp
    double strain_rate_factor = std::min(1.0, effective_strain_rate / (100.0 * strain_rate));
    
    // Combined damage increment with thermal and strain rate effects
    double damage_increment = failure_model->calculate_damage_increment(
        node.stress,
        effective_strain_rate / strain_rate,
        node.accumulated_damage
    ) * (0.7 + 0.3 * (stress_factor + temp_factor + strain_rate_factor) / 3.0);
    
    // Progressive damage accumulation with thermal scaling
    double thermal_sensitivity = std::min(1.0, std::max(0.0, (node.temperature - 200.0) / 800.0));
    double accumulation_chance = 0.05 + 0.15 * thermal_sensitivity;
    
    // Probabilistic damage accumulation with temperature influence
    if ((static_cast<double>(rand()) / RAND_MAX) < accumulation_chance) {
        node.accumulated_damage += damage_increment;
    }
    
    // Check for failure with temperature-dependent threshold
    double base_threshold = 0.95 + 0.1 * (static_cast<double>(rand()) / RAND_MAX);
    double temp_adjusted_threshold = base_threshold * (1.0 + 0.2 * (1.0 - thermal_sensitivity));
    
    if (node.accumulated_damage >= temp_adjusted_threshold) {
        check_node_failure(node);
    }
}

void FEASolver::update_node_thermal(Node& node, double dt, double chip_heat_flux) {
    // Heat generation from plastic work
    double heat_gen = thermal_model->calculate_heat_generation(node.stress, strain_rate);
    
    // Heat dissipation to environment
    double heat_loss = thermal_model->calculate_heat_dissipation(node.temperature, 
        config_data["physics_parameters"]["ambient_temperature_C"].get<double>(), 1e-7);
    
    // Net temperature change
    double dT = (heat_gen - heat_loss + chip_heat_flux) * dt;
    node.temperature += dT;
    
    // Clamp to physical limits
    double T_melt = config_data["material_properties"]["melting_point_C"].get<double>();
    if (node.temperature > T_melt) {
        node.temperature = T_melt;
    }
}

bool FEASolver::check_node_failure(Node& node) {
    bool failed = failure_model->check_failure(node.stress, node.temperature, node.accumulated_damage);
    
    if (failed) {
        node.status = NodeStatus::FRACTURED;
        // Fractured nodes stop contributing to mechanics
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

void Simulation::load_geometry() {
    std::string stl_path = config_data["file_paths"]["tool_geometry"].get<std::string>();
    std::cout << "  Loading tool geometry: " << stl_path << std::endl;

    try {
        stl_reader::StlMesh<float, unsigned int> stl_mesh(stl_path);
        mesh.nodes.reserve(stl_mesh.num_vrts());
        
        double ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
        
        for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
            const float* pos = stl_mesh.vrt_coords(i);
            Node node;
            node.position = Eigen::Vector3d(pos[0], pos[1], pos[2]);
            node.temperature = ambient_temp;
            mesh.nodes.push_back(node);
        }
        
        std::cout << "  ✓ Loaded " << mesh.nodes.size() << " nodes from " << stl_path << "\n" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read STL file '" + stl_path + "': " + e.what());
    }
}

void Simulation::calculate_tool_life_prediction() {
    // Calculate tool life based on wear rate progression
    
    if (time_series_output.empty()) {
        predicted_tool_life_hours = 0.0;
        return;
    }
    
    // Get final total wear
    double final_wear_m = time_series_output.back()["total_accumulated_wear_m"].get<double>();
    double wear_threshold_m = wear_threshold_mm * 0.001; // Convert mm to m
    
    // Get simulation duration
    double total_time_s = time_series_output.back()["time_s"].get<double>();
    
    // Calculate current wear rate
    double wear_rate_per_s = final_wear_m / total_time_s;
    
    // Enhanced wear rate calculation with acceleration factor
    double initial_wear_rate = time_series_output.front()["total_accumulated_wear_m"].get<double>() / 
                              time_series_output.front()["time_s"].get<double>();
    double final_wear_rate = wear_rate_per_s;
    
    // Calculate wear rate acceleration
    double wear_acceleration = (final_wear_rate - initial_wear_rate) / total_time_s;
    
    // Use quadratic extrapolation for more realistic prediction
    // Time to threshold = (-b + sqrt(b^2 + 2ac))/a
    // Where: a = wear_acceleration/2, b = initial_wear_rate, c = wear_threshold_m
    double a = wear_acceleration / 2.0;
    double b = initial_wear_rate;
    double c = wear_threshold_m;
    
    if (std::abs(a) < 1e-20) {
        // Linear wear case
        predicted_tool_life_hours = (c / b) / 3600.0;
    } else {
        // Accelerating wear case
        predicted_tool_life_hours = (-b + std::sqrt(b*b + 2*a*c)) / a / 3600.0;
    }
    
    // Apply realistic bounds
    double min_life = 0.5;  // 30 minutes minimum
    double max_life = 200.0; // 200 hours maximum
    predicted_tool_life_hours = std::max(min_life, std::min(max_life, predicted_tool_life_hours));

    // Adjust prediction based on fractured nodes (early fractures reduce life)
    int cumulative_fractured = 0;
    try {
        cumulative_fractured = time_series_output.back().value("cumulative_fractured_nodes", 0);
    } catch (...) {
        cumulative_fractured = 0;
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
    std::cout << "  Wear threshold: " << wear_threshold_mm << " mm" << std::endl;
    std::cout << "  Final total wear: " << std::scientific << std::setprecision(3) << final_wear_m << " m" << std::endl;
    std::cout << "  📊 PREDICTED TOOL LIFE: " << std::fixed << std::setprecision(2) << predicted_tool_life_hours << " hours" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════\n" << std::endl;
}

void Simulation::write_output() {
    std::string output_path = config_data["file_paths"]["output_results"].get<std::string>();
    std::cout << "  Writing results to: " << output_path << std::endl;

    json output_data;
    output_data["metadata"] = {
        {"engine_version", "2.0-production"},
        {"num_nodes", mesh.nodes.size()},
        {"simulation_complete", true}
    };
    
    output_data["tool_life_prediction"] = {
        {"predicted_hours", predicted_tool_life_hours},
        {"wear_threshold_mm", wear_threshold_mm},
        {"confidence", "linear_extrapolation"}
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
            {"accumulated_wear_m", node.accumulated_wear},
            {"status", (node.status == NodeStatus::OK) ? "OK" : "FRACTURED"}
        });
    }

    std::ofstream o(output_path);
    o << std::setw(2) << output_data << std::endl;
    
    std::cout << "  ✓ Results written successfully\n" << std::endl;
}