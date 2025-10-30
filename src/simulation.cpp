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
// Include necessary headers for the classes used
#include "physics_models.h" 
#include "navier_stokes.h"
#include "particle_system.h" 

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
        cfd_enabled = config_data["cfd_parameters"].value("enable_cfd", false); // Robust read
        
        if (cfd_enabled) {
            std::cout << "  ✓ CFD Analysis ENABLED" << std::endl;
            
            // Initialize chip formation model
            chip_model = std::make_unique<ChipFormationModel>(config_data);
            
            // Initialize fluid solver (20x20x20 grid, 0.0005m cells = 10mm domain)
            // Store pointer in FEASolver
            fluid_solver = std::make_unique<NavierStokesSolver>(config_data); 
            fluid_solver->initialize_grid(20, 20, 20, 0.0005);
            
            // Set inlet velocity (chip flow)
            double chip_velocity_calc = chip_model->calculate_chip_velocity(sliding_velocity, 2.5); // Use local variable
            double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
            Eigen::Vector3d chip_dir(cos(rake_angle), 0, sin(rake_angle));
            fluid_solver->set_inlet_velocity(chip_dir * chip_velocity_calc); // Pass calculated value
            
            // Initialize particle system (max 2000 particles)
            // Store pointer in FEASolver
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
    
    // Initialize all nodes with proper temperature and state
    #pragma omp parallel for
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        Node& node = mesh.nodes[i];
        node.temperature = ambient_temperature;
        node.plastic_strain = 1e-6; // Small initial strain to prevent zero-division
        node.strain = 1e-6;         // Initial elastic strain
        // Use member variable material_model safely
        if(material_model) {
            node.stress = material_model->calculate_stress(ambient_temperature, node.plastic_strain, strain_rate);
        } else {
             node.stress = 0.0; // Fallback if model not initialized
             // Add error logging here if this case is unexpected
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
    // ───────────────────────────────────────────────────────
    // PHASE 1: FEA Update (Solid Mechanics + Thermal)
    // ───────────────────────────────────────────────────────
    
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    // Parallel FEA calculation loop
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
        if (node.status == NodeStatus::OK) {
            update_node_mechanics(node, dt);

            double chip_heat_flux = 0.0;
            if (cfd_enabled && fluid_solver) {
                // Get temp *from* CFD domain at node pos
                double chip_temp = fluid_solver->get_temperature_at(node.position);
                 // Ensure thermal_model is valid before calling
                if(thermal_model) {
                    chip_heat_flux = thermal_model->calculate_heat_transfer_from_chip(
                        chip_temp, node.temperature, 1e-7); // Area placeholder
                }
            }
            update_node_thermal(node, dt, chip_heat_flux); // Applies clamp internally

            if (check_node_failure(node)) {
                fractured_this_step++;
            }
        }
        
        // Track maxima
        if (node.temperature > max_temp) max_temp = node.temperature;
        if (node.stress > max_stress) max_stress = node.stress;
        total_wear += node.accumulated_wear;
    }
    
    total_fractured_nodes += fractured_this_step;
    
    // ───────────────────────────────────────────────────────
    // PHASE 2: CFD Update (Chip Flow Simulation)
    // ───────────────────────────────────────────────────────
    
    json cfd_metrics;
    // Check pointers are valid before dereferencing
    if (cfd_enabled && fluid_solver && particle_system && thermal_model && chip_model) { 

        // --- STRUCTURAL FIX APPLIED ---
        // Call the new solver function, passing the mesh and other required info directly.
        fluid_solver->update_boundaries_and_sources(*active_mesh, ambient_temperature, strain_rate, *thermal_model);
        // --- END STRUCTURAL FIX ---

        // Update fluid solver (time step)
        fluid_solver->update(dt);

        // Emit new particles (same logic as before)
        if (step_num % 2 == 0) {
             Eigen::Vector3d shear_zone(0.0, 0.0, 0.0);
             double chip_ratio = config_data["cfd_parameters"]["chip_thickness_ratio"].get<double>();
             double chip_vel = chip_model->calculate_chip_velocity(sliding_velocity, chip_ratio);
             double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
             Eigen::Vector3d emission_vel(cos(rake_angle) * chip_vel, 0, sin(rake_angle) * chip_vel);
             double avg_tool_temp = 0.0;
             // Ensure mesh is not empty before calculating average
             if (!active_mesh->nodes.empty()) {
                for(const auto& node : active_mesh->nodes) { avg_tool_temp += node.temperature; }
                avg_tool_temp /= active_mesh->nodes.size();
             } else {
                 avg_tool_temp = ambient_temperature; // Fallback
             }
             particle_system->emit_particles(shear_zone, emission_vel, avg_tool_temp, 15);
        }

        // Update particles (same logic as before)
        particle_system->update(dt, *fluid_solver);

        // Calculate CFD metrics (same logic as before)
        double friction_angle = atan(config_data["cfd_parameters"]["friction_coefficient"].get<double>());
        double rake_angle = config_data["cfd_parameters"]["rake_angle_degrees"].get<double>() * M_PI / 180.0;
        double shear_angle = chip_model->calculate_shear_angle(rake_angle, friction_angle);
        double chip_velocity_calc = chip_model->calculate_chip_velocity(sliding_velocity, 2.5); // Use local variable
        auto rake_pressures = fluid_solver->get_rake_face_pressure(Eigen::Vector3d(0,0,1));
        double avg_rake_pressure = 0.0;
        double max_rake_pressure = 0.0;
        for (double p : rake_pressures) {
            avg_rake_pressure += p;
            if (p > max_rake_pressure) max_rake_pressure = p;
        }
        avg_rake_pressure /= (rake_pressures.empty() ? 1.0 : rake_pressures.size());
        
        // Safely access particle system methods
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

        // Store particle snapshot (same logic as before)
        if (step_num % 5 == 0) {
            json frame;
            frame["step"] = step_num;
            frame["time_s"] = step_num * dt;
            frame["particles"] = particle_system->export_particles(); // Assumes this is safe
            // Ensure parent_sim is valid before accessing member
            if(parent_sim) {
                parent_sim->cfd_visualization_data.push_back(frame);
            }
        }
    }

    // ───────────────────────────────────────────────────────
    // PHASE 3: Report Progress (Same as before)
    // ───────────────────────────────────────────────────────
     std::cout << "  Step " << std::setw(4) << step_num << "/" << config_data["simulation_parameters"]["num_steps"].get<int>()
               << " │ T_max: " << std::fixed << std::setprecision(1) << std::setw(6) << max_temp << "°C"
               << " │ σ_max: " << std::fixed << std::setprecision(1) << std::setw(7) << max_stress << " MPa" // Added precision
               << " │ Wear: " << std::scientific << std::setprecision(2) << total_wear << " m"
               << " │ Failed: " << std::setw(3) << fractured_this_step;
    
    // Safely check if cfd_metrics was populated and contains the key
    if (cfd_enabled && cfd_metrics.contains("chip_flow") && cfd_metrics["chip_flow"].contains("active_particles")) {
        std::cout << " │ Particles: " << std::setw(4) << cfd_metrics["chip_flow"]["active_particles"];
    }
    std::cout << std::endl;


    // ───────────────────────────────────────────────────────
    // PHASE 4: Return Metrics (Same as before)
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

// --- update_node_mechanics remains unchanged from previous version ---
void FEASolver::update_node_mechanics(Node& node, double dt) {
    // Ensure models are valid before using
    if (!material_model || !wear_model || !failure_model) {
        // Log error or throw exception - models not initialized
        return; 
    }

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
        effective_strain_rate / strain_rate, // Using cycles=effective_strain_rate/strain_rate as proxy? Check units/logic.
        node.accumulated_damage
    ) * (0.7 + 0.3 * (stress_factor + temp_factor + strain_rate_factor) / 3.0);
    
    double thermal_sensitivity = std::min(1.0, std::max(0.0, (node.temperature - 200.0) / 800.0));
    double accumulation_chance = 0.05 + 0.15 * thermal_sensitivity;
    
    // Seed random number generator once if needed (e.g., in constructor or main)
    // srand(time(NULL)); 
    if ((static_cast<double>(rand()) / RAND_MAX) < accumulation_chance) {
        node.accumulated_damage += damage_increment;
    }
    
    double base_threshold = 0.95 + 0.1 * (static_cast<double>(rand()) / RAND_MAX);
    double temp_adjusted_threshold = base_threshold * (1.0 + 0.2 * (1.0 - thermal_sensitivity));
    
    if (node.accumulated_damage >= temp_adjusted_threshold) {
        check_node_failure(node); // This will set status and stress=0 if failed
    }
}


// --- update_node_thermal includes the CRITICAL FIX ---
void FEASolver::update_node_thermal(Node& node, double dt, double chip_heat_flux) {
    // Ensure thermal_model is valid
    if (!thermal_model) return; 

    // Heat generation from plastic work
    double heat_gen = thermal_model->calculate_heat_generation(node.stress, strain_rate);
    
    // Heat dissipation to environment
    double heat_loss_rate = thermal_model->calculate_heat_dissipation(node.temperature, 
        ambient_temperature, // Use the member variable directly
        1e-7); // Assuming surface area
    
    // Net temperature change calculation
    double dT = (heat_gen - heat_loss_rate + chip_heat_flux) * dt;
    node.temperature += dT;
    
    // Clamp to physical limits (Upper Bound - Melting Point)
    double T_melt = config_data["material_properties"]["melting_point_C"].get<double>();
    if (node.temperature > T_melt) {
        node.temperature = T_melt;
    }
    
    // --- CRITICAL FIX: Lower temperature clamp to ambient temperature ---
    if (node.temperature < ambient_temperature) {
        node.temperature = ambient_temperature;
    }
    // --- END CRITICAL FIX ---
}

// --- check_node_failure remains unchanged ---
bool FEASolver::check_node_failure(Node& node) {
    // Ensure failure_model is valid
    if (!failure_model) return false; 

    bool failed = failure_model->check_failure(node.stress, node.temperature, node.accumulated_damage);
    
    if (failed) {
        node.status = NodeStatus::FRACTURED;
        node.stress = 0.0; // Ensure stress drops on failure
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════
// SIMULATION CLASS IMPLEMENTATION (Unchanged from previous stable version)
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
        
        double initial_ambient_temp = config_data["physics_parameters"]["ambient_temperature_C"].get<double>();
        
        for (size_t i = 0; i < stl_mesh.num_vrts(); ++i) {
            const float* pos = stl_mesh.vrt_coords(i);
            Node node;
            node.position = Eigen::Vector3d(pos[0], pos[1], pos[2]);
            node.temperature = initial_ambient_temp; // Initialize correctly
            // Initialize other node members to default/zero values as needed
            node.plastic_strain = 0.0;
            node.strain = 0.0;
            node.stress = 0.0;
            node.accumulated_wear = 0.0;
            node.accumulated_damage = 0.0; // Initialize damage
            node.status = NodeStatus::OK;

            mesh.nodes.push_back(node);
        }
        
        std::cout << "  ✓ Loaded " << mesh.nodes.size() << " nodes from " << stl_path << "\n" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read STL file '" + stl_path + "': " + e.what());
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
    
    // --- Refined Quadratic Extrapolation (Optional - can be complex) ---
    // You could try fitting a quadratic curve to the wear data if linear seems too simple,
    // but the simple linear extrapolation based on average rate is often sufficient.
    // Let's stick with the robust average rate calculation for now.


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
        {"engine_version", "3.0-production"}, // UPDATED VERSION STRING
        {"num_nodes", mesh.nodes.size()},
        {"simulation_complete", true}
    };
    
    output_data["tool_life_prediction"] = {
        {"predicted_hours", predicted_tool_life_hours},
        {"wear_threshold_mm", wear_threshold_mm},
        {"confidence", "average_rate_extrapolation"} // Updated confidence metric
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
            {"strain", node.strain}, // Includes elastic + plastic
            {"plastic_strain", node.plastic_strain}, // Specifically plastic strain
            {"accumulated_wear_m", node.accumulated_wear},
            {"accumulated_damage", node.accumulated_damage}, // Include final damage
            {"status", (node.status == NodeStatus::OK) ? "OK" : "FRACTURED"},
            // Note: Add CFD nodal data here if needed for visualization
        });
    }

    std::ofstream o(output_path);
    // Use indentation of 2 for better readability but smaller file size than 4
    o << std::setw(2) << output_data << std::endl; 
    
    std::cout << "  ✓ Results written successfully\n" << std::endl;
}