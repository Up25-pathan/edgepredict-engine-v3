#define _USE_MATH_DEFINES
#include "TurningStrategy.h"
#include "physics_models.h" //
#include "navier_stokes.h"  //
#include "particle_system.h"//
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <cmath>

// ═══════════════════════════════════════════════════════════
// --- This is your entire OLD FEASolver class, renamed to LinearFEASolver ---
// --- It is cut/pasted from your original src/simulation.cpp file ---
// ═══════════════════════════════════════════════════════════

class LinearFEASolver {
public:
    LinearFEASolver(const json& config, Simulation* sim);
    ~LinearFEASolver();
    
    void solve(Mesh& mesh, int num_steps);
    void auto_detect_cutting_edge(Mesh& mesh);
    json solve_time_step(int step_num);

    // Make members public so the strategy can access them
    std::unique_ptr<NavierStokesSolver> fluid_solver;
    std::unique_ptr<ParticleSystem> particle_system;
    json config_data;
    Simulation* parent_sim;
    Mesh* active_mesh;
    int total_fractured_nodes;
    double dt, strain_rate, strain_increment, sliding_velocity, ambient_temperature;
    std::unique_ptr<JohnsonCookModel> material_model;
    std::unique_ptr<UsuiWearModel> wear_model;
    std::unique_ptr<ThermalModel> thermal_model;
    std::unique_ptr<FailureCriterion> failure_model;
    std::unique_ptr<ChipFormationModel> chip_model;
    bool cfd_enabled;

private:
    void update_node_mechanics(Node& node, double dt, bool is_in_contact_zone);
    void update_node_thermal(Node& node, double dt, double chip_heat_flux, bool is_in_contact_zone);
    bool check_node_failure(Node& node);
};

// --- Implementation of your original FEASolver (now LinearFEASolver) ---
// [This code is cut directly from your src/simulation.cpp]

LinearFEASolver::LinearFEASolver(const json& config, Simulation* sim) 
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

LinearFEASolver::~LinearFEASolver() = default;

void LinearFEASolver::solve(Mesh& mesh, int num_steps) {
    active_mesh = &mesh;
    auto_detect_cutting_edge(mesh); 

    std::cout << "  [LinearFEASolver] Running..." << std::endl;
    
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
        // Note: The strategy will handle progress reporting
    }
}

json LinearFEASolver::solve_time_step(int step_num) {
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < active_mesh->nodes.size(); ++i) {
        Node& node = active_mesh->nodes[i];
        
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
        
        auto rake_pressures = fluid_solver->get_rake_face_pressure(Eigen::Vector3d(0.2, 0, 0.8).normalized());
        double max_rake_pressure = 0.0;
        for (double p : rake_pressures) {
             if (p > max_rake_pressure) max_rake_pressure = p;
        }

        if (max_rake_pressure < 1e-6) {
             double specific_energy = config_data["material_properties"].value("specific_cutting_energy_MPa", 3000.0) * 1e6;
             max_rake_pressure = specific_energy * (std::min(1.0, strain_rate / 10000.0));
        }

        double reported_chip_temp = std::min(particle_system->get_average_temperature(), T_melt_limit);

        cfd_metrics["chip_flow"] = {
            {"avg_chip_temperature_C", reported_chip_temp},
            {"chip_velocity_m_s", chip_model->calculate_chip_velocity(sliding_velocity, 2.5)}
        };
        cfd_metrics["rake_face"] = { {"avg_pressure_MPa", max_rake_pressure / 1e6} };

        if (step_num % 5 == 0) {
            json frame;
            frame["step"] = step_num;
            frame["particles"] = particle_system->export_particles();
            parent_sim->cfd_visualization_data.push_back(frame);
        }
    }

    json metrics;
    metrics["max_temperature_C"] = max_temp;
    metrics["max_stress_MPa"] = max_stress;
    metrics["total_accumulated_wear_m"] = total_wear;
    metrics["cumulative_fractured_nodes"] = total_fractured_nodes;
    if (!cfd_metrics.empty()) metrics["cfd"] = cfd_metrics;
    return metrics;
}

void LinearFEASolver::update_node_mechanics(Node& node, double dt, bool is_in_contact_zone) {
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

void LinearFEASolver::update_node_thermal(Node& node, double dt, double chip_heat_flux, bool is_in_contact_zone) {
    if (!thermal_model) return;
    double heat_gen = is_in_contact_zone ? thermal_model->calculate_heat_generation(node.stress, strain_rate) : 0.0;
    double heat_loss = thermal_model->calculate_heat_dissipation(node.temperature, ambient_temperature, 1e-7);
    
    node.temperature += (heat_gen - heat_loss + (is_in_contact_zone ? chip_heat_flux : 0.0)) * dt;

    double T_melt = config_data["material_properties"]["melting_point_C"].get<double>();

    if (node.temperature > T_melt) node.temperature = T_melt;
    if (node.temperature < ambient_temperature) node.temperature = ambient_temperature;
}

bool LinearFEASolver::check_node_failure(Node& node) {
    if (node.accumulated_damage >= 1.0) {
        node.status = NodeStatus::FRACTURED;
        node.stress = 0.0;
        return true;
    }
    return false;
}

// Your original auto_detect_cutting_edge function
void LinearFEASolver::auto_detect_cutting_edge(Mesh& mesh) {
    std::cout << "  [Auto-Detect] Scanning tool geometry for cutting tip..." << std::endl;
    if (mesh.nodes.empty()) return;
    size_t tip_index = 0;
    double max_score = -1e9;
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        const auto& pos = mesh.nodes[i].position;
        double score = (pos.x() * 0.7) + (pos.z() * 0.3); 
        if (score > max_score) {
            max_score = score;
            tip_index = i;
        }
    }
    Eigen::Vector3d tip_position = mesh.nodes[tip_index].position;
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
// --- This is the new "wrapper" code that implements the interface ---
// ═══════════════════════════════════════════════════════════

TurningStrategy::TurningStrategy(Simulation* sim) 
    : m_parent_sim(sim), m_mesh(nullptr), m_config(sim->config_data)
{
    // Create the "legacy" solver
    m_fea_solver = std::make_unique<LinearFEASolver>(m_config, m_parent_sim);
    std::cout << "[Strategy] TurningStrategy (Legacy Core) created." << std::endl;
}

TurningStrategy::~TurningStrategy() {}

void TurningStrategy::initialize(Mesh& mesh, const TopoDS_Shape& /*cad_shape*/, const json& config) {
    std::cout << "[TurningStrategy] Initializing..." << std::endl;
    m_mesh = &mesh;
    m_config = config;
    
    // Set the mesh for the solver
    m_fea_solver->active_mesh = m_mesh;

    // Run the old auto-detection logic
    m_fea_solver->auto_detect_cutting_edge(*m_mesh); 

    // Initialize node states
    #pragma omp parallel for
    for (size_t i = 0; i < m_mesh->nodes.size(); ++i) {
        Node& node = m_mesh->nodes[i];
        node.temperature = m_fea_solver->ambient_temperature;
        node.plastic_strain = 1e-6; 
        node.strain = 1e-6;         
        node.stress = 0.0;
        node.accumulated_wear = 0.0;
        node.status = NodeStatus::OK;
    }
    std::cout << "[TurningStrategy] Ready." << std::endl;
}

json TurningStrategy::run_time_step(double /*dt*/, int step_num) {
    if (!m_fea_solver) {
        throw std::runtime_error("TurningStrategy: Not initialized.");
    }
    
    // Just call the old solver's time step function
    return m_fea_solver->solve_time_step(step_num);
}

json TurningStrategy::get_final_results() {
    // Package up the final metrics from the time series
    if (m_parent_sim->time_series_output.empty()) {
        return {
            {"total_accumulated_wear_m", 0.0},
            {"total_time_s", 0.0},
            {"final_fractured_nodes", 0}
        };
    }
    
    const auto& last_metrics = m_parent_sim->time_series_output.back();
    return {
        {"total_accumulated_wear_m", last_metrics.value("total_accumulated_wear_m", 0.0)},
        {"total_time_s", last_metrics.value("time_s", 0.0)},
        {"final_fractured_nodes", last_metrics.value("cumulative_fractured_nodes", 0)}
    };
}

json TurningStrategy::get_visualization_data() {
    // Return the particle data that was collected by the parent sim
    if (m_fea_solver->cfd_enabled) {
        return m_parent_sim->cfd_visualization_data;
    }
    // Return an empty array if CFD was off
    return json::array();
}