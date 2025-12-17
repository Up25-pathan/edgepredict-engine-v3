#include "TurningStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

TurningStrategy::TurningStrategy(Simulation* sim) 
    : m_parent_sim(sim), m_mesh(nullptr) 
{
    if (!m_parent_sim) throw std::runtime_error("TurningStrategy: Parent sim is null.");
}

TurningStrategy::~TurningStrategy() {}

// --- HELPER: Smart Insert Nose Detection ---
void TurningStrategy::detect_cutting_edges(Mesh& mesh) {
    if (mesh.nodes.empty()) return;
    
    std::cout << "[TurningStrategy] Scanning insert nose..." << std::endl;
    double max_proj = -1e9; 
    int tip_idx = -1;
    
    for (size_t i=0; i<mesh.nodes.size(); ++i) {
        // Find the node furthest in the -X/-Z direction (The Nose)
        double proj = -(mesh.nodes[i].position.x() + mesh.nodes[i].position.z());
        if (proj > max_proj) { 
            max_proj = proj; 
            tip_idx = i; 
        }
    }
    
    if (tip_idx == -1) return;
    
    Eigen::Vector3d tip_pos = mesh.nodes[tip_idx].position;
    int count = 0;
    // Active zone: 2mm around the tip
    for (auto& n : mesh.nodes) {
        if ((n.position - tip_pos).norm() <= 0.002) { 
            n.is_active_contact = true; 
            count++; 
        } else {
            n.is_active_contact = false;
        }
    }
    std::cout << "  Activated " << count << " nodes on insert nose." << std::endl;
}

void TurningStrategy::initialize(Mesh& mesh, const TopoDS_Shape& cad, const json& config) {
    m_mesh = &mesh; 
    m_config = config;

    // 1. Tool Setup
    detect_cutting_edges(*m_mesh);
    m_fea_solver = std::make_unique<RotationalFEA>(m_config, m_mesh);
    
    // 2. Workpiece Setup
    if (!m_config.contains("workpiece_setup")) {
        std::cerr << "WARNING: 'workpiece_setup' missing. Using defaults." << std::endl;
    }
    
    // FIX: JSON value copy
    json wp = m_config.value("workpiece_setup", json::object());
    
    Eigen::Vector3d min_c(
        wp.value("min_corner", std::vector<double>{0,0,0})[0], 
        wp.value("min_corner", std::vector<double>{0,0,0})[1], 
        wp.value("min_corner", std::vector<double>{0,0,0})[2]
    );
    Eigen::Vector3d max_c(
        wp.value("max_corner", std::vector<double>{0,0,0})[0], 
        wp.value("max_corner", std::vector<double>{0,0,0})[1], 
        wp.value("max_corner", std::vector<double>{0,0,0})[2]
    );
    
    m_workpiece_model = std::make_unique<SPHWorkpieceModel>(m_config);
    m_workpiece_model->initialize_workpiece(min_c, max_c);

    // 3. CFD Setup
    if (m_config["cfd_parameters"].value("enable_cfd", false)) {
        m_fluid_solver = std::make_unique<NavierStokesSolver>(m_config);
        m_thermal_helper = std::make_unique<ThermalModel>(m_config);
        m_fluid_solver->initialize_grid(20, 20, 20, 0.001);
        // Side flow for turning
        m_fluid_solver->set_inlet_velocity(Eigen::Vector3d(5.0, 0, 0));
    }
}

json TurningStrategy::run_time_step(double target_dt, int step_num) {
    if (!m_fea_solver || !m_workpiece_model) throw std::runtime_error("Not initialized.");
    
    double current_time = step_num * target_dt;

    // 1. Tool Motion
    m_fea_solver->update_tool_motion(target_dt, current_time);
    
    // 2. Adaptive Sub-Cycling
    double time_accumulated = 0.0;
    double safe_dt = m_fea_solver->compute_stable_time_step();
    
    // Clamp DT for performance (prevent infinite recursion on stiff materials)
    if (safe_dt < 5e-8) safe_dt = 5e-8; 
    if (safe_dt > target_dt) safe_dt = target_dt;

    if (step_num == 1) {
        std::cout << "[Adaptive] Target: " << target_dt 
                  << " | Safe: " << safe_dt 
                  << " | SubSteps: " << (int)(target_dt/safe_dt) << std::endl;
    }

    json metrics;
    while (time_accumulated < target_dt) {
        double dt = std::min(safe_dt, target_dt - time_accumulated);
        
        m_workpiece_model->interact_with_tool(*m_mesh, dt);
        metrics = m_fea_solver->run_physics_step(dt);
        m_workpiece_model->update_step(dt);
        
        time_accumulated += dt;
    }
    
    // 3. CFD Update
    if (m_fluid_solver) {
        m_fluid_solver->update_boundaries_and_sources(*m_mesh, 20.0, 0.0, *m_thermal_helper);
        m_fluid_solver->update(target_dt);
    }
    
    return metrics;
}

json TurningStrategy::get_final_results() {
    if (m_parent_sim->time_series_output.empty()) return {};
    return m_parent_sim->time_series_output.back();
}

json TurningStrategy::get_visualization_data() {
    json frame; 
    frame["particles"] = json::array();
    
    // Use new Accessor for Data-Oriented SPH model
    const auto& data = m_workpiece_model->get_data();
    
    for(size_t i=0; i<data.count; ++i) {
        if(data.status[i] != (int)ParticleStatus::INACTIVE) {
            frame["particles"].push_back({
                {"p", {data.position[i].x(), data.position[i].y(), data.position[i].z()}}, 
                {"t", data.temperature[i]}
            });
        }
    }
    return json::array({frame});
}