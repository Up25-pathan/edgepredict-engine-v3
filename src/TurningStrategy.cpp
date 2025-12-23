#include "TurningStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"
#include "EnergyMonitor.h" // NEW
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

void TurningStrategy::detect_cutting_edges(Mesh& mesh) {
    if (mesh.nodes.empty()) return;
    std::cout << "[TurningStrategy] Scanning insert nose..." << std::endl;
    double min_x=1e9, max_x=-1e9, min_z=1e9, max_z=-1e9;
    double max_proj = -1e9; 
    int tip_idx = -1;
    for (size_t i=0; i<mesh.nodes.size(); ++i) {
        min_x = std::min(min_x, mesh.nodes[i].position.x());
        max_x = std::max(max_x, mesh.nodes[i].position.x());
        min_z = std::min(min_z, mesh.nodes[i].position.z());
        max_z = std::max(max_z, mesh.nodes[i].position.z());
        double proj = -(mesh.nodes[i].position.x() + mesh.nodes[i].position.z());
        if (proj > max_proj) { max_proj = proj; tip_idx = i; }
    }
    if (tip_idx == -1) return;
    double tool_diag = std::sqrt(std::pow(max_x-min_x, 2) + std::pow(max_z-min_z, 2));
    double zone = std::max(tool_diag * 0.05, 0.0001); 
    Eigen::Vector3d tip_pos = mesh.nodes[tip_idx].position;
    int count = 0;
    for (auto& n : mesh.nodes) {
        if ((n.position - tip_pos).norm() <= zone) { 
            n.is_active_contact = true; count++; 
        } else {
            n.is_active_contact = false;
        }
    }
    std::cout << "  Activated " << count << " nodes on insert nose (Zone: " << zone*1000.0 << "mm)." << std::endl;
}

void TurningStrategy::initialize(Mesh& mesh, const TopoDS_Shape& cad, const json& config) {
    m_mesh = &mesh; 
    m_config = config;
    detect_cutting_edges(*m_mesh);
    m_fea_solver = std::make_unique<RotationalFEA>(m_config, m_mesh);
    
    if (!m_config.contains("workpiece_setup")) std::cerr << "WARNING: 'workpiece_setup' missing." << std::endl;
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

    if (m_config["cfd_parameters"].value("enable_cfd", false)) {
        m_fluid_solver = std::make_unique<NavierStokesSolver>(m_config);
        m_thermal_helper = std::make_unique<ThermalModel>(m_config);
        m_fluid_solver->initialize_grid(20, 20, 20, 0.001);
        m_fluid_solver->set_inlet_velocity(Eigen::Vector3d(5.0, 0, 0));
    }
}

json TurningStrategy::run_time_step(double target_dt, int step_num) {
    if (!m_fea_solver || !m_workpiece_model) throw std::runtime_error("Not initialized.");
    
    double current_time = step_num * target_dt;
    m_fea_solver->update_tool_motion(target_dt, current_time);
    
    double time_accumulated = 0.0;
    double safe_dt = m_fea_solver->compute_stable_time_step();
    double min_allowed_dt = m_config["simulation_parameters"].value("min_time_step_s", 1e-12);
    if (safe_dt < min_allowed_dt) safe_dt = min_allowed_dt; 
    if (safe_dt > target_dt) safe_dt = target_dt;

    if (step_num == 1) {
        std::cout << "[Adaptive] Target: " << target_dt << " | Safe: " << safe_dt << " | SubSteps: " << (int)(target_dt/safe_dt) << std::endl;
    }

    json metrics;
    int sub_cycle_count = 0;

    while (time_accumulated < target_dt) {
        double dt = std::min(safe_dt, target_dt - time_accumulated);
        
        // 1. Physics & FSI
        m_workpiece_model->interact_with_tool(*m_mesh, dt);
        if (m_fluid_solver) {
             m_workpiece_model->apply_fluid_forces(*m_fluid_solver, dt);
        }

        metrics = m_fea_solver->run_physics_step(dt);
        m_workpiece_model->update_step(dt);
        
        // 2. Adaptive & Eco Logic
        if (sub_cycle_count % 100 == 0) {
            double s = metrics.value("max_stress_MPa", 0.0);
            double t = metrics.value("torque_Nm", 0.0);
            
            auto* optimizer = m_parent_sim->get_optimizer();
            if (optimizer) {
                AdaptiveCommand cmd = optimizer->monitor_process(s, t, dt * 100);
                if (cmd.should_update) {
                    m_fea_solver->set_feed_rate(cmd.new_feed_rate);
                    m_fea_solver->set_rotation_speed(cmd.new_rpm);
                    std::cout << "\n[ADAPTIVE AI] " << cmd.action_reason << " | New Feed: " << cmd.new_feed_rate << std::endl;
                }
            }

            auto* energy = m_parent_sim->get_energy_monitor();
            if (energy) {
                double rpm = m_config["machining_parameters"]["rpm"];
                double omega = rpm * (2.0 * 3.14159 / 60.0);
                double feed_speed = m_config["machining_parameters"]["feed_rate_mm_min"].get<double>() / 60000.0;
                energy->update(t, omega, 100.0, feed_speed, dt * 100.0);
            }
        }
        sub_cycle_count++;
        time_accumulated += dt;
    }
    
    if (m_fluid_solver) {
        m_fluid_solver->update_boundaries_and_sources(*m_mesh, 20.0, 0.0, *m_thermal_helper);
        m_fluid_solver->update(target_dt);
    }
    
    return metrics;
}

json TurningStrategy::get_final_results() {
    json res;
    if (!m_parent_sim->time_series_output.empty()) {
        res = m_parent_sim->time_series_output.back();
    }
    if (m_workpiece_model) {
        res["distortion_analysis"] = m_workpiece_model->analyze_distortion();
    }
    return res;
}

json TurningStrategy::get_visualization_data() {
    json frame; 
    frame["particles"] = json::array();
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