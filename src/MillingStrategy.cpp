#include "MillingStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"
#include "EnergyMonitor.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

MillingStrategy::MillingStrategy(Simulation* sim) 
    : m_parent_sim(sim), m_mesh(nullptr) 
{
    if (!m_parent_sim) throw std::runtime_error("MillingStrategy: Parent sim is null.");
}

MillingStrategy::~MillingStrategy() {}

void MillingStrategy::detect_cutting_edges(Mesh& mesh) {
    if (mesh.nodes.empty()) return;
    std::cout << "[MillingStrategy] Scanning tool for cutting edges..." << std::endl;

    double max_radius_sq = 0.0;
    double min_z = 1e9;
    for (const auto& node : mesh.nodes) {
        double r2 = node.position.x()*node.position.x() + node.position.y()*node.position.y();
        if (r2 > max_radius_sq) max_radius_sq = r2;
        if (node.position.z() < min_z) min_z = node.position.z();
    }
    double max_radius = std::sqrt(max_radius_sq);
    double diameter = max_radius * 2.0;
    double zone_size = std::max(diameter * 0.05, 0.0001); 
    
    double radius_threshold = max_radius - zone_size;
    double bottom_threshold = min_z + zone_size;

    int active_count = 0;
    for (auto& node : mesh.nodes) {
        double r = std::sqrt(node.position.x()*node.position.x() + node.position.y()*node.position.y());
        if (r >= radius_threshold || node.position.z() <= bottom_threshold) {
            node.is_active_contact = true;
            active_count++;
        } else {
            node.is_active_contact = false;
        }
    }
    std::cout << "  Activated " << active_count << " nodes (Zone size: " << zone_size*1000.0 << "mm)" << std::endl;
}

void MillingStrategy::initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) {
    m_mesh = &mesh;
    m_config = config;

    detect_cutting_edges(*m_mesh);
    m_fea_solver = std::make_unique<RotationalFEA>(m_config, m_mesh);
    
    if (!m_config.contains("workpiece_setup")) {
        std::cerr << "WARNING: 'workpiece_setup' missing in input.json." << std::endl;
    }
    json wp = m_config.value("workpiece_setup", json::object());
    
    Eigen::Vector3d min_c(
        wp.value("min_corner", std::vector<double>{0.015, -0.005, 0.0}).at(0),
        wp.value("min_corner", std::vector<double>{0.015, -0.005, 0.0}).at(1),
        wp.value("min_corner", std::vector<double>{0.015, -0.005, 0.0}).at(2)
    );
    Eigen::Vector3d max_c(
        wp.value("max_corner", std::vector<double>{0.025, 0.005, 0.01}).at(0),
        wp.value("max_corner", std::vector<double>{0.025, 0.005, 0.01}).at(1),
        wp.value("max_corner", std::vector<double>{0.025, 0.005, 0.01}).at(2)
    );

    m_workpiece_model = std::make_unique<SPHWorkpieceModel>(m_config);
    m_workpiece_model->initialize_workpiece(min_c, max_c);

    bool enable_cfd = m_config["cfd_parameters"].value("enable_cfd", false);
    if (enable_cfd) {
        std::cout << "[MillingStrategy] Initializing CFD Solver (Navier-Stokes)..." << std::endl;
        m_fluid_solver = std::make_unique<NavierStokesSolver>(m_config);
        m_thermal_helper = std::make_unique<ThermalModel>(m_config);
        
        double t_min_x = 1e9, t_max_x = -1e9;
        double t_min_y = 1e9, t_max_y = -1e9;
        double t_min_z = 1e9, t_max_z = -1e9;
        for(const auto& n : mesh.nodes) {
            t_min_x = std::min(t_min_x, n.position.x()); t_max_x = std::max(t_max_x, n.position.x());
            t_min_y = std::min(t_min_y, n.position.y()); t_max_y = std::max(t_max_y, n.position.y());
            t_min_z = std::min(t_min_z, n.position.z()); t_max_z = std::max(t_max_z, n.position.z());
        }
        double pad = 0.005; 
        double width = std::max(t_max_x - t_min_x, 0.01) + 2*pad;
        double height = std::max(t_max_y - t_min_y, 0.01) + 2*pad;
        double depth = std::max(t_max_z - t_min_z, 0.01) + 2*pad;
        double cell_size = std::max(0.0005, width / 40.0); 
        int nx = (int)(width / cell_size);
        int ny = (int)(height / cell_size);
        int nz = (int)(depth / cell_size);
        
        std::cout << "[CFD] Auto-Grid: " << nx << "x" << ny << "x" << nz << " (Cell: " << cell_size << ")" << std::endl;
        m_fluid_solver->initialize_grid(nx, ny, nz, cell_size);
        m_fluid_solver->set_inlet_velocity(Eigen::Vector3d(0, -5.0, 0));
    }
}

json MillingStrategy::run_time_step(double target_dt, int step_num) {
    if (!m_fea_solver || !m_workpiece_model) throw std::runtime_error("MillingStrategy: Not initialized.");
    
    double current_time = step_num * target_dt;
    m_fea_solver->update_tool_motion(target_dt, current_time);

    double time_accumulated = 0.0;
    double safe_dt = m_fea_solver->compute_stable_time_step();
    double min_allowed_dt = m_config["simulation_parameters"].value("min_time_step_s", 1e-12); 
    if (safe_dt < min_allowed_dt) safe_dt = min_allowed_dt;
    if (safe_dt > target_dt) safe_dt = target_dt;

    if (step_num == 1) {
        int estimated_substeps = (int)(target_dt / safe_dt);
        std::cout << "[Adaptive] Target DT: " << target_dt << " | Safe DT: " << safe_dt << " | Sub-steps: " << estimated_substeps << std::endl;
    }

    json last_metrics;
    int sub_cycle_count = 0;

    while (time_accumulated < target_dt) {
        double step_dt = safe_dt;
        if (time_accumulated + step_dt > target_dt) step_dt = target_dt - time_accumulated;

        // 1. Tool-Workpiece Physics
        m_workpiece_model->interact_with_tool(*m_mesh, step_dt);
        
        // 2. NEW: Fluid-Structure Interaction (Coupling)
        if (m_fluid_solver) {
            m_workpiece_model->apply_fluid_forces(*m_fluid_solver, step_dt);
        }

        last_metrics = m_fea_solver->run_physics_step(step_dt);
        m_workpiece_model->update_step(step_dt);

        // 3. NEW: Adaptive & Sustainability Logic (Check every 100 sub-steps)
        if (sub_cycle_count % 100 == 0) {
            double s = last_metrics.value("max_stress_MPa", 0.0);
            double t = last_metrics.value("torque_Nm", 0.0);
            
            // Adaptive Control
            auto* optimizer = m_parent_sim->get_optimizer();
            if (optimizer) {
                AdaptiveCommand cmd = optimizer->monitor_process(s, t, step_dt * 100);
                if (cmd.should_update) {
                    m_fea_solver->set_feed_rate(cmd.new_feed_rate);
                    m_fea_solver->set_rotation_speed(cmd.new_rpm);
                    std::cout << "\n[ADAPTIVE AI] " << cmd.action_reason << " | New Feed: " << cmd.new_feed_rate << std::endl;
                }
            }
            
            // Energy Monitoring
            auto* energy = m_parent_sim->get_energy_monitor();
            if (energy) {
                double rpm = m_config["machining_parameters"]["rpm"];
                double omega = rpm * (2.0 * M_PI / 60.0);
                double feed_speed = m_config["machining_parameters"]["feed_rate_mm_min"].get<double>() / 60000.0;
                double feed_force = 100.0; // Approximation for monitoring
                energy->update(t, omega, feed_force, feed_speed, step_dt * 100.0);
            }
        }
        sub_cycle_count++;
        time_accumulated += step_dt;
    }

    if (m_fluid_solver && m_thermal_helper) {
        m_fluid_solver->update_boundaries_and_sources(*m_mesh, 20.0, 0.0, *m_thermal_helper);
        m_fluid_solver->update(target_dt);
        double p_fluid = m_fluid_solver->get_pressure_at(Eigen::Vector3d(0.02, 0, 0)); 
        last_metrics["cfd_pressure_MPa"] = p_fluid / 1e6;
    }

    return last_metrics;
}

json MillingStrategy::get_final_results() {
    json res;
    if (!m_parent_sim->time_series_output.empty()) {
        res = m_parent_sim->time_series_output.back();
    }
    
    // --- NEW: Post-Process Distortion ---
    if (m_workpiece_model) {
        res["distortion_analysis"] = m_workpiece_model->analyze_distortion();
    }
    
    return res;
}

json MillingStrategy::get_visualization_data() {
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