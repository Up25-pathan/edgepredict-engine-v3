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

void MillingStrategy::update_machine_state(const Eigen::Vector3d& pos, double feed, double rpm) {
    if (m_fea_solver) {
        m_fea_solver->set_absolute_position(pos);
        m_fea_solver->set_feed_rate(feed);
        m_fea_solver->set_rotation_speed(rpm);
    }
}

void MillingStrategy::set_feed_rate(double f) {
    if (m_fea_solver) m_fea_solver->set_feed_rate(f);
}

void MillingStrategy::set_rotation_speed(double s) {
    if (m_fea_solver) m_fea_solver->set_rotation_speed(s);
}

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
    double zone_size = std::max(max_radius * 2.0 * 0.05, 0.0001); 
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
    
    json wp = m_config.value("workpiece_setup", json::object());
    // Safe defaults if JSON is missing values
    std::vector<double> min_def = {0.015, -0.005, 0.0};
    std::vector<double> max_def = {0.025, 0.005, 0.01};
    
    Eigen::Vector3d min_c(
        wp.value("min_corner", min_def).at(0),
        wp.value("min_corner", min_def).at(1),
        wp.value("min_corner", min_def).at(2)
    );
    Eigen::Vector3d max_c(
        wp.value("max_corner", max_def).at(0),
        wp.value("max_corner", max_def).at(1),
        wp.value("max_corner", max_def).at(2)
    );

    m_workpiece_model = std::make_unique<SPHWorkpieceModel>(m_config);
    m_workpiece_model->initialize_workpiece(min_c, max_c);

    bool enable_cfd = false;
    if(m_config.contains("cfd_parameters")) {
        enable_cfd = m_config["cfd_parameters"].value("enable_cfd", false);
    }

    if (enable_cfd) {
        std::cout << "[MillingStrategy] Initializing CFD Solver..." << std::endl;
        m_fluid_solver = std::make_unique<NavierStokesSolver>(m_config);
        m_thermal_helper = std::make_unique<ThermalModel>(m_config);
        m_fluid_solver->initialize_grid(20, 20, 20, 0.001);
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
        std::cout << "[Adaptive] Target DT: " << target_dt << " | Safe DT: " << safe_dt << " | Sub-steps: " << (int)(target_dt / safe_dt) << std::endl;
    }

    json last_metrics;
    
    // --- GOVERNOR VARIABLES ---
    static double time_since_last_check = 0.0;
    const double CHECK_INTERVAL = 0.01; // Check only every 10ms of simulation time

    while (time_accumulated < target_dt) {
        double step_dt = safe_dt;
        if (time_accumulated + step_dt > target_dt) step_dt = target_dt - time_accumulated;

        // Physics
        m_workpiece_model->interact_with_tool(*m_mesh, step_dt);
        if (m_fluid_solver) m_workpiece_model->apply_fluid_forces(*m_fluid_solver, step_dt);
        last_metrics = m_fea_solver->run_physics_step(step_dt);
        m_workpiece_model->update_step(step_dt);

        // --- NEW: Time-Based Adaptive Logic (The Governor) ---
        time_since_last_check += step_dt;
        
        if (time_since_last_check >= CHECK_INTERVAL) {
            time_since_last_check = 0.0; // Reset timer
            
            double s = last_metrics.value("max_stress_MPa", 0.0);
            double t = last_metrics.value("torque_Nm", 0.0);
            
            auto* optimizer = m_parent_sim->get_optimizer();
            if (optimizer) {
                // Pass accumulated time (CHECK_INTERVAL) instead of tiny step_dt
                AdaptiveCommand cmd = optimizer->monitor_process(s, t, CHECK_INTERVAL);
                if (cmd.should_update) {
                    set_feed_rate(cmd.new_feed_rate);
                    set_rotation_speed(cmd.new_rpm);
                    std::cout << "\n[ADAPTIVE AI] " << cmd.action_reason << " | New Feed: " << cmd.new_feed_rate << std::endl;
                }
            }
            
            // Energy Monitor Update
            auto* energy = m_parent_sim->get_energy_monitor();
            if (energy) {
                double rpm = 0; 
                if(m_config.contains("machining_parameters")) rpm = m_config["machining_parameters"].value("rpm", 0.0);
                double omega = rpm * (2.0 * M_PI / 60.0);
                double feed_speed = 0;
                if(m_config.contains("machining_parameters")) feed_speed = m_config["machining_parameters"].value("feed_rate_mm_min", 0.0) / 60000.0;
                energy->update(t, omega, 100.0, feed_speed, CHECK_INTERVAL);
            }
        }
        
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
    if (m_workpiece_model) {
        res["distortion_analysis"] = m_workpiece_model->analyze_distortion();
    }
    return res;
}

json MillingStrategy::get_visualization_data() {
    json frame;
    frame["particles"] = json::array();
    const auto& data = m_workpiece_model->pull_data_from_gpu(); 
    for(size_t i=0; i<data.count; ++i) {
        if(data.status[i] != (int)ParticleStatus::INACTIVE) {
            frame["particles"].push_back({
                {"p", {data.pos_x[i], data.pos_y[i], data.pos_z[i]}},
                {"t", data.temperature[i]}
            });
        }
    }
    return json::array({frame});
}