#include "MillingStrategy.h"
#include "simulation.h"
#include "RotationalFEA.h"
#include "SPHWorkpieceModel.h"
#include "navier_stokes.h"
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

// --- HELPER: Smart Cutting Edge Detection ---
void MillingStrategy::detect_cutting_edges(Mesh& mesh) {
    if (mesh.nodes.empty()) return;
    
    std::cout << "[MillingStrategy] Scanning tool for cutting edges..." << std::endl;

    // 1. Analyze Tool Dimensions to find boundaries
    double max_radius_sq = 0.0;
    double min_z = 1e9;

    for (const auto& node : mesh.nodes) {
        double r2 = node.position.x()*node.position.x() + node.position.y()*node.position.y();
        if (r2 > max_radius_sq) max_radius_sq = r2;
        if (node.position.z() < min_z) min_z = node.position.z();
    }
    
    double max_radius = std::sqrt(max_radius_sq);
    // Mark "Sharp" Nodes (Outermost radius or bottom tip)
    int active_count = 0;
    double radius_threshold = max_radius * 0.98;
    double bottom_threshold = min_z + 0.002; // 2mm from bottom

    for (auto& node : mesh.nodes) {
        double r = std::sqrt(node.position.x()*node.position.x() + node.position.y()*node.position.y());
        
        bool is_side_edge = (r >= radius_threshold);
        bool is_bottom_tip = (node.position.z() <= bottom_threshold);

        if (is_side_edge || is_bottom_tip) {
            node.is_active_contact = true;
            active_count++;
        } else {
            node.is_active_contact = false;
        }
    }
    std::cout << "  Activated " << active_count << " nodes as cutting edges." << std::endl;
}

void MillingStrategy::initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) {
    m_mesh = &mesh;
    m_config = config;

    // 1. Setup Tool (Geometry & Physics)
    detect_cutting_edges(*m_mesh);
    m_fea_solver = std::make_unique<RotationalFEA>(m_config, m_mesh);
    
    // 2. Setup Workpiece (SPH)
    if (!m_config.contains("workpiece_setup")) {
        std::cerr << "WARNING: 'workpiece_setup' missing in input.json. Using defaults." << std::endl;
    }
    
    // FIX: Use value copy for JSON object
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

    // 3. Setup CFD (Optional Fluid Simulation)
    bool enable_cfd = m_config["cfd_parameters"].value("enable_cfd", false);
    if (enable_cfd) {
        std::cout << "[MillingStrategy] Initializing CFD Solver (Navier-Stokes)..." << std::endl;
        m_fluid_solver = std::make_unique<NavierStokesSolver>(m_config);
        m_thermal_helper = std::make_unique<ThermalModel>(m_config);
        
        // Initialize a 20x20x20 grid (0.002m cell size)
        m_fluid_solver->initialize_grid(20, 20, 20, 0.002);
        
        // Set inlet velocity (Coolant flow)
        double coolant_vel = 5.0; // m/s
        m_fluid_solver->set_inlet_velocity(Eigen::Vector3d(0, -coolant_vel, 0));
    }
}

json MillingStrategy::run_time_step(double target_dt, int step_num) {
    if (!m_fea_solver || !m_workpiece_model) {
        throw std::runtime_error("MillingStrategy: Not initialized.");
    }
    
    double current_time = step_num * target_dt;
    
    // 1. Tool Kinematics (Move the tool to its position for this frame)
    m_fea_solver->update_tool_motion(target_dt, current_time);

    // 2. ADAPTIVE SUB-CYCLING (The Core Physics Loop)
    double time_accumulated = 0.0;
    
    // Ask FEA for the stable time limit (CFL Condition)
    double safe_dt = m_fea_solver->compute_stable_time_step();
    
    // Safety clamps (1 nanosecond minimum, don't exceed user target)
    if (safe_dt < 1e-9) safe_dt = 1e-9;
    if (safe_dt > target_dt) safe_dt = target_dt;

    // Debug: Print sub-steps on first frame so user knows load
    if (step_num == 1) {
        int estimated_substeps = (int)(target_dt / safe_dt);
        std::cout << "[Adaptive] Target DT: " << target_dt 
                  << " | Safe DT: " << safe_dt 
                  << " | Sub-steps: " << estimated_substeps << std::endl;
    }

    json last_metrics;

    while (time_accumulated < target_dt) {
        // Calculate this micro-step duration
        double step_dt = safe_dt;
        if (time_accumulated + step_dt > target_dt) {
            step_dt = target_dt - time_accumulated;
        }

        // Run Physics Modules
        m_workpiece_model->interact_with_tool(*m_mesh, step_dt);
        last_metrics = m_fea_solver->run_physics_step(step_dt);
        m_workpiece_model->update_step(step_dt);

        time_accumulated += step_dt;
    }

    // 3. CFD Update (Run once per full frame)
    if (m_fluid_solver && m_thermal_helper) {
        // Transfer heat from tool surface to fluid
        double dummy_strain_rate = 0.0; 
        m_fluid_solver->update_boundaries_and_sources(*m_mesh, 20.0, dummy_strain_rate, *m_thermal_helper);
        m_fluid_solver->update(target_dt);
        
        // Add CFD data to output metrics
        double p_fluid = m_fluid_solver->get_pressure_at(Eigen::Vector3d(0.02, 0, 0)); 
        last_metrics["cfd_pressure_MPa"] = p_fluid / 1e6;
    }

    return last_metrics;
}

json MillingStrategy::get_final_results() {
    if (m_parent_sim->time_series_output.empty()) return {};
    return m_parent_sim->time_series_output.back();
}

json MillingStrategy::get_visualization_data() {
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