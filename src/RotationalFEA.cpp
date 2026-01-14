#include "RotationalFEA.h"
#include <iostream>
#include <cmath>
#include <algorithm>

RotationalFEA::RotationalFEA(const json& config, Mesh* mesh) 
    : m_mesh(mesh), m_config(config) 
{
    initialize_physics();
}

RotationalFEA::~RotationalFEA() {}

void RotationalFEA::initialize_physics() {
    if (m_config.contains("material_properties")) {
        auto& mat = m_config["material_properties"];
        m_density = mat.value("density_kg_m3", 7800.0);
        m_youngs_modulus = mat.value("youngs_modulus_Pa", 200e9);
        m_damping = mat.value("damping_ratio", 0.95);
    } else {
        m_density = 7800.0;
        m_youngs_modulus = 200e9;
        m_damping = 0.95;
    }

    if (m_config.contains("machining_parameters")) {
        double rpm = m_config["machining_parameters"].value("rpm", 0.0);
        m_rotation_speed_rad_s = rpm * (2.0 * M_PI / 60.0);
        
        double feed = m_config["machining_parameters"].value("feed_rate_mm_min", 0.0);
        m_feed_speed_m_s = (feed / 60.0) * 0.001; 
    } else {
        m_rotation_speed_rad_s = 0.0;
        m_feed_speed_m_s = 0.0;
    }

    m_rotation_axis = Eigen::Vector3d(0, 0, 1);
    m_center_of_mass = Eigen::Vector3d::Zero();
    
    // --- NEW: Init Fatigue ---
    // Safely check for nested keys to prevent crashes
    if (m_config.contains("material_properties") && 
        m_config["material_properties"].contains("failure_criterion")) {
        m_fatigue_limit_MPa = m_config["material_properties"]["failure_criterion"].value("fatigue_limit_MPa", 300.0) * 1e6;
    } else {
        m_fatigue_limit_MPa = 300.0 * 1e6;
    }

    m_accumulated_damage = 0.0;
    m_tool_broken = false;

    compute_lumped_mass();
    build_stiffness_network();
}

void RotationalFEA::set_absolute_position(const Eigen::Vector3d& target_pos_m) {
    if (!m_mesh || m_mesh->nodes.empty()) return;
    Eigen::Vector3d current_com = Eigen::Vector3d::Zero();
    for (const auto& node : m_mesh->nodes) current_com += node.position;
    current_com /= (double)m_mesh->nodes.size();
    Eigen::Vector3d shift = target_pos_m - current_com;
    #pragma omp parallel for
    for (size_t i = 0; i < m_mesh->nodes.size(); ++i) {
        m_mesh->nodes[i].position += shift;
    }
    m_center_of_mass = target_pos_m;
}

void RotationalFEA::set_rotation_speed(double rpm) {
    m_rotation_speed_rad_s = rpm * (2.0 * M_PI / 60.0);
}

void RotationalFEA::set_feed_rate(double mm_min) {
    m_feed_speed_m_s = (mm_min / 60.0) * 0.001;
}

void RotationalFEA::compute_lumped_mass() {
    if (!m_mesh || m_mesh->nodes.empty()) return;
    
    // --- FIX: Increased Node Mass for Time Scaling ---
    // Original 1e-6 was too light, causing 1e-9 dt.
    // 0.001 (1g) makes the simulation ~1000x faster while keeping reasonable stability.
    double node_mass = 0.005; 
    
    for (auto& node : m_mesh->nodes) {
        node.mass = node_mass;
        m_center_of_mass += node.position;
    }
    m_center_of_mass /= (double)m_mesh->nodes.size();
}

void RotationalFEA::build_stiffness_network() {
    if (!m_mesh) return;
    std::vector<std::pair<int, int>> edges;
    for (size_t i = 0; i < m_mesh->triangle_indices.size(); i += 3) {
        int n1 = m_mesh->triangle_indices[i];
        int n2 = m_mesh->triangle_indices[i+1];
        int n3 = m_mesh->triangle_indices[i+2];
        edges.push_back({std::min(n1, n2), std::max(n1, n2)});
        edges.push_back({std::min(n2, n3), std::max(n2, n3)});
        edges.push_back({std::min(n3, n1), std::max(n3, n1)});
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    m_springs.clear();
    for (const auto& edge : edges) {
        SpringEdge s;
        s.node_a = edge.first;
        s.node_b = edge.second;
        Eigen::Vector3d diff = m_mesh->nodes[s.node_a].position - m_mesh->nodes[s.node_b].position;
        s.rest_length = diff.norm();
        s.stiffness = (m_youngs_modulus * 1e-6) / (s.rest_length + 1e-9);
        m_springs.push_back(s);
    }
    std::cout << "[FEA] Built stiffness network: " << m_springs.size() << " springs." << std::endl;
}

void RotationalFEA::update_tool_motion(double dt, double current_time) {
    if (!m_mesh) return;
    double angle = m_rotation_speed_rad_s * dt;
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    Eigen::Vector3d feed_vec(m_feed_speed_m_s * dt, 0, 0); 

    #pragma omp parallel for
    for (size_t i = 0; i < m_mesh->nodes.size(); ++i) {
        Node& n = m_mesh->nodes[i];
        double dx = n.position.x() - m_center_of_mass.x();
        double dy = n.position.y() - m_center_of_mass.y();
        double new_dx = dx * cos_a - dy * sin_a;
        double new_dy = dx * sin_a + dy * cos_a;
        n.position.x() = m_center_of_mass.x() + new_dx;
        n.position.y() = m_center_of_mass.y() + new_dy;
        n.position += feed_vec;
        n.velocity = Eigen::Vector3d(-dy * m_rotation_speed_rad_s, dx * m_rotation_speed_rad_s, 0) + Eigen::Vector3d(m_feed_speed_m_s, 0, 0);
    }
    m_center_of_mass += feed_vec;
}

double RotationalFEA::compute_stable_time_step() const {
    double min_dt = 1.0; 
    // Mass Scale 0.001
    double node_mass = 0.001; 
    
    for (const auto& s : m_springs) {
        // dt = 0.5 * sqrt(m/k)
        double dt_local = 0.5 * std::sqrt(node_mass / s.stiffness);
        if (dt_local < min_dt) min_dt = dt_local;
    }
    return min_dt;
}

void RotationalFEA::compute_internal_forces() {
    #pragma omp parallel for
    for (size_t i = 0; i < m_springs.size(); ++i) {
        const SpringEdge& s = m_springs[i];
        Node& n1 = m_mesh->nodes[s.node_a];
        Node& n2 = m_mesh->nodes[s.node_b];
        Eigen::Vector3d diff = n2.position - n1.position;
        double len = diff.norm();
        double deformation = len - s.rest_length;
        Eigen::Vector3d force_dir = diff / (len + 1e-12);
        Eigen::Vector3d f = s.stiffness * deformation * force_dir;

        #pragma omp atomic
        n1.force += f;
        #pragma omp atomic
        n2.force -= f;
        
        double stress = std::abs(deformation) * m_youngs_modulus;
        #pragma omp atomic
        n1.stress = std::max(n1.stress, stress);
        #pragma omp atomic
        n2.stress = std::max(n2.stress, stress);
    }
}

json RotationalFEA::run_physics_step(double dt) {
    compute_internal_forces();
    
    double max_stress = 0.0;
    double max_temp = 0.0;
    for (const auto& n : m_mesh->nodes) {
        if (n.stress > max_stress) max_stress = n.stress;
        if (n.temperature > max_temp) max_temp = n.temperature;
    }

    if (max_stress > m_fatigue_limit_MPa && !m_tool_broken) {
        double ultimate = 1000.0 * 1e6; // Default fallback
        if(m_config.contains("material_properties") && 
           m_config["material_properties"].contains("failure_criterion")) {
            ultimate = m_config["material_properties"]["failure_criterion"].value("ultimate_tensile_strength_MPa", 1000.0) * 1e6;
        }

        double stress_ratio = max_stress / ultimate;
        if (stress_ratio > 1.0) stress_ratio = 1.0;
        
        double damage_increment = std::pow(stress_ratio, 4.0) * dt * 10.0;
        m_accumulated_damage += damage_increment;

        if (m_accumulated_damage >= 1.0) {
            m_tool_broken = true;
            std::cout << "\n[CRITICAL FAILURE] Tool Fracture Predicted! Damage > 100%" << std::endl;
        }
    }

    json metrics;
    metrics["max_stress_MPa"] = max_stress / 1e6;
    metrics["max_temp_C"] = max_temp;
    
    double torque = 0.0;
    for (const auto& n : m_mesh->nodes) {
        torque += (n.position.x() * n.force.y() - n.position.y() * n.force.x());
    }
    metrics["torque_Nm"] = std::abs(torque);
    metrics["tool_health_percent"] = (1.0 - m_accumulated_damage) * 100.0;
    metrics["tool_status"] = m_tool_broken ? "BROKEN" : "OK";

    if (m_mesh->nodes.empty()) metrics["max_velocity_m_s"] = 0.0;
    else metrics["max_velocity_m_s"] = m_mesh->nodes[0].velocity.norm();

    return metrics;
}

double RotationalFEA::get_max_stress() const {
    double m = 0.0;
    if (m_mesh) {
        for (const auto& n : m_mesh->nodes) m = std::max(m, n.stress);
    }
    return m;
}