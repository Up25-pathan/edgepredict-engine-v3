#define _USE_MATH_DEFINES
#include "RotationalFEA.h"
#include "physics_models.h" //
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cmath>

// ═══════════════════════════════════════════════════════════
// ROTATIONAL FEA SOLVER (The "Muscle")
// ═══════════════════════════════════════════════════════════

RotationalFEA::RotationalFEA(const json& config, Mesh* mesh_ptr) 
    : m_config_data(config), 
      m_mesh(mesh_ptr), 
      m_total_fractured_nodes(0)
{
    if (!m_mesh) {
        throw std::runtime_error("RotationalFEA: Mesh pointer is null.");
    }

    // Load core physics models (these are universal)
    m_material_model = std::make_unique<JohnsonCookModel>(m_config_data);
    m_wear_model = std::make_unique<UsuiWearModel>(m_config_data);
    m_thermal_model = std::make_unique<ThermalModel>(m_config_data);
    m_failure_model = std::make_unique<FailureCriterion>(m_config_data);
    
    // Load simulation parameters
    const auto& physics = m_config_data["physics_parameters"];
    m_ambient_temperature = physics["ambient_temperature_C"].get<double>();
    
    // --- NEW R&D PARAMETERS ---
    m_spindle_speed_rpm = physics["spindle_speed_rpm"].get<double>();
    m_feed_rate_mm_per_rev = physics["feed_rate_mm_per_rev"].get<double>();
    
    // Read tool_axis and feed_direction vectors
    m_tool_axis = Eigen::Vector3d(physics["tool_axis"][0].get<double>(),
                                  physics["tool_axis"][1].get<double>(),
                                  physics["tool_axis"][2].get<double>()).normalized();
    m_feed_direction = Eigen::Vector3d(physics["feed_direction"][0].get<double>(),
                                       physics["feed_direction"][1].get<double>(),
                                       physics["feed_direction"][2].get<double>()).normalized();

    // Initialize tool transform and angle
    m_tool_transform = Eigen::Affine3d::Identity();
    m_current_rotation_angle_rad = 0.0;

    // Calculate derived physics constants
    m_angular_velocity_rad_per_s = m_spindle_speed_rpm * (2.0 * M_PI) / 60.0;
    m_feed_velocity_m_per_s = m_feed_rate_mm_per_rev * (m_spindle_speed_rpm / 60.0) / 1000.0; // mm/rev -> m/s

    std::cout << "[RotationalFEA] R&D Core Initialized." << std::endl;
    std::cout << "[RotationalFEA]   RPM: " << m_spindle_speed_rpm << std::endl;
    std::cout << "[RotationalFEA]   Feed (m/s): " << m_feed_velocity_m_per_s << std::endl;
    
    // Initialize all nodes on the mesh
    #pragma omp parallel for
    for (size_t i = 0; i < m_mesh->nodes.size(); ++i) {
        Node& node = m_mesh->nodes[i];
        node.temperature = m_ambient_temperature;
        node.plastic_strain = 1e-6; 
        node.strain = 1e-6;         
        node.stress = 0.0;
        node.accumulated_wear = 0.0;
        node.status = NodeStatus::OK;
        // is_active_contact is set by GeometricAnalyzer in the strategy
    }
}

RotationalFEA::~RotationalFEA() = default;

// This function just updates the tool's 3D position
void RotationalFEA::update_tool_motion(double dt) {
    double delta_rotation = m_angular_velocity_rad_per_s * dt;
    double delta_translation = m_feed_velocity_m_per_s * dt;

    // Create the transforms for this time step
    Eigen::Affine3d rotation = Eigen::Affine3d(Eigen::AngleAxisd(delta_rotation, m_tool_axis));
    Eigen::Affine3d translation = Eigen::Affine3d(Eigen::Translation3d(m_feed_direction * delta_translation));
    
    // Apply the transforms to the tool's global state
    m_tool_transform = translation * rotation * m_tool_transform;
    m_current_rotation_angle_rad += delta_rotation;
}

// This function runs the physics simulation *on the tool mesh*
json RotationalFEA::run_physics_step(double dt) {
    double max_temp = 0.0;
    double max_stress = 0.0;
    double total_wear = 0.0;
    int fractured_this_step = 0;
    
    // We parallelize the loop over all nodes in the tool mesh
    #pragma omp parallel for reduction(max:max_temp, max_stress) reduction(+:total_wear, fractured_this_step)
    for (size_t i = 0; i < m_mesh->nodes.size(); ++i) {
        Node& node = m_mesh->nodes[i];
        
        // This flag is now set externally by the SPHWorkpieceModel's interaction
        bool is_in_contact_zone = node.is_in_contact; 
        
        if (node.status == NodeStatus::OK) {
            
            // --- "Smart" R&D Physics Calculation ---
            double local_strain_rate = 0.0;
            double local_sliding_velocity = 0.0;

            if (is_in_contact_zone) {
                // This is the core of the R&D engine:
                // We calculate physics *per-node* based on its geometry.
                
                // 1. Calculate tangential velocity (v = ω * r)
                //    r = perpendicular distance from node to tool axis
                Eigen::Vector3d r_vec = node.position - (node.position.dot(m_tool_axis) * m_tool_axis);
                double r_dist = r_vec.norm();
                local_sliding_velocity = m_angular_velocity_rad_per_s * r_dist;
                
                // 2. Estimate local strain rate (simplified, but dynamic)
                //    This can be replaced with a more complex model later
                local_strain_rate = std::max(local_sliding_velocity / 0.0001, 1e3); // 0.1mm shear band

                // 3. Update node mechanics (Stress, Wear, Damage)
                update_node_mechanics(node, dt, local_sliding_velocity, local_strain_rate);
                
                // 4. Update node thermal (Heat Generation)
                //    (Heat *transfer* from chip is handled by SPH model)
                update_node_thermal(node, dt, local_strain_rate, true);

                // 5. Check for failure
                if (check_node_failure(node)) {
                    fractured_this_step++;
                }

            } else {
                // Node is not in contact, just cool it down
                update_node_thermal(node, dt, 0.0, false);
                node.stress *= 0.95; // Relax stress if not in contact
            }
        }

        // Update max values for reporting
        if (node.temperature > max_temp) max_temp = node.temperature;
        if (node.stress > max_stress) max_stress = node.stress;
        total_wear += node.accumulated_wear;
    }
    
    m_total_fractured_nodes += fractured_this_step;

    // Return the metrics for this step
    json metrics;
    metrics["max_temperature_C"] = max_temp;
    metrics["max_stress_MPa"] = max_stress;
    metrics["total_accumulated_wear_m"] = total_wear;
    metrics["cumulative_fractured_nodes"] = m_total_fractured_nodes;
    return metrics;
}

// --- PRIVATE HELPER FUNCTIONS ---
// These are the physics models, now using dynamic, per-node values

void RotationalFEA::update_node_mechanics(Node& node, double dt, double sliding_velocity, double strain_rate) {
    if (!m_material_model || !m_wear_model || !m_failure_model) return;

    // Use the dynamically calculated strain rate
    node.plastic_strain += strain_rate * dt; 

    // Calculate stress using the material model
    node.stress = m_material_model->calculate_stress(node.temperature, node.plastic_strain, strain_rate);
    
    // Calculate wear using the dynamic sliding velocity
    double wear_rate = m_wear_model->calculate_wear_rate(node.temperature, node.stress, sliding_velocity);
    node.accumulated_wear += wear_rate * dt;
    
    // Calculate damage
    double damage_inc = m_failure_model->calculate_damage_increment(node.stress, 1.0, node.accumulated_damage);
    node.accumulated_damage += damage_inc * 0.1; // (Scaling factor, can be tuned)
}

void RotationalFEA::update_node_thermal(Node& node, double dt, double strain_rate, bool is_in_contact_zone) {
    if (!m_thermal_model) return;
    
    // 1. Heat Generation (only if in contact)
    double heat_gen = 0.0;
    if (is_in_contact_zone) {
        heat_gen = m_thermal_model->calculate_heat_generation(node.stress, strain_rate);
    }
    
    // 2. Heat Loss (always happens)
    double heat_loss = m_thermal_model->calculate_heat_dissipation(node.temperature, m_ambient_temperature, 1e-7); // 1e-7 is placeholder surface area
    
    // 3. Apply changes
    //    NOTE: Heat *transfer* from the chip (the SPH particles) will be applied
    //    separately by the SPHWorkpieceModel::interact_with_tool function.
    //    This function only handles internal generation and external cooling.
    node.temperature += (heat_gen - heat_loss) * dt;

    // Clamp temperatures
    double T_melt = m_config_data["material_properties"]["melting_point_C"].get<double>();
    if (node.temperature > T_melt) node.temperature = T_melt;
    if (node.temperature < m_ambient_temperature) node.temperature = m_ambient_temperature;
}

bool RotationalFEA::check_node_failure(Node& node) {
    if (m_failure_model->check_failure(node.stress, node.temperature, node.accumulated_damage)) {
        node.status = NodeStatus::FRACTURED;
        node.stress = 0.0;
        return true;
    }
    return false;
}