#ifndef ROTATIONAL_FEA_H
#define ROTATIONAL_FEA_H

#include "simulation.h"
#include "json.hpp"
#include <Eigen/Dense>
#include <vector>

struct SpringEdge {
    int node_a;
    int node_b;
    double rest_length;
    double stiffness;
};

class RotationalFEA {
public:
    RotationalFEA(const json& config, Mesh* mesh);
    ~RotationalFEA();

    void initialize_physics();
    void update_tool_motion(double dt, double current_time);
    json run_physics_step(double dt);
    double get_max_stress() const;

    // Speed limit calculator
    double compute_stable_time_step() const;

    // Adaptive Control API
    void set_rotation_speed(double rpm);
    void set_feed_rate(double mm_min);

private:
    void compute_lumped_mass();
    void build_stiffness_network();
    void compute_internal_forces();
    
    Mesh* m_mesh;
    json m_config;

    std::vector<SpringEdge> m_springs;
    double m_density;
    double m_youngs_modulus;
    double m_damping;
    
    double m_rotation_speed_rad_s;
    double m_feed_speed_m_s;
    
    Eigen::Vector3d m_rotation_axis;
    Eigen::Vector3d m_center_of_mass;

    // --- NEW: Tool Fatigue & Failure ---
    double m_fatigue_limit_MPa;
    double m_accumulated_damage; // 0.0 to 1.0 (100%)
    bool m_tool_broken;
};

#endif // ROTATIONAL_FEA_H