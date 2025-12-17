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

private:
    void compute_lumped_mass();
    void build_stiffness_network();
    void compute_internal_forces();
    void integrate_explicit(double dt);

    Mesh* m_mesh;
    json m_config;

    std::vector<SpringEdge> m_springs;
    double m_density;
    double m_youngs_modulus;
    double m_damping;
    
    double m_rotation_speed_rad_s;
    Eigen::Vector3d m_rotation_axis;
    Eigen::Vector3d m_center_of_mass;
};

#endif // ROTATIONAL_FEA_H