#ifndef SPH_WORKPIECE_MODEL_H
#define SPH_WORKPIECE_MODEL_H

#include <vector>
#include <memory>
#include "json.hpp"
#include <Eigen/Dense>
#include "physics_models.h"
#include "simulation.h" 
#include "PhysicsData.h"
#include "navier_stokes.h" 

#include <sycl/sycl.hpp>

class SpatialGrid;

using json = nlohmann::json;

class SPHWorkpieceModel {
public:
    SPHWorkpieceModel(const json& config);
    ~SPHWorkpieceModel();

    void initialize_workpiece(const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner);
    void update_step(double dt);
    void interact_with_tool(Mesh& tool_mesh, double dt);
    void apply_fluid_forces(const NavierStokesSolver& cfd, double dt);
    json analyze_distortion();

    // Helper to sync GPU data back to CPU for visualization only when needed
    const ParticleData& pull_data_from_gpu(); 
    const ParticleData& get_data() const { return m_data; }

private:
    void calculate_density_and_pressure();
    void calculate_internal_forces();
    void integrate(double dt);
    void init_gpu_buffers();

    json m_config_data;
    ParticleData m_data; // CPU Shadow Copy
    
    // Physics Constants
    double m_smoothing_radius;
    double m_particle_mass;
    double m_gas_stiffness;
    double m_viscosity;
    
    std::unique_ptr<JohnsonCookModel> m_material_model;
    std::unique_ptr<FailureCriterion> m_failure_model;
    std::unique_ptr<SpatialGrid> m_grid;

    sycl::queue m_queue;

    // --- PERSISTENT GPU BUFFERS (SoA) ---
    // We use unique_ptr to delay initialization until after particles are generated
    std::unique_ptr<sycl::buffer<double, 1>> b_pos_x, b_pos_y, b_pos_z;
    std::unique_ptr<sycl::buffer<double, 1>> b_vel_x, b_vel_y, b_vel_z;
    std::unique_ptr<sycl::buffer<double, 1>> b_acc_x, b_acc_y, b_acc_z;
    std::unique_ptr<sycl::buffer<double, 1>> b_mass, b_dens, b_press, b_temp;
    std::unique_ptr<sycl::buffer<int, 1>> b_stat;
    
    // Spatial Grid Buffers
    std::unique_ptr<sycl::buffer<int, 1>> b_grid_head;
    std::unique_ptr<sycl::buffer<int, 1>> b_grid_next;
};

#endif // SPH_WORKPIECE_MODEL_H