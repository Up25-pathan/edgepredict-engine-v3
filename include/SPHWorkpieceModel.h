#ifndef SPH_WORKPIECE_MODEL_H
#define SPH_WORKPIECE_MODEL_H

#include <vector>
#include <memory>
#include "json.hpp"
#include <Eigen/Dense>
#include "physics_models.h"
#include "simulation.h" 
#include "PhysicsData.h"

// --- GPU HEADER ---
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
    
    const ParticleData& get_data() const { return m_data; }

private:
    void calculate_density_and_pressure();
    void calculate_internal_forces();
    void integrate(double dt);

    json m_config_data;
    ParticleData m_data;
    
    double m_smoothing_radius;
    double m_particle_mass;
    double m_gas_stiffness;
    double m_viscosity;
    
    std::unique_ptr<JohnsonCookModel> m_material_model;
    std::unique_ptr<FailureCriterion> m_failure_model;
    std::unique_ptr<SpatialGrid> m_grid;

    Eigen::AlignedBox3d m_active_bounds;
    bool m_bounds_initialized = false;

    // --- GPU Engine ---
    sycl::queue m_queue; // The command queue for the GPU
};

#endif // SPH_WORKPIECE_MODEL_H