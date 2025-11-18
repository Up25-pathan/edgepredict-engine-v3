#ifndef SPH_WORKPIECE_MODEL_H
#define SPH_WORKPIECE_MODEL_H

#include <vector>
#include <memory>
#include "json.hpp" //
#include <Eigen/Dense>
#include "physics_models.h" // We need the material models
#include "simulation.h"     // For the Mesh and Node structs

using json = nlohmann::json;

// --- 1. The New SPH Particle Struct ---
// This replaces the old 'Particle' struct. It is now a full physics entity.

enum class ParticleStatus {
    WORKPIECE_SOLID, // Part of the main block
    CHIP_FLOWING,    // Has been "cut" and is now a chip
    INACTIVE         // Flown out of bounds
};

struct SPHParticle {
    // Kinematic Properties
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d acceleration;

    // Physical Properties
    double mass;
    double density;
    double pressure;
    double temperature;

    // Material State (from your existing physics models)
    double strain;
    double plastic_strain;
    double stress;
    double damage; // Tracks accumulated damage before fracture

    ParticleStatus status = ParticleStatus::WORKPIECE_SOLID;
};

// --- 2. The SPH Solver Class ---
// This class manages all particles and their interactions.
// It REPLACES NavierStokesSolver and ParticleSystem.

class SPHWorkpieceModel {
public:
    SPHWorkpieceModel(const json& config);
    ~SPHWorkpieceModel();

    // --- Core Functions ---

    /**
     * @brief Creates the initial block of workpiece material as SPH particles.
     */
    void initialize_workpiece(const Eigen::Vector3d& workpiece_min, 
                              const Eigen::Vector3d& workpiece_max);

    /**
     * @brief Main physics update loop. Runs one time step (dt) of the SPH simulation.
     * This calculates density, pressure, and internal forces for all particles.
     */
    void update_step(double dt);

    /**
     * @brief CRITICAL R&D FUNCTION: Calculates interaction between tool and workpiece.
     * This is the "cutting" part. It calculates forces, damage, and heat transfer
     * between the tool's 'Nodes' and the 'SPHParticles'.
     */
    void interact_with_tool(Mesh& tool_mesh, double dt);

    /**
     * @brief Returns all active particles for rendering in the UI.
     */
    std::vector<const SPHParticle*> get_particles_for_visualization() const;


private:
    // --- SPH Helper Functions ---
    // These are the standard steps of an SPH solver
    void calculate_density_and_pressure();
    void calculate_internal_forces(); // Viscosity, etc.
    void integrate(double dt); // Move particles

    // --- Member Variables ---
    std::vector<SPHParticle> m_particles;
    
    // Physics models (we own these now, as they apply to the workpiece)
    std::unique_ptr<JohnsonCookModel> m_material_model;
    std::unique_ptr<FailureCriterion> m_failure_model;

    // Simulation parameters
    double m_particle_mass;
    double m_smoothing_radius; // 'h' in SPH
    double m_gas_stiffness;    // For pressure calculation
    double m_viscosity;
    
    json m_config_data;
};

#endif // SPH_WORKPIECE_MODEL_H