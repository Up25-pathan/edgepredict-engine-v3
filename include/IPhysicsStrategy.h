#ifndef I_PHYSICS_STRATEGY_H
#define I_PHYSICS_STRATEGY_H

#include "json.hpp" //
#include "simulation.h"

using json = nlohmann::json;

// This is the abstract "plug".
// Both our old "dumb" engine and new "smart" engine will fit this.
class IPhysicsStrategy {
public:
    virtual ~IPhysicsStrategy() = default;

    /**
     * @brief Loads the geometry (Mesh) and specific config parameters for this strategy.
     */
    virtual void initialize(Mesh& mesh, const json& config) = 0;

    /**
     * @brief Runs a single physics time step.
     */
    virtual json run_time_step(double dt, int step_num) = 0;

    /**
     * @brief Provides the final R&D report data (e.g., tool life).
     */
    virtual json get_final_results() = 0;

    /**
     * @brief Provides the particle data for visualization.
     */
    virtual json get_visualization_data() = 0;
};

#endif // I_PHYSICS_STRATEGY_H