#ifndef IPHYSICS_STRATEGY_H
#define IPHYSICS_STRATEGY_H

#include "json.hpp"
#include "simulation.h" 
#include <opencascade/TopoDS_Shape.hxx>

using json = nlohmann::json;

// Abstract Base Class (Interface)
class IPhysicsStrategy {
public:
    virtual ~IPhysicsStrategy() = default;

    // Initialize the strategy with mesh, CAD geometry, and config
    virtual void initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) = 0;

    // Run a single time step of physics
    virtual json run_time_step(double dt, int step_num) = 0;

    // Retrieve final results
    virtual json get_final_results() = 0;

    // Retrieve visualization data (particles, etc.)
    virtual json get_visualization_data() = 0;
};

#endif // IPHYSICS_STRATEGY_H