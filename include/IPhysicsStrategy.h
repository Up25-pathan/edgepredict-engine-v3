#ifndef I_PHYSICS_STRATEGY_H
#define I_PHYSICS_STRATEGY_H

#include "json.hpp"
#include <opencascade/TopoDS_Shape.hxx>
#include <Eigen/Dense>

using json = nlohmann::json;
struct Mesh;

class IPhysicsStrategy {
public:
    virtual ~IPhysicsStrategy() = default;

    virtual void initialize(Mesh& mesh, const TopoDS_Shape& cad_shape, const json& config) = 0;
    virtual json run_time_step(double dt, int step_num) = 0;
    virtual json get_final_results() = 0;
    virtual json get_visualization_data() = 0;
    
    // NEW: G-Code Support Interface
    virtual void update_machine_state(const Eigen::Vector3d& pos, double feed, double rpm) = 0;
    
    // Helper setters
    virtual void set_feed_rate(double f) = 0;
    virtual void set_rotation_speed(double s) = 0;
};

#endif