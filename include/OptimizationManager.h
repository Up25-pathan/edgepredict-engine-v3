#ifndef OPTIMIZATION_MANAGER_H
#define OPTIMIZATION_MANAGER_H

#include "json.hpp"
#include <string>

using json = nlohmann::json;

struct AdaptiveCommand {
    bool should_update;
    double new_feed_rate; // mm/min
    double new_rpm;
    std::string action_reason;
};

class OptimizationManager {
public:
    OptimizationManager(const json& config);
    
    // Called every N steps to check tool health
    AdaptiveCommand monitor_process(double current_stress_MPa, double current_torque_Nm, double dt);

private:
    double target_stress_MPa;
    double max_torque_Nm;
    double min_feed_rate;
    double max_feed_rate;
    double current_feed_rate;
    double current_rpm;
    
    // PID Control State
    double integral_error;
    double prev_error;
};

#endif // OPTIMIZATION_MANAGER_H