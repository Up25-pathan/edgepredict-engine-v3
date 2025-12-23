#ifndef OPTIMIZATION_MANAGER_H
#define OPTIMIZATION_MANAGER_H

#include "json.hpp"
#include <vector>
#include <cmath>
#include <iostream>

using json = nlohmann::json;

struct AdaptiveCommand {
    bool should_update;
    double new_rpm;
    double new_feed_rate;
    std::string action_reason;
};

class OptimizationManager {
public:
    OptimizationManager(const json& config);
    ~OptimizationManager();

    // The Main Loop calls this every few frames
    AdaptiveCommand monitor_process(double current_stress_MPa, double current_torque_Nm, double dt);

private:
    // Limits read from Input
    double m_max_stress_MPa;
    double m_max_torque_Nm;
    double m_target_load_factor; // e.g., keep tool at 80% capacity

    // Current State
    double m_current_rpm;
    double m_current_feed;
    
    // Bounds
    double m_min_rpm, m_max_rpm;
    double m_min_feed, m_max_feed;

    // Stability
    double m_cooldown_timer; // Don't change params too fast
};

#endif // OPTIMIZATION_MANAGER_H