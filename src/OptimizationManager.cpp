#include "OptimizationManager.h"
#include <iostream>
#include <algorithm>

OptimizationManager::OptimizationManager(const json& config) {
    // Load constraints or defaults
    if (config.contains("optimization_constraints")) {
        auto& c = config["optimization_constraints"];
        target_stress_MPa = c.value("target_stress_MPa", 500.0);
        max_torque_Nm = c.value("max_torque_Nm", 50.0);
        min_feed_rate = c.value("min_feed_rate_mm_min", 50.0);
        max_feed_rate = c.value("max_feed_rate_mm_min", 2000.0);
    } else {
        target_stress_MPa = 500.0;
        max_torque_Nm = 50.0;
        min_feed_rate = 50.0;
        max_feed_rate = 1000.0;
    }

    // --- FIX: Check if machining_parameters exists before accessing ---
    if (config.contains("machining_parameters")) {
        current_feed_rate = config["machining_parameters"].value("feed_rate_mm_min", 300.0);
        current_rpm = config["machining_parameters"].value("rpm", 1000.0);
    } else {
        // Defaults if running purely from G-Code or undefined
        current_feed_rate = 300.0; 
        current_rpm = 1000.0;
    }
    
    integral_error = 0.0;
    prev_error = 0.0;
    
    std::cout << "[Optimizer] Adaptive Control AI Online. Target Stress: " << target_stress_MPa << " MPa" << std::endl;
}

AdaptiveCommand OptimizationManager::monitor_process(double current_stress, double current_torque, double dt) {
    AdaptiveCommand cmd;
    cmd.should_update = false;
    cmd.new_feed_rate = current_feed_rate;
    cmd.new_rpm = current_rpm;

    // 1. Safety Override (Torque Spike)
    if (current_torque > max_torque_Nm) {
        cmd.should_update = true;
        cmd.new_feed_rate = current_feed_rate * 0.5; // Emergency slow down
        cmd.action_reason = "CRITICAL: Torque overload! Reducing feed 50%";
        current_feed_rate = cmd.new_feed_rate;
        return cmd;
    }

    // 2. Stress Optimization (PID Control)
    double error = target_stress_MPa - current_stress;
    
    double Kp = 0.5;   // Proportional gain
    double Ki = 0.01;  // Integral gain
    double Kd = 0.05;  // Derivative gain
    
    integral_error += error * dt;
    double derivative = (error - prev_error) / (dt + 1e-9);
    
    double adjustment = (Kp * error) + (Ki * integral_error) + (Kd * derivative);
    
    // Apply adjustment
    double proposed_feed = current_feed_rate + adjustment;
    
    // Clamp limits
    proposed_feed = std::max(min_feed_rate, std::min(max_feed_rate, proposed_feed));
    
    // Only update if change is significant (> 5%)
    if (std::abs(proposed_feed - current_feed_rate) > (current_feed_rate * 0.05)) {
        cmd.should_update = true;
        cmd.new_feed_rate = proposed_feed;
        if (proposed_feed < current_feed_rate) {
            cmd.action_reason = "Load High - Reducing Feed";
        } else {
            cmd.action_reason = "Load Low - Increasing Feed";
        }
        current_feed_rate = proposed_feed;
    }
    
    prev_error = error;
    return cmd;
}