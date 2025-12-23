#include "OptimizationManager.h"
#include <algorithm>

OptimizationManager::OptimizationManager(const json& config) {
    // 1. Load Initial Parameters
    const auto& mach = config["machining_parameters"];
    m_current_rpm = mach.value("rpm", 1000.0);
    m_current_feed = mach.value("feed_rate_mm_min", 100.0);

    // 2. Load Limits (or defaults if missing)
    if (config.contains("adaptive_control")) {
        const auto& ac = config["adaptive_control"];
        m_max_stress_MPa = ac.value("max_stress_limit_MPa", 1000.0);
        m_max_torque_Nm = ac.value("max_torque_limit_Nm", 50.0);
        m_target_load_factor = ac.value("target_load_percentage", 0.85); // Aim for 85% utilization
    } else {
        // Safe Defaults
        m_max_stress_MPa = 800.0;
        m_max_torque_Nm = 20.0;
        m_target_load_factor = 0.8;
    }

    // 3. Set Safety Bounds ( +/- 50% of nominal)
    m_min_rpm = m_current_rpm * 0.5;
    m_max_rpm = m_current_rpm * 1.5;
    m_min_feed = m_current_feed * 0.1; // Can slow down significantly to save tool
    m_max_feed = m_current_feed * 1.2;

    m_cooldown_timer = 0.0;
    
    std::cout << "[Adaptive] Optimizer initialized. Target Load: " << m_target_load_factor * 100 << "%" << std::endl;
}

OptimizationManager::~OptimizationManager() {}

AdaptiveCommand OptimizationManager::monitor_process(double current_stress, double current_torque, double dt) {
    AdaptiveCommand cmd;
    cmd.should_update = false;
    cmd.new_rpm = m_current_rpm;
    cmd.new_feed_rate = m_current_feed;

    // Update Timer
    if (m_cooldown_timer > 0.0) {
        m_cooldown_timer -= dt;
        return cmd; // Cooldown active, do nothing
    }

    // --- LOGIC 1: STRESS OVERLOAD (Feed Rate Control) ---
    // If stress > limit, reduce feed immediately
    if (current_stress > m_max_stress_MPa) {
        double reduction = 0.8; // Drop 20%
        m_current_feed = std::max(m_min_feed, m_current_feed * reduction);
        
        cmd.should_update = true;
        cmd.new_feed_rate = m_current_feed;
        cmd.action_reason = "CRITICAL STRESS (" + std::to_string((int)current_stress) + " MPa) -> Reduced Feed";
        m_cooldown_timer = 0.005; // 5ms cooldown
    }
    // Optimization: If stress is low (< 50%), safely increase feed
    else if (current_stress < m_max_stress_MPa * 0.5 && m_current_feed < m_max_feed) {
        double increase = 1.05; // +5%
        m_current_feed = std::min(m_max_feed, m_current_feed * increase);
        
        cmd.should_update = true;
        cmd.new_feed_rate = m_current_feed;
        cmd.action_reason = "Low Load -> Optimized Feed (+5%)";
        m_cooldown_timer = 0.01;
    }

    // --- LOGIC 2: TORQUE OVERLOAD (RPM Control) ---
    // High torque usually means the tool is biting too hard or clogging.
    // Strategy: Increase RPM slightly to cut smaller chips, or reduce Feed.
    if (current_torque > m_max_torque_Nm) {
        // Complex logic: Check if we are stalling
        // Simple V1: Reduce feed first
        m_current_feed *= 0.9;
        cmd.should_update = true;
        cmd.new_feed_rate = m_current_feed;
        cmd.action_reason = "HIGH TORQUE -> Reduced Feed";
        m_cooldown_timer = 0.005;
    }

    return cmd;
}