#include "EnergyMonitor.h"
#include <iostream>
#include <cmath>

EnergyMonitor::EnergyMonitor(const json& config) {
    total_energy_joules = 0.0;
    peak_power_watts = 0.0;
    co2_factor = 0.475; // Global average
    machine_efficiency = 0.85; // Motor efficiency
    
    if (config.contains("sustainability")) {
        co2_factor = config["sustainability"].value("co2_factor_kg_kWh", 0.475);
    }
}

void EnergyMonitor::update(double torque, double omega, double force, double velocity, double dt) {
    // Power = (Torque * AngularVelocity) + (LinearForce * LinearVelocity)
    double p_rotary = std::abs(torque * omega);
    double p_linear = std::abs(force * velocity);
    
    double p_total_mech = p_rotary + p_linear;
    double p_electrical = p_total_mech / machine_efficiency;
    
    if (p_electrical > peak_power_watts) peak_power_watts = p_electrical;
    
    total_energy_joules += p_electrical * dt;
}

json EnergyMonitor::get_report() const {
    double kwh = total_energy_joules / 3.6e6;
    double co2_kg = kwh * co2_factor;
    
    return {
        {"total_energy_kWh", kwh},
        {"peak_power_kW", peak_power_watts / 1000.0},
        {"co2_footprint_kg", co2_kg},
        {"cost_estimate_usd", kwh * 0.12}
    };
}