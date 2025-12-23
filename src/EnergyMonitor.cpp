#include "EnergyMonitor.h"

EnergyMonitor::EnergyMonitor(const json& config) {
    m_total_energy_joules = 0.0;
    m_max_power_watts = 0.0;
    
    // Default: Global average carbon intensity ~0.475 kg/kWh
    m_co2_factor_kg_kWh = config.value("sustainability", json::object()).value("co2_factor_kg_kWh", 0.475);
    m_efficiency_factor = 0.85; // Typical spindle motor efficiency
    
    std::cout << "[Sustainability] Energy Monitor Active. Grid Factor: " << m_co2_factor_kg_kWh << " kgCO2/kWh" << std::endl;
}

EnergyMonitor::~EnergyMonitor() {}

void EnergyMonitor::update(double torque_Nm, double angular_vel_rad_s, double feed_force_N, double feed_vel_m_s, double dt) {
    // 1. Spindle Power = Torque * Omega
    double p_spindle = std::abs(torque_Nm * angular_vel_rad_s);
    
    // 2. Feed Power = Force * Velocity
    double p_feed = std::abs(feed_force_N * feed_vel_m_s);
    
    // 3. Total Electrical Power (Input Power)
    double p_total = (p_spindle + p_feed) / m_efficiency_factor;
    
    if (p_total > m_max_power_watts) m_max_power_watts = p_total;
    
    m_total_energy_joules += p_total * dt;
}

json EnergyMonitor::get_report() const {
    double energy_kWh = m_total_energy_joules / 3600000.0;
    double co2_kg = energy_kWh * m_co2_factor_kg_kWh;
    
    return {
        {"total_energy_kWh", energy_kWh},
        {"peak_power_kW", m_max_power_watts / 1000.0},
        {"co2_footprint_kg", co2_kg},
        {"cost_estimate_usd", energy_kWh * 0.15} // Avg industrial rate
    };
}