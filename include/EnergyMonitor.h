#ifndef ENERGY_MONITOR_H
#define ENERGY_MONITOR_H

#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

class EnergyMonitor {
public:
    EnergyMonitor(const json& config);
    ~EnergyMonitor();

    void update(double torque_Nm, double angular_vel_rad_s, double feed_force_N, double feed_vel_m_s, double dt);
    json get_report() const;

private:
    double m_total_energy_joules;
    double m_max_power_watts;
    double m_co2_factor_kg_kWh; // Grid factor
    double m_efficiency_factor; // Machine efficiency (motors < 100%)
};

#endif // ENERGY_MONITOR_H