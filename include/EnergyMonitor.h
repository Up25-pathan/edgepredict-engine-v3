#ifndef ENERGY_MONITOR_H
#define ENERGY_MONITOR_H

#include "json.hpp"

using json = nlohmann::json;

class EnergyMonitor {
public:
    EnergyMonitor(const json& config);
    
    void update(double torque_Nm, double angular_vel_rad_s, double feed_force_N, double feed_vel_m_s, double dt);
    json get_report() const;

private:
    double total_energy_joules;
    double peak_power_watts;
    double co2_factor; // kg CO2 per kWh
    double machine_efficiency;
};

#endif // ENERGY_MONITOR_H