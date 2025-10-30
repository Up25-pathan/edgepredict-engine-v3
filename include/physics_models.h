#ifndef PHYSICS_MODELS_H
#define PHYSICS_MODELS_H

#include "json.hpp"
#include <cmath>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════
// MATERIAL CONSTITUTIVE MODELS
// ═══════════════════════════════════════════════════════════

class JohnsonCookModel {
public:
    JohnsonCookModel(const json& config);
    double calculate_stress(double temperature_C, double plastic_strain, double strain_rate) const;
    
private:
    double A, B, n, C, m;
    double T_room, T_melt;
    double ref_strain_rate;
};

// ═══════════════════════════════════════════════════════════
// WEAR MODELS
// ═══════════════════════════════════════════════════════════

class UsuiWearModel {
public:
    UsuiWearModel(const json& config);
    double calculate_wear_rate(double temperature_C, double stress_MPa, double sliding_velocity) const;
    
private:
    double A_constant;
    double B_inv_temp_K;
};

// ═══════════════════════════════════════════════════════════
// THERMAL MODELS
// ═══════════════════════════════════════════════════════════

class ThermalModel {
public:
    ThermalModel(const json& config);
    
    double calculate_heat_generation(double stress, double strain_rate) const;
    double calculate_heat_dissipation(double current_temp, double ambient_temp, double surface_area) const;
    double calculate_heat_transfer_from_chip(double chip_temp, double tool_temp, double contact_area) const;
    
private:
    double density;
    double specific_heat;
    double heat_transfer_coeff;
    double thermal_conductivity;
};

// ═══════════════════════════════════════════════════════════
// FAILURE CRITERIA
// ═══════════════════════════════════════════════════════════

class FailureCriterion {
public:
    FailureCriterion(const json& config);
    
    bool check_failure(double stress, double temperature, double accumulated_damage) const;
    double calculate_damage_increment(double stress, double cycles, double current_damage) const;
    
private:
    double ultimate_tensile_strength;
    double fatigue_limit;
    double damage_threshold;
};

// ═══════════════════════════════════════════════════════════
// CHIP FORMATION MECHANICS
// ═══════════════════════════════════════════════════════════

class ChipFormationModel {
public:
    ChipFormationModel(const json& config);
    
    // Merchant's Circle Theory
    double calculate_shear_angle(double rake_angle, double friction_angle) const;
    double calculate_shear_stress(double flow_stress, double shear_angle) const;
    double calculate_chip_velocity(double cutting_speed, double chip_thickness_ratio) const;
    
    // Chip geometry
    double calculate_chip_thickness(double uncut_thickness, double shear_angle, double rake_angle) const;
    double calculate_contact_length(double chip_thickness, double rake_angle) const;
    
    // Forces
    double calculate_cutting_force(double shear_stress, double shear_area) const;
    double calculate_thrust_force(double cutting_force, double friction_angle) const;
    
private:
    double rake_angle_rad;
    double friction_coefficient;
    double uncut_chip_thickness;
    double cutting_width;
};

#endif // PHYSICS_MODELS_H
