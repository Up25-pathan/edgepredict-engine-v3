#define _USE_MATH_DEFINES
#include "physics_models.h"
#include <algorithm>
#include <cmath>
#include <cstdlib> 

// ═══════════════════════════════════════════════════════════
// JOHNSON-COOK MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

JohnsonCookModel::JohnsonCookModel(const json& config) {
    const auto& props = config["material_properties"];
    A = props["A_yield_strength_MPa"].get<double>();
    B = props["B_strain_hardening_MPa"].get<double>();
    n = props["n_strain_hardening_exp"].get<double>();
    m = props["m_thermal_softening_exp"].get<double>();

    T_room = config["physics_parameters"]["ambient_temperature_C"].get<double>();
    T_melt = props["melting_point_C"].get<double>();

    C = props.value("C_strain_rate_sensitivity", 0.014);
    ref_strain_rate = config["physics_parameters"]["strain_rate"].get<double>();
}

double JohnsonCookModel::calculate_stress(double temperature_C, double plastic_strain, double strain_rate) const {
    double T = std::max(T_room, std::min(T_melt, temperature_C));
    double safe_plastic_strain = std::max(plastic_strain, 1e-9);
    double strain_term = A + B * std::pow(safe_plastic_strain, n);

    double safe_ref_strain_rate = std::max(ref_strain_rate, 1e-6);
    double safe_strain_rate = std::max(strain_rate, 1e-6);
    double strain_rate_ratio = safe_strain_rate / safe_ref_strain_rate;
    
    double rate_term = 1.0 + C * std::log(std::max(1.0, strain_rate_ratio));

    double safe_T_melt = std::max(T_melt, T_room + 1.0); 
    double homologous_temp = std::max(0.0, std::min(1.0, (T - T_room) / (safe_T_melt - T_room)));
    double thermal_term = 1.0 - std::pow(homologous_temp, m);

    return std::max(0.0, strain_term * rate_term * thermal_term);
}

// ═══════════════════════════════════════════════════════════
// USUI WEAR MODEL IMPLEMENTATION (ROBUST & SMART)
// ═══════════════════════════════════════════════════════════

UsuiWearModel::UsuiWearModel(const json& config) {
    const auto& wear = config["material_properties"]["usui_wear_model"];
    A_constant = wear["A_constant"].get<double>();
    B_inv_temp_K = wear["B_inv_temp_K"].get<double>();
    calibration_factor = config["material_properties"].value("wear_calibration_factor", 1.0);
    
    // SMART UPDATE: Load yield strength to normalize stress
    ref_yield_strength = config["material_properties"]["A_yield_strength_MPa"].get<double>();
}

double UsuiWearModel::calculate_wear_rate(double temperature_C, double stress_MPa, double sliding_velocity) const {
    if (temperature_C <= -273.15 || stress_MPa <= 0 || sliding_velocity <= 0) {
        return 0.0;
    }

    double temp_K = temperature_C + 273.15;
    if (temp_K <= 0) return 0.0;

    // Base Usui Equation
    double base_wear = A_constant * stress_MPa * sliding_velocity * std::exp(-B_inv_temp_K / temp_K);

    // Temperature scaling
    double temp_scale = std::pow(std::max(0.0, std::min(1.0, temperature_C / 1000.0)), 1.5);
    double temp_multiplier = 1.0 + 5.0 * temp_scale;

    // SMART UPDATE: Use relative stress instead of hardcoded 800.0
    double stress_ratio = stress_MPa / std::max(ref_yield_strength, 1.0);
    double stress_factor = std::pow(stress_ratio, 1.2);

    double final_wear_rate = (base_wear * temp_multiplier * stress_factor) * (calibration_factor * 1e-7);

    return std::max(1e-15, final_wear_rate);
}

// ═══════════════════════════════════════════════════════════
// THERMAL MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

ThermalModel::ThermalModel(const json& config) {
    const auto& props = config["material_properties"];
    density = props["density_kg_m3"].get<double>();
    specific_heat = props["specific_heat_J_kgC"].get<double>();
    heat_transfer_coeff = config["physics_parameters"]["heat_transfer_coefficient"].get<double>();
    thermal_conductivity = props.value("thermal_conductivity_W_mK", 20.0); 
    heat_gen_factor = props.value("heat_generation_factor", 1.8);
    
    // SMART UPDATE: Load yield strength
    ref_yield_strength = props["A_yield_strength_MPa"].get<double>();

    density = std::max(density, 1.0);
    specific_heat = std::max(specific_heat, 1.0);
}

double ThermalModel::calculate_heat_generation(double stress, double strain_rate) const {
    double safe_stress = std::max(stress, 0.0);
    double safe_strain_rate = std::max(strain_rate, 0.0);

    double beta = 0.95;
    double base_heat_flux = beta * safe_stress * safe_strain_rate;

    // SMART UPDATE: Relative stress scaling
    double stress_ratio = safe_stress / std::max(ref_yield_strength, 1.0);
    double stress_scale = std::pow(stress_ratio, 1.5);
    
    double base_enhancement = 30.0;
    double enhancement = base_enhancement * (1.0 + 2.0 * stress_scale);

    double denominator = density * specific_heat;
    if (denominator <= 0) return 0.0;
    
    return (base_heat_flux * enhancement * heat_gen_factor) / denominator;
}

double ThermalModel::calculate_heat_dissipation(double current_temp, double ambient_temp, double surface_area) const {
    double volume = 1e-9; 
    double delta_T = current_temp - ambient_temp;

    if (delta_T <= 0) return 0.0;

    double temp_factor = std::max(0.4, std::exp(-delta_T / 600.0));
    double effective_h = heat_transfer_coeff * temp_factor;
    double heat_loss_watts = effective_h * surface_area * delta_T;

    double cooling_coeff = 0.25; 
    double temp_scale = std::max(0.3, std::exp(-delta_T / 800.0));

    double denominator = density * specific_heat * volume;
    if (denominator <= 0) return 0.0;
    
    return std::max(0.0, (heat_loss_watts * cooling_coeff * temp_scale) / denominator);
}

double ThermalModel::calculate_heat_transfer_from_chip(double chip_temp, double tool_temp, double contact_area) const {
    double contact_thickness = 5e-7;
    double delta_T_chip_tool = chip_temp - tool_temp;
    if (delta_T_chip_tool <= 0) return 0.0;

    double k_factor = 1.0 + 0.5 * std::pow(delta_T_chip_tool / 500.0, 0.8);
    double effective_k = thermal_conductivity * k_factor;
    double heat_flux_watts = (effective_k * contact_area * delta_T_chip_tool) / contact_thickness;
    double transfer_enhancement = 2.0;
    double volume = 1e-9;
    double denominator = density * specific_heat * volume;
    
    if (denominator <= 0) return 0.0;
    return std::max(0.0, (heat_flux_watts * transfer_enhancement) / denominator);
}

// ═══════════════════════════════════════════════════════════
// FAILURE CRITERION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

FailureCriterion::FailureCriterion(const json& config) {
    const auto& props = config["material_properties"];
    const auto& failure = props["failure_criterion"];
    ultimate_tensile_strength = failure["ultimate_tensile_strength_MPa"].get<double>();
    fatigue_limit = failure.value("fatigue_limit_MPa", ultimate_tensile_strength * 0.5);
    damage_threshold = 1.0;
    melting_point = props["melting_point_C"].get<double>();

    ultimate_tensile_strength = std::max(ultimate_tensile_strength, 1.0);
    std::srand(12345); 
}

bool FailureCriterion::check_failure(double stress, double temperature, double accumulated_damage) const {
    if (temperature > melting_point * 0.95) return true; 
    
    double safe_stress = std::max(stress, 0.0);
    double safe_accumulated_damage = std::max(accumulated_damage, 0.0);
    
    double stress_fluctuation = 0.95 + 0.1 * ((int(stress) % 100) / 100.0);

    if (safe_stress * stress_fluctuation > ultimate_tensile_strength * 1.3) {
        return true;
    }

    return safe_accumulated_damage >= damage_threshold;
}

double FailureCriterion::calculate_damage_increment(double stress, double cycles, double current_damage) const {
    double safe_stress = std::max(stress, 0.0);
    double safe_cycles = std::max(cycles, 0.0);
    double safe_current_damage = std::max(current_damage, 0.0);

    double stress_ratio = safe_stress / ultimate_tensile_strength;
    double threshold_ratio = 0.4;

    if (stress_ratio < threshold_ratio) return 0.0;

    double effective_stress = safe_stress * pow(std::max(0.0, (stress_ratio - threshold_ratio) / std::max(1e-6, 1.0 - threshold_ratio)), 1.5);
    double b = 5.0;
    double C = 5e21;
    double variation = 0.95 + 0.1 * ((int(stress) % 10) / 10.0);
    
    double stress_pow_arg = std::max(1e-9, effective_stress / ultimate_tensile_strength);
    double N = std::max(C * variation * std::pow(stress_pow_arg, -b), 1.0);

    double base_damage = safe_cycles / N;
    double progressive_factor = pow(safe_current_damage + 0.1, 0.7);
    
    return std::max(0.0, base_damage * 0.05 * progressive_factor);
}

// ═══════════════════════════════════════════════════════════
// CHIP FORMATION MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

ChipFormationModel::ChipFormationModel(const json& config) {
    const auto& cfd = config["cfd_parameters"];
    rake_angle_rad = cfd["rake_angle_degrees"].get<double>() * (M_PI / 180.0);
    friction_coefficient = std::max(0.0, cfd["friction_coefficient"].get<double>());

    const auto& physics = config["physics_parameters"];
    uncut_chip_thickness = std::max(1e-6, physics.value("uncut_chip_thickness_mm", 0.1) * 0.001);
    cutting_width = std::max(1e-6, physics.value("cutting_width_mm", 3.0) * 0.001);
}

double ChipFormationModel::calculate_shear_angle(double rake_angle, double friction_angle) const {
    return M_PI / 4.0 + rake_angle / 2.0 - friction_angle / 2.0;
}

double ChipFormationModel::calculate_shear_stress(double flow_stress, double shear_angle) const {
    return std::max(0.0, flow_stress) * std::cos(shear_angle);
}

double ChipFormationModel::calculate_chip_velocity(double cutting_speed, double chip_thickness_ratio) const {
    return std::max(0.0, cutting_speed) / std::max(chip_thickness_ratio, 0.1);
}

double ChipFormationModel::calculate_chip_thickness(double uncut_thickness, double shear_angle, double rake_angle) const {
    double numerator = uncut_thickness * std::sin(shear_angle);
    double denominator = std::cos(shear_angle - rake_angle);
    if (std::abs(denominator) < 1e-9) return uncut_thickness * 10.0;
    return std::max(1e-6, numerator / denominator);
}

double ChipFormationModel::calculate_contact_length(double chip_thickness, double rake_angle) const {
    double sin_rake = std::sin(rake_angle);
    if (std::abs(sin_rake) < 1e-9) return chip_thickness * 5.0;
    return std::max(1e-6, chip_thickness / sin_rake);
}

double ChipFormationModel::calculate_cutting_force(double shear_stress, double shear_area) const {
    return std::max(0.0, shear_stress) * std::max(0.0, shear_area);
}

double ChipFormationModel::calculate_thrust_force(double cutting_force, double friction_angle) const {
    return std::max(0.0, cutting_force) * std::tan(friction_angle);
}