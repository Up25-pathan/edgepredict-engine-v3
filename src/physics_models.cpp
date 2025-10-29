#define _USE_MATH_DEFINES
#include "physics_models.h"
#include <algorithm>
#include <cmath>

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
    
    // Strain rate sensitivity (optional, default C=0.014 for most metals)
    C = props.value("C_strain_rate_sensitivity", 0.014);
    ref_strain_rate = config["physics_parameters"]["strain_rate"].get<double>();
}

double JohnsonCookModel::calculate_stress(double temperature_C, double plastic_strain, double strain_rate) const {
    // Clamp temperature
    double T = std::max(T_room, std::min(T_melt, temperature_C));
    
    // Term 1: Strain hardening
    double strain_term = A + B * std::pow(plastic_strain, n);
    
    // Term 2: Strain rate sensitivity
    double strain_rate_ratio = std::max(1.0, strain_rate / ref_strain_rate);
    double rate_term = 1.0 + C * std::log(strain_rate_ratio);
    
    // Term 3: Thermal softening
    double homologous_temp = (T - T_room) / (T_melt - T_room);
    double thermal_term = 1.0 - std::pow(homologous_temp, m);
    
    return strain_term * rate_term * thermal_term;
}

// ═══════════════════════════════════════════════════════════
// USUI WEAR MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

UsuiWearModel::UsuiWearModel(const json& config) {
    const auto& wear = config["material_properties"]["usui_wear_model"];
    A_constant = wear["A_constant"].get<double>();
    B_inv_temp_K = wear["B_inv_temp_K"].get<double>();
}

double UsuiWearModel::calculate_wear_rate(double temperature_C, double stress_MPa, double sliding_velocity) const {
    if (temperature_C <= 0 || stress_MPa <= 0 || sliding_velocity <= 0) {
        return 0.0;
    }
    
    double temp_K = temperature_C + 273.15;
    
    // Usui Model base: W = A × σ × V × exp(-B/T)
    double base_wear = A_constant * stress_MPa * sliding_velocity * std::exp(-B_inv_temp_K / temp_K);

    // Enhanced temperature multiplier with exponential scaling
    double temp_C = temperature_C;
    double max_scale = 5.0; // Wear can increase up to 6x at high temperatures
    // Exponential temperature scaling for more aggressive wear at higher temps
    double scale = std::pow(std::max(0.0, std::min(1.0, temp_C / 800.0)), 1.5);
    double temp_multiplier = 1.0 + max_scale * scale;

    // Stress intensity factor - wear increases more rapidly under high stress
    double stress_threshold = 800.0; // MPa
    double stress_factor = std::pow(stress_MPa / stress_threshold, 1.2);
    
    // Scale factor to make wear more apparent (adjusted to realistic tool life)
    double base_scale = 2e7; // Further increased scale factor for more realistic wear rates

    return base_wear * temp_multiplier * stress_factor * base_scale;
}

// ═══════════════════════════════════════════════════════════
// THERMAL MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

ThermalModel::ThermalModel(const json& config) {
    const auto& props = config["material_properties"];
    density = props["density_kg_m3"].get<double>();
    specific_heat = props["specific_heat_J_kgC"].get<double>();
    heat_transfer_coeff = config["physics_parameters"]["heat_transfer_coefficient"].get<double>();
    thermal_conductivity = props.value("thermal_conductivity_W_mK", 20.0); // Default if not specified
}

double ThermalModel::calculate_heat_generation(double stress, double strain_rate) const {
    // Enhanced heat generation model for machining
    double beta = 0.95;  // Taylor-Quinney coefficient
    double base_heat_flux = beta * stress * strain_rate; // W/m³
    
    // Stress-based enhancement with higher scaling
    double stress_scale = std::pow(stress / 800.0, 1.5); // More aggressive stress scaling
    
    // Temperature-dependent enhancement factor
    double base_enhancement = 12.0; // Significantly increased from 5.0
    double enhancement = base_enhancement * (1.0 + 2.0 * stress_scale);
    
    // Additional heat generation factor to account for high-speed machining effects
    double machining_factor = 2.5;
    
    // Convert to temperature rate with enhanced scaling
    return (base_heat_flux * enhancement * machining_factor) / (density * specific_heat);
}

double ThermalModel::calculate_heat_dissipation(double current_temp, double ambient_temp, double surface_area) const {
    // Enhanced thermal model for machining conditions
    double volume = 1e-9; // m³
    
    // Progressive temperature-dependent cooling reduction
    double delta_T = current_temp - ambient_temp;
    double temp_factor = std::max(0.2, std::exp(-delta_T / 400.0));
    
    // Reduced effective heat transfer coefficient for high-speed machining
    double effective_h = heat_transfer_coeff * temp_factor;
    
    // Base heat loss with temperature differential
    double heat_loss = effective_h * surface_area * delta_T;
    
    // Minimal cooling coefficient for better heat retention
    double cooling_coeff = 0.15; // Further reduced from 0.3
    
    // Temperature-based cooling reduction at higher temperatures
    double temp_scale = std::max(0.1, std::exp(-delta_T / 600.0));
    
    // Convert to cooling rate with enhanced temperature dependence
    return (heat_loss * cooling_coeff * temp_scale) / (density * specific_heat * volume);
}

double ThermalModel::calculate_heat_transfer_from_chip(double chip_temp, double tool_temp, double contact_area) const {
    // Enhanced chip-tool heat transfer model
    double contact_thickness = 5e-7; // Reduced to 0.5 micron for better heat transfer
    
    // Temperature-dependent thermal conductivity enhancement
    double k_factor = 1.0 + 0.5 * std::pow((chip_temp - tool_temp) / 500.0, 0.8);
    double effective_k = thermal_conductivity * k_factor;
    
    // Enhanced heat flux calculation
    double heat_flux = effective_k * contact_area * (chip_temp - tool_temp) / contact_thickness;
    
    // Increased heat transfer effectiveness
    double transfer_enhancement = 2.0;
    
    double volume = 1e-9;
    return (heat_flux * transfer_enhancement) / (density * specific_heat * volume);
}

// ═══════════════════════════════════════════════════════════
// FAILURE CRITERION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

FailureCriterion::FailureCriterion(const json& config) {
    const auto& failure = config["material_properties"]["failure_criterion"];
    ultimate_tensile_strength = failure["ultimate_tensile_strength_MPa"].get<double>();
    fatigue_limit = failure.value("fatigue_limit_MPa", ultimate_tensile_strength * 0.5);
    damage_threshold = 1.0; // Failure occurs at D = 1.0
}

bool FailureCriterion::check_failure(double stress, double temperature, double accumulated_damage) const {
    // Enhanced temperature-dependent material properties
    double temp_K = temperature + 273.15;
    double homologous_temp = temp_K / 2000.0; // Normalize by approx. melting point
    
    // Improved temperature-based softening curve with higher initial threshold
    double temp_factor = exp(-2.0 * pow(homologous_temp, 1.5));
    double effective_uts = ultimate_tensile_strength * (1.2 + 0.3 * temp_factor);
    
    // More gradual stress-based failure probability
    double stress_ratio = stress / effective_uts;
    double base_failure_prob = 1.0 - exp(-pow(stress_ratio, 6));
    
    // Temperature-dependent probability scaling
    double temp_scale = 0.8 + 0.4 * (1.0 - homologous_temp);
    double failure_prob = base_failure_prob * temp_scale;
    
    // Local stress fluctuation factor (±10%)
    double stress_fluctuation = 0.9 + 0.2 * (static_cast<double>(rand()) / RAND_MAX);
    
    // Immediate failure check with enhanced randomization
    if (stress * stress_fluctuation > effective_uts * 1.2 && 
        (static_cast<double>(rand()) / RAND_MAX) < failure_prob * 0.5) {
        return true;
    }
    
    // Progressive damage with enhanced randomization and thermal effects
    double damage_threshold_mod = damage_threshold * 
        (1.1 + 0.3 * (static_cast<double>(rand()) / RAND_MAX)) *
        (1.0 + 0.2 * temp_factor);
    
    // Additional randomization to prevent cascade failures
    double cascade_prevention = (static_cast<double>(rand()) / RAND_MAX) < 0.95 ? 1.0 : 1.5;
    
    return accumulated_damage >= damage_threshold_mod * cascade_prevention;
}

double FailureCriterion::calculate_damage_increment(double stress, double cycles, double current_damage) const {
    // Enhanced Palmgren-Miner rule with stress-dependent threshold
    double stress_ratio = stress / ultimate_tensile_strength;
    double threshold_ratio = 0.4; // Increased from default 0.5 fatigue ratio
    
    if (stress_ratio < threshold_ratio) {
        return 0.0; // No damage below adjusted threshold
    }
    
    // Progressive threshold effect
    double effective_stress = stress * pow((stress_ratio - threshold_ratio) / (1.0 - threshold_ratio), 1.5);
    
    // Modified Basquin's equation with improved scaling and randomization
    double b = 5.0;  // Further reduced exponent for more gradual failure
    double C = 5e21; // Increased for longer tool life
    
    // Add small random variation to cycles-to-failure calculation
    double variation = 0.95 + 0.1 * (static_cast<double>(rand()) / RAND_MAX);
    double N = C * variation * std::pow(effective_stress / ultimate_tensile_strength, -b);
    
    // Non-linear damage increment with progressive scaling
    double base_damage = cycles / N;
    double progressive_factor = pow(current_damage + 0.1, 0.7); // Slower initial damage
    
    // Scale factor reduced for more gradual accumulation
    return base_damage * 0.005 * progressive_factor;
}

// ═══════════════════════════════════════════════════════════
// CHIP FORMATION MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

ChipFormationModel::ChipFormationModel(const json& config) {
    const auto& cfd = config["cfd_parameters"];
    rake_angle_rad = cfd["rake_angle_degrees"].get<double>() * (3.14159265358979323846 / 180.0);
    friction_coefficient = cfd["friction_coefficient"].get<double>();
    
    const auto& physics = config["physics_parameters"];
    uncut_chip_thickness = physics.value("uncut_chip_thickness_mm", 0.1) * 0.001; // Convert to m
    cutting_width = physics.value("cutting_width_mm", 3.0) * 0.001;
}

double ChipFormationModel::calculate_shear_angle(double rake_angle, double friction_angle) const {
    // Merchant's Circle Theory
    // φ = 45° + α/2 - β/2
    // where α = rake angle, β = friction angle = arctan(μ)
    
    return M_PI / 4.0 + rake_angle / 2.0 - friction_angle / 2.0;
}

double ChipFormationModel::calculate_shear_stress(double flow_stress, double shear_angle) const {
    // τ = σ_flow × cos(φ)
    return flow_stress * std::cos(shear_angle);
}

double ChipFormationModel::calculate_chip_velocity(double cutting_speed, double chip_thickness_ratio) const {
    // V_chip = V_cut / r_c
    return cutting_speed / chip_thickness_ratio;
}

double ChipFormationModel::calculate_chip_thickness(double uncut_thickness, double shear_angle, double rake_angle) const {
    // t_chip = t_0 × sin(φ) / cos(φ - α)
    double numerator = uncut_thickness * std::sin(shear_angle);
    double denominator = std::cos(shear_angle - rake_angle);
    return numerator / denominator;
}

double ChipFormationModel::calculate_contact_length(double chip_thickness, double rake_angle) const {
    // Simplified: L_c ≈ t_chip / sin(α)
    return chip_thickness / std::sin(rake_angle);
}

double ChipFormationModel::calculate_cutting_force(double shear_stress, double shear_area) const {
    // F_c = τ × A_shear
    return shear_stress * shear_area;
}

double ChipFormationModel::calculate_thrust_force(double cutting_force, double friction_angle) const {
    // F_t = F_c × tan(friction_angle)
    return cutting_force * std::tan(friction_angle);
}