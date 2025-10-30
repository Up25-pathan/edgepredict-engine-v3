#define _USE_MATH_DEFINES
#include "physics_models.h"
#include <algorithm>
#include <cmath>
#include <cstdlib> // For rand()

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
    // Ensure plastic_strain is positive to avoid issues with pow
    double safe_plastic_strain = std::max(plastic_strain, 1e-9);
    double strain_term = A + B * std::pow(safe_plastic_strain, n);

    // Term 2: Strain rate sensitivity
    // ROBUSTNESS FIX: Ensure ref_strain_rate is non-zero to prevent division by zero (NaNs/Infs)
    double safe_ref_strain_rate = std::max(ref_strain_rate, 1e-6);
    // Ensure strain_rate is positive
    double safe_strain_rate = std::max(strain_rate, 1e-6);
    double strain_rate_ratio = safe_strain_rate / safe_ref_strain_rate;
    // Use std::max(1.0, ...) to prevent log of values < 1 which could yield negative results if C is positive
    double rate_term = 1.0 + C * std::log(std::max(1.0, strain_rate_ratio));

    // Term 3: Thermal softening
    // Ensure T_melt is strictly greater than T_room
    double safe_T_melt = std::max(T_melt, T_room + 1.0); // Add 1 degree buffer
    double homologous_temp = std::max(0.0, std::min(1.0, (T - T_room) / (safe_T_melt - T_room))); // Clamp between 0 and 1
    double thermal_term = 1.0 - std::pow(homologous_temp, m);

    // Ensure stress is non-negative
    return std::max(0.0, strain_term * rate_term * thermal_term);
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
    // Ensure inputs are physically plausible
    if (temperature_C <= -273.15 || stress_MPa <= 0 || sliding_velocity <= 0) {
        return 0.0;
    }

    double temp_K = temperature_C + 273.15;
    // Ensure temp_K is positive for the exponential calculation
    if (temp_K <= 0) {
        return 0.0;
    }

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
    // Ensure stress_threshold is positive
    double safe_stress_threshold = std::max(stress_threshold, 1.0);
    double stress_factor = std::pow(stress_MPa / safe_stress_threshold, 1.2);

    // Scale factor to make wear more apparent (adjusted to realistic tool life)
    double base_scale = 2e7; // Further increased scale factor for more realistic wear rates

    // Ensure final wear rate is non-negative
    return std::max(0.0, base_wear * temp_multiplier * stress_factor * base_scale);
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

    // Ensure physical properties are positive
    density = std::max(density, 1.0);
    specific_heat = std::max(specific_heat, 1.0);
    heat_transfer_coeff = std::max(heat_transfer_coeff, 0.0);
    thermal_conductivity = std::max(thermal_conductivity, 0.01);
}

double ThermalModel::calculate_heat_generation(double stress, double strain_rate) const {
    // Ensure inputs are non-negative
    double safe_stress = std::max(stress, 0.0);
    double safe_strain_rate = std::max(strain_rate, 0.0);

    // Enhanced heat generation model for machining
    double beta = 0.95;  // Taylor-Quinney coefficient
    double base_heat_flux = beta * safe_stress * safe_strain_rate; // W/m³

    // Stress-based enhancement with higher scaling
    // Use a safe threshold to prevent division issues if stress is very low
    double stress_scale = std::pow(safe_stress / std::max(800.0, 1.0), 1.5);

    // Temperature-dependent enhancement factor
    double base_enhancement = 12.0; // Significantly increased from 5.0
    double enhancement = base_enhancement * (1.0 + 2.0 * stress_scale);

    // Additional heat generation factor to account for high-speed machining effects
    double machining_factor = 2.5;

    // Convert to temperature rate with enhanced scaling (°C/s)
    // Ensure denominator is positive
    double denominator = density * specific_heat;
    if (denominator <= 0) return 0.0; // Avoid division by zero/negative
    return (base_heat_flux * enhancement * machining_factor) / denominator;
}

// --- THIS IS THE CORRECTED HEAT DISSIPATION FUNCTION ---
double ThermalModel::calculate_heat_dissipation(double current_temp, double ambient_temp, double surface_area) const {
    // Enhanced thermal model for machining conditions
    double volume = 1e-9; // m³ - Assumed micro-volume for nodal analysis

    // Temperature difference drives heat loss
    double delta_T = current_temp - ambient_temp;

    // --- If temperature is already at or below ambient, NO heat loss occurs ---
    if (delta_T <= 0) {
        return 0.0; // Prevent further cooling if already at ambient
    }
    // --- End Check ---

    // Progressive temperature-dependent cooling reduction (Original: exp(-delta_T / 400.0))
    // Make this less sensitive at lower delta_T
    double temp_factor = std::max(0.4, std::exp(-delta_T / 600.0)); // Less reduction initially

    // Effective heat transfer coefficient
    double effective_h = heat_transfer_coeff * temp_factor;

    // Base heat loss (W)
    double heat_loss_watts = effective_h * surface_area * delta_T;

    // --- MODIFICATION: Make cooling less aggressive ---
    // Minimal cooling coefficient (Original: 0.15)
    double cooling_coeff = 0.5; // Increased: Allows more cooling, but balanced by temp_scale
    // Temperature-based cooling reduction at higher temperatures (Original: exp(-delta_T / 600.0))
    // Make this reduction start later and be less steep
    double temp_scale = std::max(0.3, std::exp(-delta_T / 800.0)); // Less reduction at high temps
    // --- END MODIFICATION ---

    // Convert Watts to temperature rate (°C/s)
    // Ensure denominator is positive
    double denominator = density * specific_heat * volume;
    if (denominator <= 0) return 0.0; // Avoid division by zero/negative
    double cooling_rate_degC_per_s = (heat_loss_watts * cooling_coeff * temp_scale) / denominator;

    // Ensure cooling rate is non-negative
    return std::max(0.0, cooling_rate_degC_per_s);
}
// --- END CORRECTED FUNCTION ---


double ThermalModel::calculate_heat_transfer_from_chip(double chip_temp, double tool_temp, double contact_area) const {
    // Enhanced chip-tool heat transfer model
    double contact_thickness = 5e-7; // Reduced to 0.5 micron for better heat transfer

    double delta_T_chip_tool = chip_temp - tool_temp;
    // If chip is not hotter, no heat transfer *to* the tool
    if (delta_T_chip_tool <= 0) {
        return 0.0;
    }

    // Temperature-dependent thermal conductivity enhancement
    double k_factor = 1.0 + 0.5 * std::pow(delta_T_chip_tool / 500.0, 0.8);
    double effective_k = thermal_conductivity * k_factor;

    // Enhanced heat flux calculation (W)
    double heat_flux_watts = (effective_k * contact_area * delta_T_chip_tool) / contact_thickness;

    // Increased heat transfer effectiveness
    double transfer_enhancement = 2.0;

    double volume = 1e-9; // Assumed micro-volume for nodal analysis
    // Ensure denominator is positive
    double denominator = density * specific_heat * volume;
    if (denominator <= 0) return 0.0; // Avoid division by zero/negative

    // Convert heat flux (W) to temperature rate (°C/s)
    double heat_transfer_rate_degC_per_s = (heat_flux_watts * transfer_enhancement) / denominator;

    // Ensure rate is non-negative
    return std::max(0.0, heat_transfer_rate_degC_per_s);
}


// ═══════════════════════════════════════════════════════════
// FAILURE CRITERION IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

FailureCriterion::FailureCriterion(const json& config) {
    const auto& failure = config["material_properties"]["failure_criterion"];
    ultimate_tensile_strength = failure["ultimate_tensile_strength_MPa"].get<double>();
    fatigue_limit = failure.value("fatigue_limit_MPa", ultimate_tensile_strength * 0.5);
    damage_threshold = 1.0; // Failure occurs at D = 1.0

    // Ensure UTS is positive
    ultimate_tensile_strength = std::max(ultimate_tensile_strength, 1.0);
    fatigue_limit = std::max(fatigue_limit, 0.0);
}

bool FailureCriterion::check_failure(double stress, double temperature, double accumulated_damage) const {
    // Ensure stress and damage are non-negative
    double safe_stress = std::max(stress, 0.0);
    double safe_accumulated_damage = std::max(accumulated_damage, 0.0);

    // Enhanced temperature-dependent material properties
    double temp_K = temperature + 273.15;
    // Use a safe melting point estimate if needed, ensure positive
    double safe_melting_K = 2000.0; // Default fallback melting point in K
    double homologous_temp = std::max(0.0, std::min(1.0, temp_K / safe_melting_K)); // Clamp 0-1

    // Improved temperature-based softening curve with higher initial threshold
    double temp_factor = exp(-2.0 * pow(homologous_temp, 1.5));
    // Ensure effective UTS is positive
    double effective_uts = std::max(1.0, ultimate_tensile_strength * (1.2 + 0.3 * temp_factor));

    // More gradual stress-based failure probability
    double stress_ratio = safe_stress / effective_uts; // Will be >= 0
    double base_failure_prob = 1.0 - exp(-pow(stress_ratio, 6)); // Between 0 and 1

    // Temperature-dependent probability scaling
    double temp_scale = 0.8 + 0.4 * (1.0 - homologous_temp); // Between 0.8 and 1.2
    double failure_prob = std::max(0.0, std::min(1.0, base_failure_prob * temp_scale)); // Clamp 0-1

    // Local stress fluctuation factor (±10%)
    double stress_fluctuation = 0.9 + 0.2 * (static_cast<double>(rand()) / RAND_MAX);

    // Immediate failure check with enhanced randomization
    if (safe_stress * stress_fluctuation > effective_uts * 1.2 &&
        (static_cast<double>(rand()) / RAND_MAX) < failure_prob * 0.5) {
        return true;
    }

    // Progressive damage with enhanced randomization and thermal effects
    double damage_threshold_mod = damage_threshold * (1.1 + 0.3 * (static_cast<double>(rand()) / RAND_MAX)) *
        (1.0 + 0.2 * temp_factor);

    // Additional randomization to prevent cascade failures
    double cascade_prevention = (static_cast<double>(rand()) / RAND_MAX) < 0.95 ? 1.0 : 1.5;

    // Ensure threshold modification is positive
    damage_threshold_mod = std::max(damage_threshold_mod, 1e-6);

    return safe_accumulated_damage >= damage_threshold_mod * cascade_prevention;
}


double FailureCriterion::calculate_damage_increment(double stress, double cycles, double current_damage) const {
    // Ensure inputs are non-negative
    double safe_stress = std::max(stress, 0.0);
    double safe_cycles = std::max(cycles, 0.0);
    double safe_current_damage = std::max(current_damage, 0.0);

    // Enhanced Palmgren-Miner rule with stress-dependent threshold
    double stress_ratio = safe_stress / ultimate_tensile_strength; // UTS guaranteed positive in constructor
    double threshold_ratio = 0.4; // Increased from default 0.5 fatigue ratio

    if (stress_ratio < threshold_ratio) {
        return 0.0; // No damage below adjusted threshold
    }

    // Progressive threshold effect - ensure denominator is not zero
    double effective_stress = safe_stress * pow(std::max(0.0, (stress_ratio - threshold_ratio) / std::max(1e-6, 1.0 - threshold_ratio)), 1.5);

    // Modified Basquin's equation with improved scaling and randomization
    double b = 5.0;  // Further reduced exponent for more gradual failure
    double C = 5e21; // Increased for longer tool life

    // Add small random variation to cycles-to-failure calculation
    double variation = 0.95 + 0.1 * (static_cast<double>(rand()) / RAND_MAX);
    // Ensure effective_stress ratio is positive for pow calculation
    double stress_pow_arg = std::max(1e-9, effective_stress / ultimate_tensile_strength);
    double N = C * variation * std::pow(stress_pow_arg, -b);
    // Ensure N (cycles to failure) is positive and reasonably large to prevent division by zero
    N = std::max(N, 1.0);

    // Non-linear damage increment with progressive scaling
    double base_damage = safe_cycles / N;
    double progressive_factor = pow(safe_current_damage + 0.1, 0.7); // Slower initial damage

    // Scale factor reduced for more gradual accumulation
    double damage_increment = base_damage * 0.005 * progressive_factor;

    // Ensure damage increment is non-negative
    return std::max(0.0, damage_increment);
}

// ═══════════════════════════════════════════════════════════
// CHIP FORMATION MODEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════

ChipFormationModel::ChipFormationModel(const json& config) {
    const auto& cfd = config["cfd_parameters"];
    rake_angle_rad = cfd["rake_angle_degrees"].get<double>() * (M_PI / 180.0);
    friction_coefficient = cfd["friction_coefficient"].get<double>();
    // Ensure friction coefficient is non-negative for atan
    friction_coefficient = std::max(0.0, friction_coefficient);

    const auto& physics = config["physics_parameters"];
    uncut_chip_thickness = physics.value("uncut_chip_thickness_mm", 0.1) * 0.001; // Convert to m
    cutting_width = physics.value("cutting_width_mm", 3.0) * 0.001;
    // Ensure dimensions are positive
    uncut_chip_thickness = std::max(1e-6, uncut_chip_thickness);
    cutting_width = std::max(1e-6, cutting_width);
}

double ChipFormationModel::calculate_shear_angle(double rake_angle, double friction_angle) const {
    // Merchant's Circle Theory
    // φ = 45° + α/2 - β/2
    return M_PI / 4.0 + rake_angle / 2.0 - friction_angle / 2.0;
}

double ChipFormationModel::calculate_shear_stress(double flow_stress, double shear_angle) const {
    // τ = σ_flow × cos(φ)
    // Ensure flow_stress is non-negative
    return std::max(0.0, flow_stress) * std::cos(shear_angle);
}

double ChipFormationModel::calculate_chip_velocity(double cutting_speed, double chip_thickness_ratio) const {
    // V_chip = V_cut / r_c
    // Ensure ratio is positive to avoid division issues
    double safe_ratio = std::max(chip_thickness_ratio, 0.1); // Assume min ratio of 0.1
    return std::max(0.0, cutting_speed) / safe_ratio;
}

double ChipFormationModel::calculate_chip_thickness(double uncut_thickness, double shear_angle, double rake_angle) const {
    // t_chip = t_0 × sin(φ) / cos(φ - α)
    double numerator = uncut_thickness * std::sin(shear_angle);
    double denominator = std::cos(shear_angle - rake_angle);
    // Avoid division by zero if angle difference is 90 degrees
    if (std::abs(denominator) < 1e-9) {
        return uncut_thickness * 10.0; // Return a large but finite value
    }
    return std::max(1e-6, numerator / denominator); // Ensure positive thickness
}

double ChipFormationModel::calculate_contact_length(double chip_thickness, double rake_angle) const {
    // Simplified: L_c ≈ t_chip / sin(α)
    // Avoid division by zero for zero rake angle
    double sin_rake = std::sin(rake_angle);
    if (std::abs(sin_rake) < 1e-9) {
        return chip_thickness * 5.0; // Estimate based on thickness
    }
    return std::max(1e-6, chip_thickness / sin_rake); // Ensure positive length
}


double ChipFormationModel::calculate_cutting_force(double shear_stress, double shear_area) const {
    // F_c = τ × A_shear
    // Ensure inputs are non-negative
    return std::max(0.0, shear_stress) * std::max(0.0, shear_area);
}

double ChipFormationModel::calculate_thrust_force(double cutting_force, double friction_angle) const {
    // F_t = F_c × tan(friction_angle)
    // Ensure cutting force is non-negative
    return std::max(0.0, cutting_force) * std::tan(friction_angle);
}