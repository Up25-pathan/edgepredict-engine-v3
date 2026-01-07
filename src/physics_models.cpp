#include "physics_models.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// =========================================================
// 1. Johnson-Cook Model Implementation
// =========================================================
JohnsonCookModel::JohnsonCookModel(const json& config) {
    auto& props = config["material_properties"];
    // Load constants with safe defaults
    A = props.value("A_yield_strength_MPa", 300.0) * 1e6;
    B = props.value("B_strain_hardening_MPa", 500.0) * 1e6;
    n = props.value("n_strain_hardening_exp", 0.3);
    C = props.value("C_strain_rate_sensitivity", 0.01);
    m = props.value("m_thermal_softening_exp", 1.0);
    melt_temp = props.value("melting_point_C", 1400.0);
    ambient_temp = config["physics_parameters"].value("ambient_temperature_C", 25.0);

    // Anisotropy (Grain Direction) Logic
    if (props.contains("anisotropy")) {
        auto& aniso = props["anisotropy"];
        grain_dir = Eigen::Vector3d(
            aniso.value("grain_x", 1.0), 
            aniso.value("grain_y", 0.0), 
            aniso.value("grain_z", 0.0)
        ).normalized();
        anisotropy_factor = aniso.value("strength_factor", 1.0); 
    } else {
        grain_dir = Eigen::Vector3d::Zero();
        anisotropy_factor = 1.0;
    }
}

double JohnsonCookModel::calculate_yield_stress(double strain, double strain_rate, double temp, const Eigen::Vector3d& deformation_dir) const {
    // 1. Strain Hardening Term
    double strain_term = A + B * std::pow(std::max(strain, 0.0), n);
    
    // 2. Strain Rate Term
    double rate_term = 1.0;
    // Avoid log(0)
    double eff_rate = std::max(strain_rate, 1e-6); 
    if (eff_rate > 1.0) rate_term = 1.0 + C * std::log(eff_rate);
    
    // 3. Thermal Softening Term
    double temp_term = 1.0;
    double t_hom = (temp - ambient_temp) / (melt_temp - ambient_temp);
    
    if (t_hom > 0.0 && t_hom < 1.0) temp_term = 1.0 - std::pow(t_hom, m);
    else if (t_hom >= 1.0) temp_term = 0.0; // Melted

    double base_stress = strain_term * rate_term * temp_term;

    // 4. Anisotropy Scaling
    // If deformation direction opposes grain, stress increases.
    if (anisotropy_factor > 1.0 && grain_dir.norm() > 0.1 && deformation_dir.norm() > 0.1) {
        // Dot product: 1.0 = Aligned (Weakest), 0.0 = Cross (Strongest)
        double alignment = std::abs(deformation_dir.normalized().dot(grain_dir)); 
        
        // Linear interpolation: Factor ranges from 1.0 (Aligned) to anisotropy_factor (Cross)
        double factor = 1.0 + (anisotropy_factor - 1.0) * (1.0 - alignment);
        base_stress *= factor;
    }

    return base_stress;
}

// =========================================================
// 2. Failure Criterion Implementation
// =========================================================
FailureCriterion::FailureCriterion(const json& config) {
    auto& fc = config["material_properties"]["failure_criterion"];
    ultimate_stress = fc.value("ultimate_tensile_strength_MPa", 1000.0) * 1e6;
    fatigue_limit = fc.value("fatigue_limit_MPa", 300.0) * 1e6;
}

bool FailureCriterion::check_failure(double stress) const {
    return stress > ultimate_stress;
}

// =========================================================
// 3. Thermal Model Implementation
// =========================================================
ThermalModel::ThermalModel(const json& config) {
    // Taylor-Quinney coefficient (fraction of plastic work converted to heat)
    // Typically 0.9 (90% of deformation becomes heat)
    beta = 0.9; 
}

double ThermalModel::calculate_heat_generation(double stress, double strain_rate) const {
    return beta * stress * strain_rate;
}

// =========================================================
// 4. Usui Wear Model Implementation
// =========================================================
UsuiWearModel::UsuiWearModel(const json& config) {
    // Standard Usui coefficients for Carbide vs Steel
    A_w = 1e-9;
    B_w = 1000.0;
    
    if (config.contains("wear_parameters")) {
        A_w = config["wear_parameters"].value("A_usui", 1e-9);
        B_w = config["wear_parameters"].value("B_usui", 1000.0);
    }
}

double UsuiWearModel::calculate_wear_rate(double temp_C, double stress_MPa, double sliding_velocity) const {
    double temp_K = temp_C + 273.15;
    // Usui Equation: dW/dt = A * stress * velocity * exp(-B / T)
    // Prevent division by zero temperature
    if (temp_K < 100.0) temp_K = 100.0;
    
    return A_w * stress_MPa * sliding_velocity * std::exp(-B_w / temp_K);
}