#ifndef PHYSICS_MODELS_H
#define PHYSICS_MODELS_H

#include "json.hpp"
#include <cmath>
#include <Eigen/Dense>

using json = nlohmann::json;

class JohnsonCookModel {
public:
    JohnsonCookModel(const json& config) {
        auto& props = config["material_properties"];
        A = props.value("A_yield_strength_MPa", 300.0) * 1e6;
        B = props.value("B_strain_hardening_MPa", 500.0) * 1e6;
        n = props.value("n_strain_hardening_exp", 0.3);
        C = props.value("C_strain_rate_sensitivity", 0.01);
        m = props.value("m_thermal_softening_exp", 1.0);
        melt_temp = props.value("melting_point_C", 1400.0);
        ambient_temp = config["physics_parameters"].value("ambient_temperature_C", 25.0);

        // --- NEW: Anisotropy (Grain Direction) ---
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

    double calculate_yield_stress(double strain, double strain_rate, double temp, const Eigen::Vector3d& deformation_dir = Eigen::Vector3d::Zero()) const {
        // 1. Standard Johnson-Cook Terms
        double strain_term = A + B * std::pow(strain, n);
        
        double rate_term = 1.0;
        if (strain_rate > 1.0) rate_term = 1.0 + C * std::log(strain_rate);
        
        double temp_term = 1.0;
        double t_hom = (temp - ambient_temp) / (melt_temp - ambient_temp);
        if (t_hom > 0.0 && t_hom < 1.0) temp_term = 1.0 - std::pow(t_hom, m);
        else if (t_hom >= 1.0) temp_term = 0.0;

        double base_stress = strain_term * rate_term * temp_term;

        // 2. NEW: Anisotropy Scaling
        // If deformation direction is provided and opposes grain, stress increases.
        if (anisotropy_factor > 1.0 && grain_dir.norm() > 0.1 && deformation_dir.norm() > 0.1) {
            // Dot product: 1.0 = Aligned (Weakest), 0.0 = Cross (Strongest)
            double alignment = std::abs(deformation_dir.normalized().dot(grain_dir)); 
            
            // Linear interpolation: Factor ranges from 1.0 (Aligned) to anisotropy_factor (Cross)
            double factor = 1.0 + (anisotropy_factor - 1.0) * (1.0 - alignment);
            base_stress *= factor;
        }

        return base_stress;
    }

private:
    double A, B, n, C, m;
    double melt_temp, ambient_temp;
    
    // Anisotropy
    Eigen::Vector3d grain_dir;
    double anisotropy_factor;
};

class FailureCriterion {
public:
    FailureCriterion(const json& config) {
        auto& fc = config["material_properties"]["failure_criterion"];
        ultimate_stress = fc.value("ultimate_tensile_strength_MPa", 1000.0) * 1e6;
        fatigue_limit = fc.value("fatigue_limit_MPa", 300.0) * 1e6;
    }

    bool check_failure(double stress) const {
        return stress > ultimate_stress;
    }

    double ultimate_stress;
    double fatigue_limit;
};

// Simple Thermal Model helper for heat generation
class ThermalModel {
public:
    ThermalModel(const json& config) {
        // Taylor-Quinney coefficient (fraction of plastic work converted to heat)
        beta = 0.9; 
    }

    double calculate_heat_generation(double stress, double strain_rate) const {
        return beta * stress * strain_rate;
    }

private:
    double beta;
};

#endif // PHYSICS_MODELS_H