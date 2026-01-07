#ifndef PHYSICS_MODELS_H
#define PHYSICS_MODELS_H

#include "json.hpp"
#include <Eigen/Dense>

using json = nlohmann::json;

// --- 1. Johnson-Cook Strength Model ---
class JohnsonCookModel {
public:
    JohnsonCookModel(const json& config);
    
    // Calculates material strength based on Strain, Rate, Temp, and Grain Direction
    double calculate_yield_stress(double strain, double strain_rate, double temp, const Eigen::Vector3d& deformation_dir = Eigen::Vector3d::Zero()) const;

private:
    double A, B, n, C, m;
    double melt_temp, ambient_temp;
    
    // Anisotropy (Grain Direction)
    Eigen::Vector3d grain_dir;
    double anisotropy_factor;
};

// --- 2. Failure & Fatigue Criterion ---
class FailureCriterion {
public:
    FailureCriterion(const json& config);
    bool check_failure(double stress) const;
    
    double ultimate_stress;
    double fatigue_limit;
};

// --- 3. Thermal Generation Model ---
class ThermalModel {
public:
    ThermalModel(const json& config);
    double calculate_heat_generation(double stress, double strain_rate) const;
private:
    double beta; // Taylor-Quinney coefficient
};

// --- 4. Tool Wear Model (Usui) ---
class UsuiWearModel {
public:
    UsuiWearModel(const json& config);
    double calculate_wear_rate(double temp_C, double stress_MPa, double sliding_velocity) const;
private:
    double A_w, B_w;
};

#endif // PHYSICS_MODELS_H