#pragma once
/**
 * @file SurfaceRoughnessPredictor.h
 * @brief Surface roughness prediction from simulation data
 * 
 * Predicts:
 * - Ra (arithmetic average roughness)
 * - Rz (average max height)
 * - Rmax (maximum height)
 * - Rq (RMS roughness)
 * 
 * Uses:
 * - Kinematic/theoretical roughness from tool geometry
 * - Dynamic roughness from tool vibration
 * - Plastic deformation effects
 * - Built-up edge (BUE) effects
 */

#include "Types.h"
#include "Config.h"
#include "IMachiningStrategy.h"
#include <vector>
#include <functional>

namespace edgepredict {

/**
 * @brief Surface roughness parameters (ISO 4287)
 */
struct SurfaceRoughnessParams {
    double Ra = 0;      // Arithmetic average roughness (μm)
    double Rz = 0;      // Average max height of profile (μm)
    double Rmax = 0;    // Maximum height of profile (μm)
    double Rq = 0;      // RMS roughness (μm)
    double Rsk = 0;     // Skewness
    double Rku = 0;     // Kurtosis
    double Rsm = 0;     // Mean spacing of peaks (μm)
    
    // Components
    double Ra_kinematic = 0;    // From tool geometry (ideal)
    double Ra_dynamic = 0;      // From tool vibration
    double Ra_plastic = 0;      // From plastic flow/BUE
    
    // Quality assessment
    bool meetsTolerance = false;
    double targetRa = 0;
};

/**
 * @brief Surface profile point
 */
struct ProfilePoint {
    double position;    // Distance along profile (μm)
    double height;      // Height deviation from mean (μm)
};

/**
 * @brief Surface roughness predictor
 * 
 * Combines kinematic, dynamic, and material effects
 */
class SurfaceRoughnessPredictor {
public:
    SurfaceRoughnessPredictor();
    ~SurfaceRoughnessPredictor() = default;
    
    /**
     * @brief Initialize from configuration
     */
    void initialize(const Config& config);
    
    /**
     * @brief Set machining strategy for kinematic calculation
     */
    void setStrategy(IMachiningStrategy* strategy) { m_strategy = strategy; }
    
    /**
     * @brief Predict roughness from current machining output
     */
    SurfaceRoughnessParams predict(const MachiningOutput& output);
    
    /**
     * @brief Predict roughness from basic parameters
     * @param feedPerRev Feed per revolution (mm)
     * @param noseRadius Tool nose radius (mm)
     * @param cuttingSpeed Cutting speed (m/min)
     */
    SurfaceRoughnessParams predictFromParams(double feedPerRev, 
                                              double noseRadius,
                                              double cuttingSpeed);
    
    /**
     * @brief Update with dynamic effects from simulation
     * @param toolVibrationAmplitude Peak-to-peak vibration (μm)
     * @param vibrationFrequency Hz
     */
    void updateDynamicEffects(double toolVibrationAmplitude, 
                               double vibrationFrequency);
    
    /**
     * @brief Generate surface profile from SPH particles
     * @param particles Particles near machined surface
     * @param surfaceNormal Normal direction of surface
     * @param profileLength Length to sample (mm)
     * @return Profile points
     */
    std::vector<ProfilePoint> generateProfileFromParticles(
        const std::vector<SPHParticle>& particles,
        const Vec3& surfaceNormal,
        double profileLength);
    
    /**
     * @brief Calculate roughness from profile
     */
    SurfaceRoughnessParams calculateFromProfile(const std::vector<ProfilePoint>& profile);
    
    /**
     * @brief Set target roughness for quality check
     */
    void setTargetRa(double targetRa) { m_targetRa = targetRa; }
    
    /**
     * @brief Get last prediction
     */
    const SurfaceRoughnessParams& getLastPrediction() const { return m_lastPrediction; }

private:
    IMachiningStrategy* m_strategy = nullptr;
    
    // Configuration
    double m_targetRa = 1.6;        // Target Ra (μm)
    double m_materialSpringback = 0.1;  // Material elastic recovery factor
    double m_bueThreshold = 0.3;    // Temperature at which BUE forms (fraction of melting)
    
    // Dynamic state
    double m_vibrationAmplitude = 0;
    double m_vibrationFrequency = 0;
    bool m_hasBUE = false;
    
    // Last prediction
    SurfaceRoughnessParams m_lastPrediction;
    
    // Helper methods
    double calculateKinematicRoughness(double feed, double noseRadius) const;
    double calculateDynamicRoughness() const;
    double calculatePlasticRoughness(double temperature, double stress) const;
    double estimateBUEEffect(double cuttingSpeed, double temperature) const;
};

} // namespace edgepredict
