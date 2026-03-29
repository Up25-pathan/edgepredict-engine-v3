/**
 * @file SurfaceRoughnessPredictor.cpp
 * @brief Surface roughness prediction implementation
 */

#include "SurfaceRoughnessPredictor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

SurfaceRoughnessPredictor::SurfaceRoughnessPredictor() = default;

void SurfaceRoughnessPredictor::initialize(const Config& config) {
    std::cout << "[SurfaceRoughness] Initializing predictor..." << std::endl;
    
    // Could read target Ra from config if available
    // m_targetRa = config.getQuality().targetRa;
    
    // Material-dependent springback
    const auto& material = config.getMaterial();
    double E = material.youngsModulus;
    double sigma_y = material.yieldStrength;
    
    // Elastic recovery factor: higher E/sigma_y = less springback
    m_materialSpringback = sigma_y / E;  // Typically 0.001 - 0.01
    
    // BUE threshold: fraction of melting point where BUE starts forming
    // Higher for softer materials
    m_bueThreshold = 0.3;
    
    std::cout << "[SurfaceRoughness] Target Ra: " << m_targetRa << " μm" << std::endl;
}

SurfaceRoughnessParams SurfaceRoughnessPredictor::predict(const MachiningOutput& output) {
    SurfaceRoughnessParams params;
    
    // Get parameters from strategy if available
    double feed = 0.1;  // mm/rev default
    double noseRadius = 0.8;  // mm default
    double cuttingSpeed = 100;  // m/min default
    
    if (m_strategy) {
        // Extract from strategy
        feed = output.chipThickness / 2.0 * 1000;  // Approximate from chip
        noseRadius = 0.8;  // Would come from ToolGeometry
    }
    
    // Component 1: Kinematic (ideal) roughness
    params.Ra_kinematic = calculateKinematicRoughness(feed, noseRadius);
    
    // Component 2: Dynamic roughness from vibration
    params.Ra_dynamic = calculateDynamicRoughness();
    
    // Component 3: Plastic/material effects
    params.Ra_plastic = calculatePlasticRoughness(
        output.maxToolTemperature, 
        output.maxToolStress
    );
    
    // BUE effect
    double bueEffect = estimateBUEEffect(cuttingSpeed, output.maxToolTemperature);
    params.Ra_plastic += bueEffect;
    
    // Total Ra (root sum of squares for independent effects)
    params.Ra = std::sqrt(
        params.Ra_kinematic * params.Ra_kinematic +
        params.Ra_dynamic * params.Ra_dynamic +
        params.Ra_plastic * params.Ra_plastic
    );
    
    // Estimate other parameters from Ra
    // Typical ratios for turning/milling
    params.Rz = params.Ra * 4.0;     // Rz ≈ 4 * Ra
    params.Rmax = params.Ra * 6.0;   // Rmax ≈ 6 * Ra
    params.Rq = params.Ra * 1.25;    // Rq ≈ 1.25 * Ra
    
    // Skewness and kurtosis (typical for machined surfaces)
    params.Rsk = -0.3;  // Slightly negative (bearing surfaces)
    params.Rku = 3.0;   // Gaussian distribution
    
    // Mean spacing (from feed marks)
    params.Rsm = feed * 1000;  // μm
    
    // Quality check
    params.targetRa = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);
    
    m_lastPrediction = params;
    return params;
}

SurfaceRoughnessParams SurfaceRoughnessPredictor::predictFromParams(
    double feedPerRev, double noseRadius, double cuttingSpeed) {
    
    SurfaceRoughnessParams params;
    
    // Kinematic component
    params.Ra_kinematic = calculateKinematicRoughness(feedPerRev, noseRadius);
    
    // Dynamic component (use stored vibration)
    params.Ra_dynamic = calculateDynamicRoughness();
    
    // Plastic component (estimate from cutting speed)
    // Higher speed = less plastic deformation
    double plasticFactor = 0.1 / std::max(0.5, cuttingSpeed / 100.0);
    params.Ra_plastic = params.Ra_kinematic * plasticFactor;
    
    // BUE effect
    bool hasBUE = (cuttingSpeed < 50);  // BUE common at low speeds
    if (hasBUE) {
        params.Ra_plastic += params.Ra_kinematic * 0.5;
    }
    
    // Total
    params.Ra = std::sqrt(
        params.Ra_kinematic * params.Ra_kinematic +
        params.Ra_dynamic * params.Ra_dynamic +
        params.Ra_plastic * params.Ra_plastic
    );
    
    params.Rz = params.Ra * 4.0;
    params.Rmax = params.Ra * 6.0;
    params.Rq = params.Ra * 1.25;
    params.Rsm = feedPerRev * 1000;
    
    params.targetRa = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);
    
    m_lastPrediction = params;
    return params;
}

void SurfaceRoughnessPredictor::updateDynamicEffects(
    double toolVibrationAmplitude, double vibrationFrequency) {
    
    m_vibrationAmplitude = toolVibrationAmplitude;
    m_vibrationFrequency = vibrationFrequency;
}

std::vector<ProfilePoint> SurfaceRoughnessPredictor::generateProfileFromParticles(
    const std::vector<SPHParticle>& particles,
    const Vec3& surfaceNormal,
    double profileLength) {
    
    std::vector<ProfilePoint> profile;
    
    if (particles.empty()) {
        return profile;
    }
    
    // Determine profile direction (perpendicular to normal, along feed)
    Vec3 profileDir;
    if (std::abs(surfaceNormal.z) < 0.9) {
        profileDir = Vec3(-surfaceNormal.y, surfaceNormal.x, 0);
    } else {
        profileDir = Vec3(1, 0, 0);
    }
    double norm = std::sqrt(profileDir.x*profileDir.x + profileDir.y*profileDir.y);
    profileDir.x /= norm;
    profileDir.y /= norm;
    
    // Find surface particles (within threshold of surface position)
    double surfaceThreshold = 0.0001;  // 0.1mm
    std::vector<const SPHParticle*> surfaceParticles;
    
    for (const auto& p : particles) {
        // Check if particle is near surface (simplified)
        double distToSurface = std::abs(
            p.x * surfaceNormal.x + 
            p.y * surfaceNormal.y + 
            p.z * surfaceNormal.z
        );
        
        if (distToSurface < surfaceThreshold) {
            surfaceParticles.push_back(&p);
        }
    }
    
    // Sample profile at regular intervals
    int numSamples = static_cast<int>(profileLength * 1000);  // 1 μm spacing
    double sampleSpacing = profileLength / numSamples;
    
    for (int i = 0; i < numSamples; ++i) {
        double pos = i * sampleSpacing;
        
        ProfilePoint point;
        point.position = pos * 1000;  // mm to μm
        
        // Find height at this position (interpolate from nearby particles)
        double height = 0;
        double totalWeight = 0;
        
        for (const auto* p : surfaceParticles) {
            // Distance along profile direction
            double projDist = p->x * profileDir.x + p->y * profileDir.y;
            double diff = std::abs(projDist - pos);
            
            if (diff < sampleSpacing * 2) {
                double weight = 1.0 / (diff + 0.001);
                // Height in surface normal direction
                double h = p->x * surfaceNormal.x + 
                           p->y * surfaceNormal.y + 
                           p->z * surfaceNormal.z;
                height += h * weight;
                totalWeight += weight;
            }
        }
        
        if (totalWeight > 0) {
            point.height = (height / totalWeight) * 1e6;  // m to μm
        }
        
        profile.push_back(point);
    }
    
    // Remove mean line
    double meanHeight = 0;
    for (const auto& p : profile) {
        meanHeight += p.height;
    }
    meanHeight /= profile.size();
    
    for (auto& p : profile) {
        p.height -= meanHeight;
    }
    
    return profile;
}

SurfaceRoughnessParams SurfaceRoughnessPredictor::calculateFromProfile(
    const std::vector<ProfilePoint>& profile) {
    
    SurfaceRoughnessParams params;
    
    if (profile.size() < 10) {
        return params;
    }
    
    // Ra: arithmetic average
    double sumAbs = 0;
    for (const auto& p : profile) {
        sumAbs += std::abs(p.height);
    }
    params.Ra = sumAbs / profile.size();
    
    // Rq: RMS
    double sumSquares = 0;
    for (const auto& p : profile) {
        sumSquares += p.height * p.height;
    }
    params.Rq = std::sqrt(sumSquares / profile.size());
    
    // Rmax: max peak to valley
    double maxHeight = profile[0].height;
    double minHeight = profile[0].height;
    for (const auto& p : profile) {
        maxHeight = std::max(maxHeight, p.height);
        minHeight = std::min(minHeight, p.height);
    }
    params.Rmax = maxHeight - minHeight;
    
    // Rz: average of 5 highest peaks and 5 lowest valleys
    std::vector<double> heights;
    for (const auto& p : profile) {
        heights.push_back(p.height);
    }
    std::sort(heights.begin(), heights.end());
    
    int n = std::min(5, static_cast<int>(heights.size() / 10));
    double sumPeaks = 0, sumValleys = 0;
    for (int i = 0; i < n; ++i) {
        sumValleys += heights[i];
        sumPeaks += heights[heights.size() - 1 - i];
    }
    params.Rz = (sumPeaks - sumValleys) / n;
    
    // Skewness
    double sumCubed = 0;
    for (const auto& p : profile) {
        sumCubed += std::pow(p.height, 3);
    }
    if (params.Rq > 0) {
        params.Rsk = sumCubed / (profile.size() * std::pow(params.Rq, 3));
    }
    
    // Kurtosis
    double sumFourth = 0;
    for (const auto& p : profile) {
        sumFourth += std::pow(p.height, 4);
    }
    if (params.Rq > 0) {
        params.Rku = sumFourth / (profile.size() * std::pow(params.Rq, 4));
    }
    
    // Mean spacing (simple zero-crossing method)
    int zeroCrossings = 0;
    for (size_t i = 1; i < profile.size(); ++i) {
        if ((profile[i-1].height < 0 && profile[i].height >= 0) ||
            (profile[i-1].height >= 0 && profile[i].height < 0)) {
            zeroCrossings++;
        }
    }
    if (zeroCrossings > 0) {
        double totalLength = profile.back().position - profile.front().position;
        params.Rsm = totalLength / (zeroCrossings / 2.0);
    }
    
    // Quality check
    params.targetRa = m_targetRa;
    params.meetsTolerance = (params.Ra <= m_targetRa);
    
    m_lastPrediction = params;
    return params;
}

double SurfaceRoughnessPredictor::calculateKinematicRoughness(
    double feed, double noseRadius) const {
    
    // Classic formula: Ra = f² / (32 * r)
    // Where f = feed per rev (mm), r = nose radius (mm)
    // Result in mm, convert to μm
    
    if (noseRadius <= 0) {
        noseRadius = 0.4;  // Default 0.4mm
    }
    
    double Ra_mm = (feed * feed) / (32.0 * noseRadius);
    return Ra_mm * 1000;  // mm to μm
}

double SurfaceRoughnessPredictor::calculateDynamicRoughness() const {
    // Contribution from tool vibration
    // Ra_dynamic ≈ vibration_amplitude / 4 (empirical)
    
    if (m_vibrationAmplitude <= 0) {
        return 0;
    }
    
    // High frequency vibration has less effect on Ra
    double freqFactor = 1.0 / (1.0 + m_vibrationFrequency / 1000.0);
    
    return m_vibrationAmplitude * 0.25 * freqFactor;
}

double SurfaceRoughnessPredictor::calculatePlasticRoughness(
    double temperature, double stress) const {
    
    // Plastic side flow and material deformation
    // Increases with temperature (softening) and stress
    
    double basePlastic = 0.1;  // μm base contribution
    
    // Temperature effect (normalized to 1000°C)
    double tempFactor = temperature / 1000.0;
    
    // Stress effect (normalized to 1 GPa)
    double stressFactor = stress / 1e9;
    
    return basePlastic * (1.0 + tempFactor + stressFactor);
}

double SurfaceRoughnessPredictor::estimateBUEEffect(
    double cuttingSpeed, double temperature) const {
    
    // BUE (Built-Up Edge) forms at:
    // - Low cutting speeds (< 50 m/min typically)
    // - Specific temperature range (warm but not hot)
    
    // BUE zone: ~0.25-0.35 of melting point
    double meltingPoint = 1660;  // Ti-6Al-4V default
    double tempRatio = temperature / meltingPoint;
    
    // Check if in BUE formation zone
    bool inBUEZone = (tempRatio > 0.2 && tempRatio < 0.4) && (cuttingSpeed < 80);
    
    if (!inBUEZone) {
        return 0;
    }
    
    // BUE severity (worst at ~0.3 temp ratio and ~30 m/min)
    double tempSeverity = 1.0 - std::abs(tempRatio - 0.3) / 0.1;
    double speedSeverity = 1.0 - cuttingSpeed / 100.0;
    
    double severity = std::max(0.0, tempSeverity) * std::max(0.0, speedSeverity);
    
    // BUE can double or triple the roughness
    return severity * 2.0;  // μm contribution
}

} // namespace edgepredict
