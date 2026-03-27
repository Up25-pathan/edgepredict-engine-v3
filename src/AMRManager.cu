/**
 * @file AMRManager.cu
 * @brief Implementation of Adaptive Mesh Refinement for SPH particles
 */

#include "AMRManager.cuh"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace edgepredict {

// ============================================================================
// AMRManager Implementation
// ============================================================================

AMRManager::AMRManager() = default;
AMRManager::~AMRManager() = default;

void AMRManager::initialize(const AMRConfig& config) {
    m_config = config;
    m_lastUpdateStep = 0;
    m_lastSplitCount = 0;
    m_lastMergeCount = 0;
    
    if (m_config.enabled) {
        std::cout << "[AMRManager] Initialized:" << std::endl;
        std::cout << "  Refine radius: " << m_config.refineRadius * 1000 << " mm" << std::endl;
        std::cout << "  Coarsen radius: " << m_config.coarsenRadius * 1000 << " mm" << std::endl;
        std::cout << "  Min particle size: " << m_config.minParticleSize * 1e6 << " um" << std::endl;
        std::cout << "  Max particle size: " << m_config.maxParticleSize * 1e6 << " um" << std::endl;
    }
}

void AMRManager::updateRefinement(SPHSolver& solver, int currentStep) {
    if (!m_config.enabled) return;
    
    // Only update every N steps
    if (currentStep - m_lastUpdateStep < m_config.refineIntervalSteps) {
        return;
    }
    m_lastUpdateStep = currentStep;
    
    // Get particles from solver (must copy from device first)
    // Note: This is expensive, so we don't do it every step
    auto& particles = solver.getHostParticles();
    if (particles.empty()) return;
    
    int originalCount = static_cast<int>(particles.size());
    m_lastSplitCount = 0;
    m_lastMergeCount = 0;
    
    // Ensure we have level tracking for all particles
    if (m_particleLevels.size() < particles.size()) {
        m_particleLevels.resize(particles.size(), 0);
    }
    
    // Phase 1: Find and execute splits (particles near tool)
    std::vector<int> splitCandidates = findSplitCandidates(particles);
    std::vector<SPHParticle> newParticles;
    
    // Limit splits to avoid explosion
    int maxSplitsPerUpdate = 1000;
    int splitsDone = 0;
    
    for (int idx : splitCandidates) {
        if (splitsDone >= maxSplitsPerUpdate) break;
        if (particles.size() + newParticles.size() + 8 > static_cast<size_t>(m_config.maxParticles)) break;
        
        // Only split if not already at max refinement
        if (m_particleLevels[idx] < 5) {
            splitParticle(particles, idx, newParticles);
            m_particleLevels[idx] = -1; // Mark for removal
            splitsDone++;
        }
    }
    
    // Phase 2: Find and execute merges (particles far from tool)
    // Skip merges for now - they require complex neighbor finding
    // and can destabilize the simulation if not done carefully
    
    // Phase 3: Apply changes
    // Remove split parents (marked with level -1)
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [this, &particles](const SPHParticle& p) {
                int idx = static_cast<int>(&p - particles.data());
                return idx < static_cast<int>(m_particleLevels.size()) && 
                       m_particleLevels[idx] == -1;
            }),
        particles.end()
    );
    
    // Add new particles
    for (auto& np : newParticles) {
        particles.push_back(np);
        m_particleLevels.push_back(m_particleLevels[0] + 1); // Increment level
    }
    
    m_lastSplitCount = splitsDone;
    
    // Resize level tracking
    m_particleLevels.resize(particles.size(), 0);
    
    // Copy back to device
    solver.updateParticlesFromHost();
    
    if (m_lastSplitCount > 0 || m_lastMergeCount > 0) {
        std::cout << "[AMRManager] Step " << currentStep 
                  << ": Split " << m_lastSplitCount 
                  << ", Merged " << m_lastMergeCount 
                  << ", Total particles: " << particles.size() << std::endl;
    }
}

std::vector<int> AMRManager::findSplitCandidates(const std::vector<SPHParticle>& particles) {
    std::vector<int> candidates;
    double refineRadiusSq = m_config.refineRadius * m_config.refineRadius;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        
        // Skip inactive particles
        if (p.status != ParticleStatus::ACTIVE) continue;
        
        // Skip already-refined particles
        if (i < m_particleLevels.size() && m_particleLevels[i] >= 5) continue;
        
        // Check distance to tool
        double dx = p.x - m_toolPosition.x;
        double dy = p.y - m_toolPosition.y;
        double dz = p.z - m_toolPosition.z;
        double distSq = dx*dx + dy*dy + dz*dz;
        
        if (distSq < refineRadiusSq) {
            candidates.push_back(static_cast<int>(i));
        }
    }
    
    return candidates;
}

std::vector<std::vector<int>> AMRManager::findMergeGroups(const std::vector<SPHParticle>& particles) {
    // Placeholder - merge logic is complex and can be added later
    // Would need spatial hashing to find groups of 8 nearby coarse particles
    return {};
}

void AMRManager::splitParticle(std::vector<SPHParticle>& particles, int parentIdx,
                                std::vector<SPHParticle>& newParticles) {
    const SPHParticle& parent = particles[parentIdx];
    
    // Calculate child properties
    double childMass = parent.mass / 8.0;
    double offset = parent.mass > 0 ? std::cbrt(parent.mass / 4430.0) * 0.25 : 0.00005; // ~quarter of particle size
    
    // Create 8 children in a 2x2x2 pattern
    for (int dz = -1; dz <= 1; dz += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            for (int dx = -1; dx <= 1; dx += 2) {
                SPHParticle child;
                
                // Position (offset from parent center)
                child.x = parent.x + dx * offset;
                child.y = parent.y + dy * offset;
                child.z = parent.z + dz * offset;
                
                // Velocity (same as parent - momentum conserved)
                child.vx = parent.vx;
                child.vy = parent.vy;
                child.vz = parent.vz;
                child.vhx = parent.vhx;
                child.vhy = parent.vhy;
                child.vhz = parent.vhz;
                
                // Other properties (divided/copied)
                child.mass = childMass;
                child.density = parent.density;
                child.pressure = parent.pressure;
                child.temperature = parent.temperature;
                child.status = parent.status;
                child.lodZone = LODZone::ACTIVE; // New particles start in active zone
                child.id = static_cast<int32_t>(particles.size() + newParticles.size());
                
                newParticles.push_back(child);
            }
        }
    }
}

void AMRManager::mergeParticles(std::vector<SPHParticle>& particles,
                                 const std::vector<int>& mergeIndices,
                                 SPHParticle& resultParticle) {
    if (mergeIndices.empty()) return;
    
    // Compute combined mass and momentum-weighted velocity
    double totalMass = 0;
    double momX = 0, momY = 0, momZ = 0;
    double comX = 0, comY = 0, comZ = 0; // Center of mass
    double avgTemp = 0;
    
    for (int idx : mergeIndices) {
        const auto& p = particles[idx];
        double m = p.mass;
        totalMass += m;
        momX += m * p.vx;
        momY += m * p.vy;
        momZ += m * p.vz;
        comX += m * p.x;
        comY += m * p.y;
        comZ += m * p.z;
        avgTemp += p.temperature;
    }
    
    // Set result particle properties
    resultParticle.mass = totalMass;
    resultParticle.x = comX / totalMass;
    resultParticle.y = comY / totalMass;
    resultParticle.z = comZ / totalMass;
    resultParticle.vx = momX / totalMass;
    resultParticle.vy = momY / totalMass;
    resultParticle.vz = momZ / totalMass;
    resultParticle.vhx = resultParticle.vx;
    resultParticle.vhy = resultParticle.vy;
    resultParticle.vhz = resultParticle.vz;
    resultParticle.temperature = avgTemp / mergeIndices.size();
    resultParticle.status = ParticleStatus::ACTIVE;
    resultParticle.lodZone = LODZone::ZONE_FAR;
}

} // namespace edgepredict
