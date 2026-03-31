#pragma once
/**
 * @file TurningStrategy.h
 * @brief Turning/Lathe operation strategy
 * 
 * Handles:
 * - Insert geometry (CNMG, TNMG, VNMG, etc.)
 * - Stationary tool + rotating workpiece kinematics
 * - Chip curl around nose radius
 * - Specific force calculations for turning
 */

#include "IMachiningStrategy.h"
#include <cmath>

namespace edgepredict {

/**
 * @brief Insert designation standard (ISO)
 */
struct InsertDesignation {
    char shape = 'C';               // C=80°, T=60°, V=35°, S=90°, D=55°
    char clearance = 'N';           // N=0°, P=11°, etc.
    char tolerance = 'M';           // M=±0.08mm
    char chipBreaker = 'G';         // Type
    double inscribedCircle = 12.7;  // mm (IC size)
    double thickness = 4.76;        // mm
    double noseRadius = 0.8;        // mm
};

/**
 * @brief Turning operation strategy
 */
class TurningStrategy : public IMachiningStrategy {
public:
    TurningStrategy();
    ~TurningStrategy() override = default;
    
    // IMachiningStrategy interface
    std::string getName() const override { return "TurningStrategy"; }
    MachiningType getType() const override { return MachiningType::TURNING; }
    
    bool initialize(const Config& config) override;
    void setToolGeometry(const ToolGeometry& geometry) override;
    void connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                        ContactSolver* contact, CFDSolverGPU* cfd) override;
    void updateConditions(const MachineState& state, double dt) override;
    std::vector<CuttingEdge> getActiveCuttingEdges() const override;
    MachiningOutput computeOutput() const override;
    void applyKinematics(double dt) override;
    bool isInitialized() const override { return m_initialized; }
    void reset() override;
    void applyAdaptiveControl(double feedMultiplier, double speedMultiplier) override;
    
    // Turning-specific methods
    
    /**
     * @brief Set insert from ISO designation
     */
    void setInsert(const InsertDesignation& insert);
    
    /**
     * @brief Set operation type
     */
    enum class TurningOperation {
        FACING,         // Reduce length
        OD_ROUGHING,    // Outside diameter roughing
        OD_FINISHING,   // Outside diameter finishing
        BORING,         // Internal diameter
        GROOVING,       // Cut grooves
        THREADING,      // Cut threads
        PARTING         // Cut off
    };
    void setOperation(TurningOperation op) { m_operation = op; }
    
    /**
     * @brief Get chip curl radius estimate
     */
    double getChipCurlRadius() const;
    
    /**
     * @brief Get specific cutting force (Kc)
     */
    double getSpecificCuttingForce() const;

private:
    // Solver connections
    SPHSolver* m_sph = nullptr;
    FEMSolver* m_fem = nullptr;
    ContactSolver* m_contact = nullptr;
    CFDSolverGPU* m_cfd = nullptr;
    
    // Configuration
    ToolGeometry m_geometry;
    InsertDesignation m_insert;
    TurningOperation m_operation = TurningOperation::OD_ROUGHING;
    
    // Current state
    CuttingConditions m_conditions;
    double m_workpieceRotationAngle = 0;
    double m_workpieceDiameter = 0.05;  // m
    
    // Material properties (from config)
    double m_specificCuttingForce = 2000e6;  // Pa (Ti-6Al-4V default)
    double m_frictionCoeff = 0.5;
    
    bool m_initialized = false;
    double m_adaptiveFeedMultiplier = 1.0;
    double m_adaptiveSpeedMultiplier = 1.0;
    
    // Helper methods
    void computeCuttingEdgeGeometry();
    double computeUndeformedChipThickness() const;
    double computeChipWidth() const;
    double computeEngagementAngle() const;
};

} // namespace edgepredict
