/**
 * @file TurningStrategy.cpp
 * @brief Turning operation strategy implementation
 */

#include "TurningStrategy.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "CFDSolverGPU.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace edgepredict {

namespace {
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;
}

TurningStrategy::TurningStrategy() = default;

bool TurningStrategy::initialize(const Config& config) {
    std::cout << "[TurningStrategy] Initializing..." << std::endl;
    
    const auto& machining = config.getMachining();
    const auto& material = config.getMaterial();
    
    // Set conditions from config
    m_conditions.spindleSpeed = machining.rpm * 2.0 * PI / 60.0;  // RPM to rad/s
    m_conditions.feedRate = machining.feedRateMmMin / 60.0 / 1000.0;  // mm/min to m/s
    m_conditions.depthOfCut = machining.depthOfCutMm / 1000.0;  // mm to m
    
    // Parse Initial Position (with safe JSON access)
    std::vector<double> initPos = {0.0, 0.0, 0.0};
    if (config.getJson().contains("machining_parameters") &&
        config.getJson()["machining_parameters"].contains("initial_tool_position_mm")) {
         initPos = config.getJson()["machining_parameters"]["initial_tool_position_mm"].get<std::vector<double>>();
    }
    m_conditions.toolPosition = Vec3(initPos[0]/1000.0, initPos[1]/1000.0, initPos[2]/1000.0);
    
    // Specific cutting force from material
    // Approximate from yield strength: Kc ≈ 2.5 * σy for metals
    m_specificCuttingForce = 2.5 * material.jc_A;
    
    // Default insert if not set
    if (m_insert.inscribedCircle == 0) {
        m_insert.shape = 'C';
        m_insert.inscribedCircle = 12.7;
        m_insert.noseRadius = 0.8;
        m_insert.thickness = 4.76;
    }
    
    // Convert insert geometry to ToolGeometry
    m_geometry.noseRadius = m_insert.noseRadius / 1000.0;  // mm to m
    m_geometry.rakeAngle = 6.0;  // Typical positive rake for turning
    m_geometry.clearanceAngle = 7.0;
    
    // Set insert angle from shape
    switch (m_insert.shape) {
        case 'C': m_geometry.insertAngle = 80.0; break;
        case 'D': m_geometry.insertAngle = 55.0; break;
        case 'T': m_geometry.insertAngle = 60.0; break;
        case 'V': m_geometry.insertAngle = 35.0; break;
        case 'S': m_geometry.insertAngle = 90.0; break;
        case 'W': m_geometry.insertAngle = 80.0; break;
        default:  m_geometry.insertAngle = 80.0; break;
    }
    
    m_initialized = true;
    
    std::cout << "[TurningStrategy] Insert: " << m_insert.shape 
              << ", IC=" << m_insert.inscribedCircle << "mm"
              << ", R=" << m_insert.noseRadius << "mm" << std::endl;
    std::cout << "[TurningStrategy] Kc=" << m_specificCuttingForce / 1e6 << " MPa" << std::endl;
    
    return true;
}

void TurningStrategy::setToolGeometry(const ToolGeometry& geometry) {
    m_geometry = geometry;
}

void TurningStrategy::setInsert(const InsertDesignation& insert) {
    m_insert = insert;
    
    // Update geometry from insert
    m_geometry.noseRadius = insert.noseRadius / 1000.0;
    
    switch (insert.shape) {
        case 'C': m_geometry.insertAngle = 80.0; break;
        case 'D': m_geometry.insertAngle = 55.0; break;
        case 'T': m_geometry.insertAngle = 60.0; break;
        case 'V': m_geometry.insertAngle = 35.0; break;
        case 'S': m_geometry.insertAngle = 90.0; break;
        default:  m_geometry.insertAngle = 80.0; break;
    }
}

void TurningStrategy::connectSolvers(SPHSolver* sph, FEMSolver* fem, 
                                       ContactSolver* contact, CFDSolverGPU* cfd) {
    m_sph = sph;
    m_fem = fem;
    m_contact = contact;
    m_cfd = cfd;
}

void TurningStrategy::updateConditions(const MachineState& state, double dt) {
    // Update from machine state
    m_conditions.spindleSpeed = state.spindleRPM * 2.0 * PI / 60.0 * m_adaptiveSpeedMultiplier;
    m_conditions.toolPosition = state.position;
    
    // Calculate cutting speed from workpiece diameter
    // Vc = π * D * N / 60000 (if D in mm, N in RPM, Vc in m/min)
    double diameter = m_workpieceDiameter;
    m_conditions.cuttingSpeed = PI * diameter * state.spindleRPM / 60.0;  // m/s
    
    // Feed rate: distance per revolution
    double feedMmPerMin = state.feedRate * 60000.0;  // m/s to mm/min
    m_conditions.feedRate = feedMmPerMin / 60.0 / 1000.0 * 
                            (60.0 / state.spindleRPM) * m_adaptiveFeedMultiplier;  // m/rev
    
    // Update workpiece rotation
    m_workpieceRotationAngle += m_conditions.spindleSpeed * dt;
    if (m_workpieceRotationAngle > 2 * PI) {
        m_workpieceRotationAngle -= 2 * PI;
    }
    
    // Directions
    // In turning: cutting direction is tangent to workpiece
    // Feed direction is along Z axis (typically)
    m_conditions.cuttingDirection = Vec3(
        -std::sin(m_workpieceRotationAngle),
        std::cos(m_workpieceRotationAngle),
        0
    );
    m_conditions.feedDirection = Vec3(0, 0, -1);  // Towards chuck
}

std::vector<CuttingEdge> TurningStrategy::getActiveCuttingEdges() const {
    std::vector<CuttingEdge> edges;
    
    // Main cutting edge (determined by insert angle and position)
    CuttingEdge mainEdge;
    
    // Edge geometry based on insert angle
    double halfAngle = m_geometry.insertAngle * DEG_TO_RAD / 2.0;
    double edgeLength = m_insert.inscribedCircle / 1000.0;  // Approximate
    
    // Position relative to tool holder
    mainEdge.startPoint = m_conditions.toolPosition;
    mainEdge.endPoint = Vec3(
        m_conditions.toolPosition.x + edgeLength * std::cos(halfAngle),
        m_conditions.toolPosition.y,
        m_conditions.toolPosition.z + edgeLength * std::sin(halfAngle)
    );
    
    // Normal points into workpiece (radially inward)
    mainEdge.normal = Vec3(-1, 0, 0);  // For OD turning
    
    // Rake direction (chip flow)
    mainEdge.rakeDirection = Vec3(
        std::sin(m_geometry.rakeAngle * DEG_TO_RAD),
        0,
        std::cos(m_geometry.rakeAngle * DEG_TO_RAD)
    );
    
    mainEdge.localRakeAngle = m_geometry.rakeAngle;
    
    // Calculate engagement based on depth of cut vs edge length
    double chipWidth = computeChipWidth();
    mainEdge.engagement = std::min(1.0, chipWidth / edgeLength);
    mainEdge.isActive = (mainEdge.engagement > 0);
    
    if (mainEdge.isActive) {
        edges.push_back(mainEdge);
    }
    
    // Nose radius engagement (if chip width is significant)
    if (m_conditions.depthOfCut < m_geometry.noseRadius) {
        // Nose radius is doing most of the cutting
        CuttingEdge noseEdge;
        noseEdge.startPoint = mainEdge.startPoint;
        noseEdge.endPoint = Vec3(
            mainEdge.startPoint.x - m_geometry.noseRadius,
            mainEdge.startPoint.y,
            mainEdge.startPoint.z
        );
        noseEdge.normal = mainEdge.normal;
        noseEdge.rakeDirection = mainEdge.rakeDirection;
        noseEdge.localRakeAngle = m_geometry.rakeAngle;
        noseEdge.engagement = 1.0;
        noseEdge.isActive = true;
        edges.push_back(noseEdge);
    }
    
    return edges;
}

MachiningOutput TurningStrategy::computeOutput() const {
    MachiningOutput output;
    
    // Undeformed chip dimensions
    double h = computeUndeformedChipThickness();  // chip thickness
    double b = computeChipWidth();                 // chip width
    double ap = m_conditions.depthOfCut;
    double f = m_conditions.feedRate;
    
    // Chip cross-section area
    double Ac = h * b;
    
    // Cutting force using Kienzle equation
    // Fc = Kc * A = Kc1.1 * b * h^(1-mc)
    // Simplified: Fc = Kc * Ac
    double Fc = m_specificCuttingForce * Ac;
    
    // Force components
    // Fc = tangential (cutting direction)
    // Ff = feed direction
    // Fp = passive/radial
    
    // Empirical ratios for turning (typical for steel/titanium)
    double kf = 0.4;   // Ff / Fc
    double kp = 0.3;   // Fp / Fc
    
    // Adjust for rake angle
    double rakeRad = m_geometry.rakeAngle * DEG_TO_RAD;
    kf *= (1.0 - 0.5 * std::sin(rakeRad));
    kp *= (1.0 + 0.3 * std::sin(rakeRad));
    
    // Adjust for nose radius effect (larger nose = more passive force)
    kp *= (1.0 + m_geometry.noseRadius / ap);
    
    double Ff = kf * Fc;
    double Fp = kp * Fc;
    
    // Set force components
    output.cuttingForce = Vec3(
        Fc * m_conditions.cuttingDirection.x,
        Fc * m_conditions.cuttingDirection.y,
        Fc * m_conditions.cuttingDirection.z
    );
    output.feedForce = Vec3(
        Ff * m_conditions.feedDirection.x,
        Ff * m_conditions.feedDirection.y,
        Ff * m_conditions.feedDirection.z
    );
    output.thrustForce = Vec3(-Fp, 0, 0);  // Radial, pointing away from workpiece
    
    // Torque = Fc * radius
    double radius = m_workpieceDiameter / 2.0;
    output.torque = Fc * radius;
    
    // Power = Fc * Vc
    output.power = Fc * m_conditions.cuttingSpeed;
    
    // Chip formation
    output.chipThickness = h * 2.0;  // Chip compression ratio ~2 for titanium
    output.chipWidth = b;
    output.chipVelocity = m_conditions.cuttingSpeed / 2.0;  // Approximate
    
    // Material removal rate: MRR = Vc * f * ap (in m³/s)
    output.materialRemovalRate = m_conditions.cuttingSpeed * f * ap * 1e9;  // mm³/s
    
    // Theoretical surface roughness (from feed and nose radius)
    // Ra ≈ f² / (32 * r)  (simplified formula)
    if (m_geometry.noseRadius > 0) {
        double f_mm = f * 1000;  // to mm
        double r_mm = m_geometry.noseRadius * 1000;
        output.theoreticalRoughness = (f_mm * f_mm) / (32.0 * r_mm);  // μm
    }
    
    // Get actual values from solvers if connected
    if (m_fem) {
        output.maxToolStress = m_fem->getMaxStress();
        output.maxToolTemperature = m_fem->getMaxTemperature();
    }
    
    return output;
}

void TurningStrategy::applyKinematics(double dt) {
    // In turning: workpiece rotates, tool is stationary (except feed)
    
    // Rotate all workpiece SPH particles around Z axis using GPU kernel
    if (m_sph) {
        double angle = m_conditions.spindleSpeed * dt;
        m_sph->rotateWorkpieceZ(angle, m_conditions.spindleSpeed);
    }
    
    // Move tool along feed direction
    if (m_fem) {
        Vec3 feedMovement = Vec3(
            m_conditions.feedRate * dt * m_conditions.feedDirection.x,
            m_conditions.feedRate * dt * m_conditions.feedDirection.y,
            m_conditions.feedRate * dt * m_conditions.feedDirection.z
        );
        
        m_fem->translateMesh(feedMovement.x, feedMovement.y, feedMovement.z);
    }
}

void TurningStrategy::applyAdaptiveControl(double feedMultiplier, double speedMultiplier) {
    m_adaptiveFeedMultiplier = feedMultiplier;
    m_adaptiveSpeedMultiplier = speedMultiplier;
    // Don't modify conditions here — updateConditions() already applies these multipliers
}

void TurningStrategy::reset() {
    m_workpieceRotationAngle = 0;
    m_conditions = CuttingConditions();
    m_initialized = false;
}

double TurningStrategy::computeUndeformedChipThickness() const {
    // h = f * sin(κ), where κ is approach angle
    // For general turning: h ≈ f (simplified when κ = 90°)
    double approachAngle = 90.0 * DEG_TO_RAD;  // Could be from insert geometry
    return m_conditions.feedRate * std::sin(approachAngle);
}

double TurningStrategy::computeChipWidth() const {
    // b = ap / sin(κ)
    double approachAngle = 90.0 * DEG_TO_RAD;
    return m_conditions.depthOfCut / std::sin(approachAngle);
}

double TurningStrategy::computeEngagementAngle() const {
    // For turning with nose radius
    if (m_geometry.noseRadius > 0 && m_conditions.depthOfCut < m_geometry.noseRadius) {
        // Only nose radius is engaged
        return std::acos(1.0 - m_conditions.depthOfCut / m_geometry.noseRadius);
    }
    return PI / 2.0;  // Full 90 degree engagement
}

double TurningStrategy::getChipCurlRadius() const {
    // Empirical formula for chip curl
    // R_chip ≈ K * h / (1 - n)
    // Where K is material constant, n is rake angle effect
    double h = computeUndeformedChipThickness();
    double K = 2.5;  // Typical for titanium
    double n = 0.1 * m_geometry.rakeAngle / 10.0;  // Normalized
    
    return K * h / (1.0 - n);
}

double TurningStrategy::getSpecificCuttingForce() const {
    // Adjust Kc for chip thickness (Kienzle correction)
    double h = computeUndeformedChipThickness();
    double mc = 0.25;  // Material constant (0.2-0.3 for metals)
    
    // Kc = Kc1.1 * h^(-mc)
    // Kc1.1 is the specific cutting force at h=1mm
    double h_mm = h * 1000;
    if (h_mm < 0.01) h_mm = 0.01;  // Prevent divide by zero
    
    return m_specificCuttingForce * std::pow(h_mm, -mc);
}

} // namespace edgepredict
