#pragma once
/**
 * @file Types.h
 * @brief Core type definitions for EdgePredict Engine v4
 * 
 * IMPORTANT: This is the SINGLE source of truth for all enums and basic types.
 * Do NOT define these types anywhere else to avoid conflicts.
 */

#include <cstdint>
#include <cmath>
#include <vector>

// ============================================================================
// CUDA Compatibility Macros
// ============================================================================
#ifdef __CUDACC__
    #define EP_HOST_DEVICE __host__ __device__
    #define EP_DEVICE __device__
    #define EP_HOST __host__
#else
    #define EP_HOST_DEVICE
    #define EP_DEVICE
    #define EP_HOST
#endif

namespace edgepredict {

// ============================================================================
// Enums (Single Definition - No Duplicates!)
// ============================================================================

/**
 * @brief Status of SPH particles in the simulation
 */
enum class ParticleStatus : int32_t {
    ACTIVE = 0,         // Active workpiece material
    INACTIVE = 1,       // Removed/deleted particle
    CHIP = 2,           // Detached chip particle
    BOUNDARY = 3        // Boundary/ghost particle
};

/**
 * @brief Level of Detail zone for particles
 * Particles far from the tool can use reduced physics updates
 */
enum class LODZone : int32_t {
    ACTIVE = 0,         // Near tool: full physics every step
    ZONE_NEAR = 1,      // Medium distance: reduced update frequency
    ZONE_FAR = 2        // Far from tool: minimal updates
};

/**
 * @brief Status of FEM nodes
 */
enum class NodeStatus : int32_t {
    OK = 0,             // Normal operating condition
    WORN = 1,           // Wear threshold exceeded
    FAILED = 2,         // Catastrophic failure
    FIXED = 3           // Fixed boundary condition
};

/**
 * @brief Type of machining operation
 */
enum class MachiningType {
    TURNING,
    MILLING,
    DRILLING,
    GRINDING
};

// ============================================================================
// CUDA-Compatible Vector Types
// ============================================================================

/**
 * @brief 3D vector for GPU computation (replaces Eigen in device code)
 */
struct Vec3 {
    double x, y, z;
    
    EP_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
    EP_HOST_DEVICE Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    EP_HOST_DEVICE Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    EP_HOST_DEVICE Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    EP_HOST_DEVICE Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    EP_HOST_DEVICE Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    
    EP_HOST_DEVICE Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    EP_HOST_DEVICE Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    EP_HOST_DEVICE Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    
    EP_HOST_DEVICE double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    EP_HOST_DEVICE Vec3 cross(const Vec3& o) const { 
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; 
    }
    EP_HOST_DEVICE double lengthSq() const { return x * x + y * y + z * z; }
    EP_HOST_DEVICE double length() const { return sqrt(lengthSq()); }
    EP_HOST_DEVICE Vec3 normalized() const { 
        double len = length();
        return len > 1e-12 ? (*this / len) : Vec3(0, 0, 0);
    }
    
    // Zero vector
    EP_HOST_DEVICE static Vec3 zero() { return {0, 0, 0}; }
};

EP_HOST_DEVICE inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ============================================================================
// SPH Particle Structure (Structure of Arrays friendly)
// ============================================================================

/**
 * @brief Single SPH particle data
 */
struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
    Vec3 force;
    
    double mass;
    double density;
    double pressure;
    double temperature;
    
    int32_t id;
    ParticleStatus status;
    
    EP_HOST_DEVICE Particle() 
        : position(), velocity(), acceleration(), force(),
          mass(0), density(0), pressure(0), temperature(25.0),
          id(-1), status(ParticleStatus::ACTIVE) {}
};

// ============================================================================
// FEM Structures
// ============================================================================

/**
 * @brief FEM node for tool stress analysis
 */
struct FEMNode {
    Vec3 position;
    Vec3 originalPosition;
    Vec3 velocity;
    Vec3 force;
    
    double mass;
    double temperature;
    double stress;           // Von Mises stress (Pa)
    double accumulatedWear;  // Usui wear depth (m)
    
    NodeStatus status;
    bool isContact;          // Currently in contact with workpiece
    
    EP_HOST_DEVICE FEMNode()
        : position(), originalPosition(), velocity(), force(),
          mass(1e-9), temperature(25.0), stress(0), accumulatedWear(0),
          status(NodeStatus::OK), isContact(false) {}
};

/**
 * @brief Tetrahedral element for FEM
 */
struct FEMElement {
    int32_t nodeIndices[4];  // Tetrahedron has 4 nodes
    double volume;
    double stiffness;
    
    EP_HOST_DEVICE FEMElement() : volume(0), stiffness(0) {
        for (int i = 0; i < 4; ++i) nodeIndices[i] = -1;
    }
};

/**
 * @brief Triangle element for surface representation
 */
struct Triangle {
    int32_t indices[3];
    Vec3 normal;
    
    EP_HOST_DEVICE Triangle() {
        indices[0] = indices[1] = indices[2] = -1;
    }
};

// ============================================================================
// Mesh Structure
// ============================================================================

/**
 * @brief Surface mesh for geometry representation
 */
struct Mesh {
    std::vector<FEMNode> nodes;
    std::vector<Triangle> triangles;
    
    void clear() {
        nodes.clear();
        triangles.clear();
    }
    
    bool empty() const { return nodes.empty(); }
    size_t nodeCount() const { return nodes.size(); }
    size_t triangleCount() const { return triangles.size(); }
};

// ============================================================================
// Machine State (from G-Code)
// ============================================================================

/**
 * @brief Current machine state from G-Code interpreter
 */
struct MachineState {
    Vec3 position;          // Current tool position (m)
    double feedRate;        // Feed rate (m/s)
    double spindleRPM;      // Spindle speed (RPM)
    bool isActive;          // Is the machine currently moving?
    
    MachineState() : position(), feedRate(0), spindleRPM(0), isActive(false) {}
};

// ============================================================================
// Simulation Constants
// ============================================================================

namespace constants {
    constexpr double PI = 3.14159265358979323846;
    constexpr double GRAVITY = 9.81;                    // m/s^2
    constexpr double BOLTZMANN = 1.380649e-23;          // J/K
    constexpr double STEFAN_BOLTZMANN = 5.670374e-8;    // W/(m^2·K^4)
    
    // Default material properties (Ti-6Al-4V)
    constexpr double DEFAULT_DENSITY = 4430.0;          // kg/m^3
    constexpr double DEFAULT_SPECIFIC_HEAT = 526.3;     // J/(kg·K)
    constexpr double DEFAULT_THERMAL_CONDUCTIVITY = 6.7; // W/(m·K)
    constexpr double DEFAULT_MELTING_POINT = 1660.0;    // °C
}

} // namespace edgepredict
