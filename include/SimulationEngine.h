#pragma once
/**
 * @file SimulationEngine.h
 * @brief Main simulation orchestrator - CLEAN and MINIMAL
 * 
 * This class is intentionally small (~100 lines vs old 371 lines).
 * It only orchestrates the simulation loop - all actual work is delegated.
 */

#include "Types.h"
#include "Config.h"
#include "IPhysicsSolver.h"
#include "IMachiningStrategy.h"
#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include "AsyncExporter.h"

namespace edgepredict {

// Forward declarations
class GeometryLoader;
class GCodeInterpreter;
class IExporter;
class IPhysicsSolver;
struct SPHParticle;
class ContactSolver;
class CFDSolverGPU;
class OptimizationManager;
class SurfaceRoughnessPredictor;
class ResidualStressAnalyzer;

/**
 * @brief Simulation step callback signature
 */
using StepCallback = std::function<void(int step, double time, double maxStress, double maxTemp)>;

/**
 * @brief Simulation completion callback
 */
using CompletionCallback = std::function<void(bool success, const std::string& message)>;

/**
 * @brief Main simulation engine orchestrator
 * 
 * Responsibilities:
 * - Load configuration
 * - Initialize solvers via dependency injection
 * - Run simulation loop
 * - Coordinate solver interactions
 * - Export results
 * 
 * NOT responsible for:
 * - Actual physics computation (delegated to solvers)
 * - Geometry loading (delegated to GeometryLoader)
 * - File export (delegated to Exporters)
 */
class SimulationEngine {
public:
    SimulationEngine();
    ~SimulationEngine();
    
    // No copy
    SimulationEngine(const SimulationEngine&) = delete;
    SimulationEngine& operator=(const SimulationEngine&) = delete;
    
    /**
     * @brief Initialize engine with configuration file
     * @param configPath Path to JSON configuration file
     * @return true if initialization succeeded
     */
    bool initialize(const std::string& configPath);
    
    /**
     * @brief Initialize with pre-loaded configuration
     * @param config Configuration object
     * @return true if initialization succeeded
     */
    bool initialize(const Config& config);
    
    /**
     * @brief Run the full simulation
     * 
     * This is the main entry point. It:
     * 1. Loads geometry
     * 2. Initializes solvers
     * 3. Runs the time-stepping loop
     * 4. Exports results
     */
    void run();
    
    /**
     * @brief Run a single time step (for external control)
     * @param dt Time step size
     * @return true if step completed successfully
     */
    bool step(double dt);
    
    /**
     * @brief Stop simulation gracefully
     */
    void stop();
    
    /**
     * @brief Check if simulation is running
     */
    bool isRunning() const { return m_isRunning; }
    
    /**
     * @brief Check if engine is initialized
     */
    bool isInitialized() const { return m_isInitialized; }
    
    // --- Dependency Injection ---
    
    /**
     * @brief Register a physics solver
     * @param solver Solver instance (ownership transferred)
     */
    void addSolver(std::unique_ptr<IPhysicsSolver> solver);
    
    /**
     * @brief Register an exporter
     * @param exporter Exporter instance (ownership transferred)
     */
    void addExporter(std::unique_ptr<IExporter> exporter);
    
    /**
     * @brief Set machining strategy
     * @param strategy Strategy instance (ownership transferred)
     */
    void setStrategy(std::unique_ptr<IMachiningStrategy> strategy);
    
    /**
     * @brief Get current machining strategy
     */
    IMachiningStrategy* getStrategy() const { return m_strategy.get(); }
    
    /**
     * @brief Set optimization manager for adaptive control
     */
    void setOptimizationManager(std::unique_ptr<OptimizationManager> manager) { 
        m_optimizationManager = std::move(manager); 
    }
    
    /**
     * @brief Set roughness predictor for final analysis
     */
    void setRoughnessPredictor(std::unique_ptr<SurfaceRoughnessPredictor> predictor) {
        m_roughnessPredictor = std::move(predictor);
    }
    
    /**
     * @brief Set stress analyzer for final results
     */
    void setStressAnalyzer(std::unique_ptr<ResidualStressAnalyzer> analyzer) {
        m_stressAnalyzer = std::move(analyzer);
    }
    
    /**
     * @brief Set contact solver (non-owning pointer)
     * @param solver ContactSolver instance (lifetime managed by caller)
     */
    void setContactSolver(ContactSolver* solver);
    
    /**
     * @brief Set CFD solver (non-owning pointer)
     * @param solver CFDSolverGPU instance (lifetime managed by caller)
     */
    void setCFDSolver(CFDSolverGPU* solver);
    
    /**
     * @brief Set step callback for progress reporting
     */
    void setStepCallback(StepCallback callback) { m_stepCallback = callback; }
    
    /**
     * @brief Set completion callback
     */
    void setCompletionCallback(CompletionCallback callback) { m_completionCallback = callback; }
    
    // --- Accessors ---
    const Config& getConfig() const { return m_config; }
    const Mesh& getToolMesh() const { return m_toolMesh; }
    double getCurrentTime() const { return m_currentTime; }
    int getCurrentStep() const { return m_currentStep; }
    
    // --- Metrics ---
    double getMaxStress() const { return m_maxStress; }
    double getMaxTemperature() const { return m_maxTemperature; }
    double getElapsedWallTime() const;

private:
    void loadGeometry();
    void initializeSolvers();
    void runMainLoop();
    void exportResults();
    void updateMetrics();
    double computeAdaptiveTimeStep() const;
    double computeAirGap() const;
    void handleAirCutting(double& dt);
    
    // Configuration
    Config m_config;
    
    // Geometry
    Mesh m_toolMesh;
    std::unique_ptr<GeometryLoader> m_geometryLoader;
    std::unique_ptr<GCodeInterpreter> m_gcodeInterpreter;
    
    // Machining strategy
    std::unique_ptr<IMachiningStrategy> m_strategy;
    
    // Solvers (dependency injection)
    std::vector<std::unique_ptr<IPhysicsSolver>> m_solvers;
    
    // Exporters
    std::vector<std::unique_ptr<IExporter>> m_exporters;
    std::unique_ptr<AsyncExporter> m_asyncExporter;
    
    // Contact solver (non-owning - lifetime managed by main())
    ContactSolver* m_contactSolver = nullptr;
    
    // CFD solver (non-owning - lifetime managed by main())
    CFDSolverGPU* m_cfdSolver = nullptr;
    
    // Advanced Physics & Optimization
    std::unique_ptr<OptimizationManager> m_optimizationManager;
    std::unique_ptr<SurfaceRoughnessPredictor> m_roughnessPredictor;
    std::unique_ptr<ResidualStressAnalyzer> m_stressAnalyzer;
    
    // Callbacks
    StepCallback m_stepCallback;
    CompletionCallback m_completionCallback;
    
    // State
    bool m_isInitialized = false;
    bool m_isRunning = false;
    bool m_shouldStop = false;
    
    // Simulation state
    double m_currentTime = 0.0;
    int m_currentStep = 0;
    
    // Metrics
    double m_maxStress = 0.0;
    double m_maxTemperature = 0.0;
    
    // Timing
    std::chrono::high_resolution_clock::time_point m_startTime;
};

/**
 * @brief Interface for result exporters
 */
class IExporter {
public:
    virtual ~IExporter() = default;
    virtual std::string getName() const = 0;
    virtual void exportStep(int step, double time, const Mesh& mesh) = 0;
    virtual void exportFinal(const Config& config, const Mesh& mesh) = 0;
    
    // Optional methods that specific exporters can implement
    virtual void exportParticles(int step, double time, const std::vector<SPHParticle>& particles) {}
    virtual void exportMetrics(int step, double time, double maxStress, double maxTemp) {}
};

} // namespace edgepredict
