/**
 * @file SimulationEngine.cpp
 * @brief Simulation orchestrator implementation
 */

#include "SimulationEngine.h"
#include "GeometryLoader.h"
#include "GCodeInterpreter.h"
#include "FEMSolver.cuh"
#include "SPHSolver.cuh"
#include "ContactSolver.cuh"
#include "CudaUtils.cuh"
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace edgepredict {

SimulationEngine::SimulationEngine() = default;
SimulationEngine::~SimulationEngine() = default;

bool SimulationEngine::initialize(const std::string& configPath) {
    try {
        m_config.loadFromFile(configPath);
        return initialize(m_config);
    } catch (const std::exception& e) {
        std::cerr << "[Engine] Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

bool SimulationEngine::initialize(const Config& config) {
    m_config = config;
    
    if (!m_config.isValid()) {
        std::cerr << "[Engine] Invalid configuration" << std::endl;
        return false;
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "   EdgePredict Engine v4.0 (Clean Architecture)   " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Simulation: " << m_config.getSimulationName() << std::endl;
    
    m_isInitialized = true;
    return true;
}

void SimulationEngine::addSolver(std::unique_ptr<IPhysicsSolver> solver) {
    if (solver) {
        std::cout << "[Engine] Added solver: " << solver->getName() << std::endl;
        m_solvers.push_back(std::move(solver));
    }
}

void SimulationEngine::addExporter(std::unique_ptr<IExporter> exporter) {
    if (exporter) {
        std::cout << "[Engine] Added exporter: " << exporter->getName() << std::endl;
        m_exporters.push_back(std::move(exporter));
    }
}

void SimulationEngine::setStrategy(std::unique_ptr<IMachiningStrategy> strategy) {
    if (strategy) {
        std::cout << "[Engine] Set strategy: " << strategy->getName() << std::endl;
        m_strategy = std::move(strategy);
    }
}

void SimulationEngine::setContactSolver(ContactSolver* solver) {
    m_contactSolver = solver;
    if (solver) {
        std::cout << "[Engine] Contact solver registered" << std::endl;
    }
}

void SimulationEngine::run() {
    if (!m_isInitialized) {
        std::cerr << "[Engine] Not initialized!" << std::endl;
        if (m_completionCallback) {
            m_completionCallback(false, "Not initialized");
        }
        return;
    }
    
    m_startTime = std::chrono::high_resolution_clock::now();
    m_isRunning = true;
    m_shouldStop = false;
    
    try {
        // Phase 1: Load geometry
        loadGeometry();
        
        // Phase 2: Initialize all solvers
        initializeSolvers();
        
        // Phase 2.5: Start async I/O
        m_asyncExporter = std::make_unique<AsyncExporter>();
        m_asyncExporter->start();
        
        // Phase 3: Run main simulation loop
        runMainLoop();
        
        // Phase 4: Wait for async exports to complete
        if (m_asyncExporter) {
            m_asyncExporter->flush();
            m_asyncExporter->stop();
        }
        
        // Phase 5: Export final results
        exportResults();
        
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Simulation completed successfully!" << std::endl;
        std::cout << "Wall time: " << std::fixed << std::setprecision(2) 
                  << getElapsedWallTime() << " seconds" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        if (m_completionCallback) {
            m_completionCallback(true, "Completed successfully");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Engine] FATAL ERROR: " << e.what() << std::endl;
        if (m_completionCallback) {
            m_completionCallback(false, e.what());
        }
    }
    
    m_isRunning = false;
}

bool SimulationEngine::step(double dt) {
    if (!m_isInitialized || m_solvers.empty()) {
        return false;
    }
    
    NVTX_PUSH("SimulationEngine::step");
    
    // Update machining strategy if present
    if (m_strategy) {
        MachineState state;
        
        // Use G-Code if available and loaded
        if (m_gcodeInterpreter && m_gcodeInterpreter->isLoaded()) {
            auto snapshot = m_gcodeInterpreter->getStateAtTime(m_currentTime);
            state.position.x = snapshot.position.x;
            state.position.y = snapshot.position.y;
            state.position.z = snapshot.position.z;
            state.spindleRPM = snapshot.spindleRPM;
            state.feedRate = snapshot.feedRate;
        } else {
            // Manual fallback
            state.position.x = 0;
            state.position.y = 0;
            state.position.z = 0;
            state.spindleRPM = m_config.getMachining().rpm;
            state.feedRate = m_config.getMachining().feedRateMmMin / 60000.0; // mm/min to m/s
        }
        
        m_strategy->updateConditions(state, dt);
        m_strategy->applyKinematics(dt);
    }
    
    // Step all solvers
    for (auto& solver : m_solvers) {
        solver->step(dt);
    }
    
    // Resolve contacts between SPH workpiece and FEM tool
    // This is the critical coupling step - without it, tool and workpiece don't interact
    if (m_contactSolver) {
        m_contactSolver->resolveContacts(dt);
    }
    
    m_currentTime += dt;
    m_currentStep++;
    
    // Optimization: Don't update metrics every step as it requires GPU->CPU copy
    // updateMetrics();
    
    NVTX_POP();
    
    return true;
}

void SimulationEngine::stop() {
    m_shouldStop = true;
}

double SimulationEngine::getElapsedWallTime() const {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - m_startTime;
    return elapsed.count();
}

void SimulationEngine::loadGeometry() {
    std::cout << "[Engine] Loading geometry..." << std::endl;
    
    const auto& filePaths = m_config.getFilePaths();
    
    if (filePaths.toolGeometry.empty()) {
        std::cout << "[Engine] No tool geometry specified - using solver-generated defaults" << std::endl;
        return;  // Not fatal - FEM solver handles missing mesh gracefully
    }
    
    if (!std::filesystem::exists(filePaths.toolGeometry)) {
        std::cerr << "[Engine] WARNING: Tool geometry file not found: " << filePaths.toolGeometry << std::endl;
        std::cerr << "[Engine] Continuing without tool mesh" << std::endl;
        return;  // Warning instead of fatal error
    }
    
    std::cout << "[Engine] Tool geometry: " << filePaths.toolGeometry << std::endl;
}

void SimulationEngine::initializeSolvers() {
    std::cout << "[Engine] Initializing " << m_solvers.size() << " solver(s)..." << std::endl;
    
    for (auto& solver : m_solvers) {
        std::cout << "  - " << solver->getName() << "..." << std::endl;
        if (!solver->initialize(m_config)) {
            throw std::runtime_error("Failed to initialize solver: " + solver->getName());
        }
    }
    
    // Initialize machining strategy
    if (m_strategy) {
        std::cout << "  - " << m_strategy->getName() << " (strategy)..." << std::endl;
        m_strategy->initialize(m_config);
    }
    
    // Apply initial tool offset (Safe Start)
    if (m_config.getJson()["machining_parameters"].contains("initial_tool_position_mm")) {
         std::vector<double> initPos = m_config.getJson()["machining_parameters"]["initial_tool_position_mm"].get<std::vector<double>>();
         double offX = initPos[0]/1000.0; // mm to m
         double offY = initPos[1]/1000.0;
         double offZ = initPos[2]/1000.0;
         
         if (std::abs(offX) > 1e-9 || std::abs(offY) > 1e-9 || std::abs(offZ) > 1e-9) {
             std::cout << "[Engine] Applying initial tool offset: (" 
                       << offX << ", " << offY << ", " << offZ << ") m" << std::endl;
             
             for (auto& solver : m_solvers) {
                 // Dynamic cast to check if this is the FEM solver
                 FEMSolver* fem = dynamic_cast<FEMSolver*>(solver.get());
                 if (fem) {
                     fem->translateMesh(offX, offY, offZ);
                 }
                 // Also handling SPH if needed, but SPH is usually workpiece (stationary)
             }
         }
    }
    
    std::cout << "[Engine] All solvers initialized" << std::endl;
}

void SimulationEngine::runMainLoop() {
    const auto& simParams = m_config.getSimulation();
    const int totalSteps = simParams.numSteps;
    const int outputInterval = simParams.outputIntervalSteps;
    
    std::cout << "[Engine] Starting simulation (" << totalSteps << " steps)..." << std::endl;
    
    m_currentStep = 0;
    m_currentTime = 0.0;
    m_maxTemperature = m_config.getMachining().ambientTemperature; // Start at ambient
    
    // Initialize G-Code if provided
    std::string gcodePath = m_config.getFilePaths().gcodeFile;
    if (!gcodePath.empty()) {
        m_gcodeInterpreter = std::make_unique<GCodeInterpreter>();
        if (m_gcodeInterpreter->loadFile(gcodePath)) {
            std::cout << "[Engine] Using G-Code for toolpath control" << std::endl;
        } else {
            std::cerr << "[Engine] Failed to load G-Code: " << m_gcodeInterpreter->getLastError() << std::endl;
            // Fallback to manual
        }
    }
    
    while (m_currentStep < totalSteps && !m_shouldStop) {
        // Compute adaptive time step
        double dt = computeAdaptiveTimeStep();
        
        // Skip air-cutting if safe
        handleAirCutting(dt);
        
        // Step simulation
        step(dt);
        
        // Progress output & Metrics update
        if (m_currentStep % outputInterval == 0 || m_currentStep == 1) {
            // Update metrics only when needed for output
            updateMetrics();
            
            double percent = (100.0 * m_currentStep) / totalSteps;
            
            // Use newline instead of \r to ensure visibility in Docker logs
            std::cout << "  Step " << std::setw(7) << m_currentStep 
                      << " | " << std::fixed << std::setprecision(1) << percent << "%" 
                      << " | dt: " << std::scientific << std::setprecision(2) << dt
                      << " | Stress: " << std::scientific << std::setprecision(2) << m_maxStress / 1e6 << " MPa"
                      << " | Temp: " << std::fixed << std::setprecision(1) << m_maxTemperature << " C"
                      << std::endl;
            
            // Extract particles if available for live UI / VTK
            std::vector<SPHParticle> capturedParticles;
            for (const auto& solver : m_solvers) {
                if (solver->getName() == "SPHSolver") {
                    if (auto* sph = dynamic_cast<SPHSolver*>(solver.get())) {
                        capturedParticles = sph->getParticles();
                        break;
                    }
                }
            }

            // Export intermediate results (async - non-blocking)
            for (auto& exporter : m_exporters) {
                // Capture by value for thread safety
                int capturedStep = m_currentStep;
                double capturedTime = m_currentTime;
                double capturedStress = m_maxStress;
                double capturedTemp = m_maxTemperature;
                Mesh capturedMesh = m_toolMesh; // Copy mesh data
                std::vector<SPHParticle> capturedPts = capturedParticles;
                IExporter* exporterPtr = exporter.get();
                
                m_asyncExporter->enqueue([exporterPtr, capturedStep, capturedTime, capturedMesh, capturedPts, capturedStress, capturedTemp]() {
                    exporterPtr->exportStep(capturedStep, capturedTime, capturedMesh);
                    exporterPtr->exportMetrics(capturedStep, capturedTime, capturedStress, capturedTemp);
                    if (!capturedPts.empty()) {
                        exporterPtr->exportParticles(capturedStep, capturedTime, capturedPts);
                    }
                }, capturedStep, capturedTime);
            }
            
            // Call step callback
            if (m_stepCallback) {
                m_stepCallback(m_currentStep, m_currentTime, m_maxStress, m_maxTemperature);
            }
        }
    }
    
    std::cout << std::endl;  // New line after progress
}

void SimulationEngine::exportResults() {
    std::cout << "[Engine] Exporting results..." << std::endl;
    
    // Create output directory
    const auto& outputDir = m_config.getFilePaths().outputDirectory;
    std::filesystem::create_directories(outputDir);
    
    // Export final state from all exporters
    for (auto& exporter : m_exporters) {
        std::cout << "  - " << exporter->getName() << "..." << std::endl;
        exporter->exportFinal(m_config, m_toolMesh);
    }
    
    std::cout << "[Engine] Results saved to: " << outputDir << std::endl;
}

void SimulationEngine::updateMetrics() {
    // Collect metrics from solvers that provide them
    for (const auto& solver : m_solvers) {
        auto* metrics = dynamic_cast<IMetricsProvider*>(solver.get());
        if (metrics) {
            metrics->syncMetrics();
            m_maxStress = std::max(m_maxStress, metrics->getMaxStress());
            m_maxTemperature = std::max(m_maxTemperature, metrics->getMaxTemperature());
        }
    }
}

double SimulationEngine::computeAdaptiveTimeStep() const {
    const auto& simParams = m_config.getSimulation();
    double dt = simParams.timeStepDuration;
    
    // Get stable time step from all solvers, use minimum
    for (const auto& solver : m_solvers) {
        double solverDt = solver->getStableTimeStep();
        if (solverDt > 0) {
            dt = std::min(dt, solverDt);
        }
    }
    
    // Clamp to allowed range
    dt = std::max(simParams.minTimeStep, std::min(simParams.maxTimeStep, dt));
    
    return dt;
}

double SimulationEngine::computeAirGap() const {
    if (m_solvers.size() < 2) return 0.0;
    
    double minX1, minY1, minZ1, maxX1, maxY1, maxZ1;
    double minX2, minY2, minZ2, maxX2, maxY2, maxZ2;
    
    // Assuming 0 is SPH (workpiece) and 1 is FEM (tool)
    m_solvers[0]->getBounds(minX1, minY1, minZ1, maxX1, maxY1, maxZ1);
    m_solvers[1]->getBounds(minX2, minY2, minZ2, maxX2, maxY2, maxZ2);
    
    // Special case: tool starting far away
    double dx = std::max(0.0, std::max(minX1 - maxX2, minX2 - maxX1));
    double dy = std::max(0.0, std::max(minY1 - maxY2, minY2 - maxY1));
    double dz = std::max(0.0, std::max(minZ1 - maxZ2, minZ2 - maxZ1));
    
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void SimulationEngine::handleAirCutting(double& dt) {
    if (!m_contactSolver || m_contactSolver->getContactCount() > 0) return;
    
    double gap = computeAirGap();
    
    // If gap is more than 0.5mm, we can aggressively skip
    if (gap > 0.0005) {
        // Fast-forward: use 100x larger time step, but stay within gap
        // Velocity of tool from strategy
        static double lastSkipTime = 0;
        if (m_currentTime - lastSkipTime > 0.1) {
             std::cout << "[Engine] Air-cutting detected (gap: " << gap*1000.0 << " mm). Accelerating..." << std::endl;
             lastSkipTime = m_currentTime;
        }
        
        // We increase dt, but no more than what would close half the gap
        // Assuming ~10m/s max tool speed, dt=0.0001 is safe for 1mm gap
        dt = std::min(0.0001, gap / 10.0); 
    }
}

} // namespace edgepredict
