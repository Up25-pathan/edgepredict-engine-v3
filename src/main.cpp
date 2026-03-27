/**
 * @file main.cpp
 * @brief EdgePredict Engine v4 - Main Entry Point
 * 
 * Clean architecture entry point with dependency injection.
 */

#include "SimulationEngine.h"
#include "Config.h"
#include "GeometryLoader.h"
#include "SPHSolver.cuh"
#include "FEMSolver.cuh"
#include "ContactSolver.cuh"
#include "IMachiningStrategy.h"
#include "VTKExporter.h"
#include "StdoutIPCExporter.h"
#include "CudaUtils.cuh"

#include <iostream>
#include <chrono>
#include <memory>

using namespace edgepredict;

void printBanner() {
    std::cout << R"(
  ______    _            _____              _ _      _   
 |  ____|  | |          |  __ \            | (_)    | |  
 | |__   __| | __ _  ___| |__) | __ ___  __| |_  ___| |_ 
 |  __| / _` |/ _` |/ _ \  ___/ '__/ _ \/ _` | |/ __| __|
 | |___| (_| | (_| |  __/ |   | | |  __/ (_| | | (__| |_ 
 |______\__,_|\__, |\___|_|   |_|  \___|\__,_|_|\___|\__|
               __/ |                                     
              |___/         Engine v4.0 (Clean Arch)     
    )" << std::endl;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <config.json> [options]\n"
              << "\nOptions:\n"
              << "  --help, -h       Show this help message\n"
              << "  --gpu-info       Print GPU information and exit\n"
              << "  --validate       Validate config only, don't run simulation\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line
    std::string configPath;
    bool gpuInfoOnly = false;
    bool validateOnly = false;
    bool ipcMode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--gpu-info") {
            gpuInfoOnly = true;
        } else if (arg == "--validate") {
            validateOnly = true;
        } else if (arg == "--ipc") {
            ipcMode = true;
        } else if (arg[0] != '-') {
            configPath = arg;
        }
    }
    
    if (!ipcMode) {
        printBanner();
    }
    
    // GPU info
    if (gpuInfoOnly) {
        if (!ipcMode) printGPUInfo();
        return 0;
    }
    
    // Check config path
    if (configPath.empty()) {
        // Default to input.json in current directory
        configPath = "input.json";
        if (!ipcMode) std::cout << "[Main] Using default config: " << configPath << std::endl;
    }
    
    // Print GPU info
    if (!ipcMode) {
        printGPUInfo();
        std::cout << std::endl;
    }
    
    // Silence stdout if IPC mode is active so JSON gets unhindered pipe
    std::streambuf* originalCoutBuffer = nullptr;
    if (ipcMode) {
        originalCoutBuffer = std::cout.rdbuf();
        std::cout.rdbuf(std::cerr.rdbuf()); // redirect normal cout to cerr
        std::cerr << "[Main] IPC Mode Active - Stdout reserved for JSON lines" << std::endl;
    }
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Initialize simulation engine
        SimulationEngine engine;
        
        if (!engine.initialize(configPath)) {
            std::cerr << "[FATAL] Failed to initialize simulation engine\n";
            return 1;
        }
        
        if (validateOnly) {
            std::cout << "[Main] Configuration is valid.\n";
            return 0;
        }
        
        // Get config reference
        const Config& config = engine.getConfig();
        
        // Create and register physics solvers
        auto sphSolver = std::make_unique<SPHSolver>();
        auto femSolver = std::make_unique<FEMSolver>();
        
        // Initialize solvers
        sphSolver->initialize(config);
        femSolver->initialize(config);
        
        // Create workpiece particle box (default if no workpiece geometry)
        Vec3 workpieceMin(0.0, -0.01, -0.01);
        Vec3 workpieceMax(0.05, 0.01, 0.01);
        double spacing = config.getSPH().smoothingRadius * config.getSPH().particleSpacingFactor;
        sphSolver->initializeParticleBox(workpieceMin, workpieceMax, spacing);
        
        // Load tool geometry
        GeometryLoader geoLoader;
        Mesh toolMesh;
        
        const auto& filePaths = config.getFilePaths();
        if (!filePaths.toolGeometry.empty()) {
            if (geoLoader.load(filePaths.toolGeometry, toolMesh)) {
                // Convert mm to meters if needed
                GeometryLoader::scaleMesh(toolMesh, 0.001);
                femSolver->initializeFromMesh(toolMesh);
            } else {
                std::cerr << "[Warning] Failed to load tool geometry: " 
                          << geoLoader.getLastError() << std::endl;
            }
        }
        
        // Create contact solver
        ContactSolver contactSolver;
        ContactConfig contactConfig;
        contactConfig.contactRadius = config.getSPH().smoothingRadius;
        contactConfig.contactStiffness = 1e7;
        contactConfig.frictionCoefficient = 0.3;
        contactSolver.initialize(sphSolver.get(), femSolver.get(), contactConfig);
        
        // Register contact solver with engine (critical: enables tool-workpiece interaction)
        engine.setContactSolver(&contactSolver);
        
        // Create and register machining strategy (critical: enables tool/workpiece kinematics)
        auto strategy = MachiningStrategyFactory::createFromConfig(config);
        if (strategy) {
            // Connect solvers to strategy for force/kinematics coupling
            strategy->connectSolvers(nullptr, nullptr, nullptr, nullptr);  // Will use engine solvers
            engine.setStrategy(std::move(strategy));
        }
        
        // Create exporters
        auto vtkExporter = std::make_unique<VTKExporter>(filePaths.outputDirectory);
        
        // Register solvers and exporters with engine
        engine.addSolver(std::move(sphSolver));
        engine.addSolver(std::move(femSolver));
        engine.addExporter(std::move(vtkExporter));
        
        if (ipcMode) {
            auto ipcExporter = std::make_unique<StdoutIPCExporter>();
            engine.addExporter(std::move(ipcExporter));
        }
        
        // Set progress callback
        engine.setStepCallback([](int step, double time, double stress, double temp) {
            (void)step; (void)time; (void)stress; (void)temp;
            // Progress is already printed by engine
        });
        
        // Run simulation
        std::cout << "\n[Main] Starting simulation...\n" << std::endl;
        engine.run();
        
        // Final timing
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        
        std::cout << "\n[Main] Total execution time: " << elapsed.count() << " seconds\n";
        std::cout << "[Main] Max stress: " << engine.getMaxStress() / 1e6 << " MPa\n";
        std::cout << "[Main] Max temperature: " << engine.getMaxTemperature() << " °C\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
        return 1;
    }
}
