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
#include "CFDSolverGPU.cuh"
#include "ToolCoatingModel.cuh"
#include "OptimizationManager.h"
#include "SurfaceRoughnessPredictor.h"
#include "ResidualStressAnalyzer.h"
#include "IMachiningStrategy.h"
#include "VTKExporter.h"
#include "StdoutIPCExporter.h"
#include "CudaUtils.cuh"

#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>

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

/**
 * @brief Generate a parametric tool mesh when no STL/STEP file is available.
 *
 * Creates a triangulated surface mesh appropriate for the machining type:
 *   - Turning: rectangular prism insert (CNMG-like)
 *   - Drilling: cone-tipped cylinder
 *   - Milling:  flat-end cylinder
 *
 * Target: ~1000 nodes for accurate contact detection.
 */
void generateParametricToolMesh(Mesh& mesh, const Config& config) {
    mesh.clear();

    const double PI = 3.14159265358979323846;
    MachiningType type = config.getMachiningType();

    auto addNode = [&](double x, double y, double z) -> int {
        int idx = static_cast<int>(mesh.nodes.size());
        FEMNode node;
        node.position = Vec3(x, y, z);
        node.originalPosition = Vec3(x, y, z);
        mesh.nodes.push_back(node);
        return idx;
    };

    auto addTri = [&](int a, int b, int c) {
        Triangle tri;
        tri.indices[0] = a;
        tri.indices[1] = b;
        tri.indices[2] = c;
        // Normal computed later by GeometryLoader::computeNormals pattern
        const Vec3& v0 = mesh.nodes[a].position;
        const Vec3& v1 = mesh.nodes[b].position;
        const Vec3& v2 = mesh.nodes[c].position;
        Vec3 e1 = v1 - v0;
        Vec3 e2 = v2 - v0;
        tri.normal = e1.cross(e2).normalized();
        mesh.triangles.push_back(tri);
    };

    if (type == MachiningType::TURNING) {
        // Rectangular prism insert: 12.7mm IC x 4.76mm thick (CNMG 120408)
        // All dimensions in metres (engine units)
        double halfW  = 0.0127 / 2.0;   // 12.7 mm inscribed circle -> half-width
        double halfH  = 0.0127 / 2.0;
        double thick  = 0.00476;         // 4.76 mm thickness
        // Subdivide each face into NxN quads for ~1000 total nodes
        int N = 8; // 8x8 grid per face, 6 faces => ~384 quads => ~400 nodes (deduplicated ~300)
        // Actually build a subdivided box
        // We'll create nodes on a 3D grid covering the insert and triangulate the surface

        // Generate surface nodes for a box [-halfW..halfW] x [-halfH..halfH] x [0..thick]
        // 6 faces, each subdivided NxN
        auto boxFace = [&](Vec3 origin, Vec3 u, Vec3 v, int nu, int nv) {
            int startIdx = static_cast<int>(mesh.nodes.size());
            for (int j = 0; j <= nv; ++j) {
                for (int i = 0; i <= nu; ++i) {
                    double s = static_cast<double>(i) / nu;
                    double t = static_cast<double>(j) / nv;
                    Vec3 p = origin + u * s + v * t;
                    addNode(p.x, p.y, p.z);
                }
            }
            // Triangulate
            for (int j = 0; j < nv; ++j) {
                for (int i = 0; i < nu; ++i) {
                    int a = startIdx + j * (nu + 1) + i;
                    int b = a + 1;
                    int c = a + (nu + 1);
                    int d = c + 1;
                    addTri(a, b, d);
                    addTri(a, d, c);
                }
            }
        };

        Vec3 lo(-halfW, -halfH, 0);
        Vec3 dx(2*halfW, 0, 0), dy(0, 2*halfH, 0), dz(0, 0, thick);
        boxFace(lo, dx, dy, N, N);                                   // bottom (z=0)
        boxFace(lo + dz, dx, dy, N, N);                              // top (z=thick)
        boxFace(lo, dx, dz, N, N);                                   // front (y=-halfH)
        boxFace(Vec3(-halfW, halfH, 0), dx, dz, N, N);              // back  (y=+halfH)
        boxFace(lo, dy, dz, N, N);                                   // left  (x=-halfW)
        boxFace(Vec3(halfW, -halfH, 0), dy, dz, N, N);              // right (x=+halfW)

    } else {
        // Drilling / Milling: cylinder (with optional cone tip for drilling)
        double radius = 0.005;   // 10mm diameter / 2
        double length = 0.025;   // 25mm flute length
        int nCirc = 32;          // circumferential segments
        int nAxial = 20;         // axial segments
        bool hasCone = (type == MachiningType::DRILLING);
        double coneHeight = hasCone ? (radius / std::tan(59.0 * PI / 180.0)) : 0; // 118° point angle

        // Cylinder body nodes
        int bodyStart = static_cast<int>(mesh.nodes.size());
        for (int j = 0; j <= nAxial; ++j) {
            double z = -static_cast<double>(j) / nAxial * length;
            for (int i = 0; i < nCirc; ++i) {
                double theta = 2.0 * PI * i / nCirc;
                addNode(radius * std::cos(theta), radius * std::sin(theta), z);
            }
        }
        // Cylinder body triangles
        for (int j = 0; j < nAxial; ++j) {
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                int a = bodyStart + j * nCirc + i;
                int b = bodyStart + j * nCirc + i2;
                int c = bodyStart + (j + 1) * nCirc + i;
                int d = bodyStart + (j + 1) * nCirc + i2;
                addTri(a, b, d);
                addTri(a, d, c);
            }
        }

        // Top cap (z = 0) - flat end for milling, cone for drilling
        if (hasCone) {
            // Cone tip: fan from apex to first ring
            int apex = addNode(0, 0, coneHeight);
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                addTri(apex, bodyStart + i, bodyStart + i2);
            }
        } else {
            // Flat cap: fan from center to first ring
            int center = addNode(0, 0, 0);
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                addTri(center, bodyStart + i2, bodyStart + i);
            }
        }

        // Bottom cap (z = -length) - closed
        int bottomRingStart = bodyStart + nAxial * nCirc;
        int bottomCenter = addNode(0, 0, -length);
        for (int i = 0; i < nCirc; ++i) {
            int i2 = (i + 1) % nCirc;
            addTri(bottomCenter, bottomRingStart + i, bottomRingStart + i2);
        }
    }

    std::cout << "[Main] Generated parametric " 
              << (type == MachiningType::TURNING ? "insert" : 
                  type == MachiningType::DRILLING ? "drill" : "endmill")
              << " mesh: " << mesh.nodeCount() << " nodes, " 
              << mesh.triangleCount() << " triangles" << std::endl;
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
        
        // Fallback: generate parametric tool mesh if no geometry was loaded
        if (femSolver->getNodeCount() == 0) {
            std::cout << "[Main] No tool mesh loaded — generating parametric tool geometry..." << std::endl;
            generateParametricToolMesh(toolMesh, config);
            femSolver->initializeFromMesh(toolMesh);
        }
        
        // Create and register contact solver
        ContactSolver contactSolver;
        ContactConfig contactConfig;
        contactConfig.contactRadius = config.getSPH().smoothingRadius;
        contactConfig.contactStiffness = 1e7;
        contactConfig.frictionCoefficient = 0.3;
        contactSolver.initialize(sphSolver.get(), femSolver.get(), contactConfig);
        
        // Advanced Physics: Tool Coating & Wear Model (Integration of Orphan)
        auto coatingModel = std::make_unique<ToolCoatingModel>();
        coatingModel->initialize(femSolver->getNodeCount());
        // Configure with TiAlN/TiN multi-layer coating
        coatingModel->addLayer("TiAlN", 0.003e-3, 3000.0); // 3μm
        coatingModel->addLayer("TiN", 0.001e-3, 2400.0);   // 1μm
        
        contactSolver.setToolCoatingModel(coatingModel.get());
        engine.setContactSolver(&contactSolver);
        
        // Create CFD solver if enabled in config
        std::unique_ptr<CFDSolverGPU> cfdSolver;
        if (config.getCFD().enabled) {
            try {
                cfdSolver = std::make_unique<CFDSolverGPU>();
                cfdSolver->initialize(config);
                engine.setCFDSolver(cfdSolver.get());
                std::cout << "[Main] CFD coolant solver enabled (" 
                          << config.getCFD().coolantType << ")" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[Main] CFD initialization failed: " << e.what() 
                          << " — continuing without coolant simulation" << std::endl;
                cfdSolver.reset();
            }
        }
        
        // Create and register machining strategy (critical: enables tool/workpiece kinematics)
        auto strategy = MachiningStrategyFactory::createFromConfig(config);
        if (strategy) {
            // Connect solvers to strategy for force/kinematics coupling
            strategy->connectSolvers(sphSolver.get(), femSolver.get(), &contactSolver, cfdSolver.get());
            engine.setStrategy(std::move(strategy));
        }
        
        // Optimization & Analytics: (Integration of Orphans)
        auto optMgr = std::make_unique<OptimizationManager>();
        optMgr->initialize(config);
        engine.setOptimizationManager(std::move(optMgr));
        
        engine.setRoughnessPredictor(std::make_unique<SurfaceRoughnessPredictor>());
        engine.setStressAnalyzer(std::make_unique<ResidualStressAnalyzer>());
        
        // Create exporters
        auto vtkExporter = std::make_unique<VTKExporter>(filePaths.outputDirectory);
        
        // Register solvers and exporters with engine
        engine.addSolver(std::move(sphSolver));
        engine.addSolver(std::move(femSolver));
        if (cfdSolver) {
            engine.addSolver(std::move(cfdSolver));
        }
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
