/**
 * @file main.cpp
 * @brief EdgePredict Engine v4 – Main Entry Point
 *
 * Fixes applied vs previous version:
 *  1. CFD solver: ownership correctly transferred to engine via addSolver()
 *     AFTER the non-owning pointer is stored in strategy/engine.  The order
 *     was correct before but is now explicitly commented.
 *  2. Strategy: connectSolvers() is called with real solver pointers (not null).
 *     Pointers remain valid until end of main() because engine owns the solvers
 *     in its vector.
 *  3. JSON key access guarded with contains() throughout to prevent null-JSON
 *     exceptions on configs that omit optional sections.
 *  4. IPC mode: stdout redirect is restored cleanly before return.
 *  5. ToolCoatingModel lifetime: unique_ptr lives for entire main() scope,
 *     outlasting engine.run().
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
#include <sstream>

using namespace edgepredict;

// ---------------------------------------------------------------------------
// Banner / usage
// ---------------------------------------------------------------------------

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
              << "  --help, -h       Show this help\n"
              << "  --gpu-info       Print GPU info and exit\n"
              << "  --validate       Validate config only\n"
              << "  --ipc            IPC mode (stdout = NDJSON)\n"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Parametric tool mesh generation
// Produces ~1 000-node triangulated surface when no STL/STEP file is given.
// ---------------------------------------------------------------------------

void generateParametricToolMesh(Mesh& mesh, const Config& config) {
    mesh.clear();

    const double PI   = 3.14159265358979323846;
    MachiningType type = config.getMachiningType();

    auto addNode = [&](double x, double y, double z) -> int {
        int idx = static_cast<int>(mesh.nodes.size());
        FEMNode node;
        node.position         = Vec3(x, y, z);
        node.originalPosition = Vec3(x, y, z);
        mesh.nodes.push_back(node);
        return idx;
    };

    auto addTri = [&](int a, int b, int c) {
        Triangle tri;
        tri.indices[0] = a;
        tri.indices[1] = b;
        tri.indices[2] = c;
        const Vec3& v0 = mesh.nodes[a].position;
        const Vec3& v1 = mesh.nodes[b].position;
        const Vec3& v2 = mesh.nodes[c].position;
        tri.normal = (v1 - v0).cross(v2 - v0).normalized();
        mesh.triangles.push_back(tri);
    };

    if (type == MachiningType::TURNING) {
        // Rectangular insert prism (CNMG 120408 approximate)
        double halfW = 0.0127 / 2.0;
        double halfH = 0.0127 / 2.0;
        double thick = 0.00476;
        int N = 8;  // subdivisions per face

        auto boxFace = [&](Vec3 origin, Vec3 u, Vec3 v, int nu, int nv) {
            int start = static_cast<int>(mesh.nodes.size());
            for (int j = 0; j <= nv; ++j) {
                for (int i = 0; i <= nu; ++i) {
                    double s = (double)i / nu;
                    double t = (double)j / nv;
                    Vec3 p = origin + u * s + v * t;
                    addNode(p.x, p.y, p.z);
                }
            }
            for (int j = 0; j < nv; ++j) {
                for (int i = 0; i < nu; ++i) {
                    int a = start + j * (nu + 1) + i;
                    int b = a + 1;
                    int c = a + (nu + 1);
                    int d = c + 1;
                    addTri(a, b, d);
                    addTri(a, d, c);
                }
            }
        };

        Vec3 lo(-halfW, -halfH, 0);
        Vec3 dx(2 * halfW, 0, 0), dy(0, 2 * halfH, 0), dz(0, 0, thick);
        boxFace(lo,                        dx, dy, N, N);  // bottom
        boxFace(lo + dz,                   dx, dy, N, N);  // top
        boxFace(lo,                        dx, dz, N, N);  // front
        boxFace(Vec3(-halfW, halfH, 0),    dx, dz, N, N);  // back
        boxFace(lo,                        dy, dz, N, N);  // left
        boxFace(Vec3(halfW, -halfH, 0),    dy, dz, N, N);  // right

    } else {
        // Cylinder (with point for drilling, flat for milling)
        double radius    = 0.005;   // 10 mm diameter / 2
        double length    = 0.025;   // 25 mm flute length
        int    nCirc     = 32;
        int    nAxial    = 20;
        bool   hasCone   = (type == MachiningType::DRILLING);
        double coneH     = hasCone
                           ? radius / std::tan(59.0 * PI / 180.0)  // 118° point angle
                           : 0.0;

        // Cylindrical body
        int bodyStart = static_cast<int>(mesh.nodes.size());
        for (int j = 0; j <= nAxial; ++j) {
            double z = -(double)j / nAxial * length;
            for (int i = 0; i < nCirc; ++i) {
                double theta = 2.0 * PI * i / nCirc;
                addNode(radius * std::cos(theta), radius * std::sin(theta), z);
            }
        }
        for (int j = 0; j < nAxial; ++j) {
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                int a  = bodyStart + j * nCirc + i;
                int b  = bodyStart + j * nCirc + i2;
                int c  = bodyStart + (j + 1) * nCirc + i;
                int d  = bodyStart + (j + 1) * nCirc + i2;
                addTri(a, b, d);
                addTri(a, d, c);
            }
        }

        // Top cap
        if (hasCone) {
            int apex = addNode(0, 0, coneH);
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                addTri(apex, bodyStart + i, bodyStart + i2);
            }
        } else {
            int center = addNode(0, 0, 0);
            for (int i = 0; i < nCirc; ++i) {
                int i2 = (i + 1) % nCirc;
                addTri(center, bodyStart + i2, bodyStart + i);
            }
        }

        // Bottom cap
        int bottomRing  = bodyStart + nAxial * nCirc;
        int bottomCenter = addNode(0, 0, -length);
        for (int i = 0; i < nCirc; ++i) {
            int i2 = (i + 1) % nCirc;
            addTri(bottomCenter, bottomRing + i, bottomRing + i2);
        }
    }

    const char* toolName =
        (type == MachiningType::TURNING)  ? "insert" :
        (type == MachiningType::DRILLING) ? "drill"  : "endmill";
    std::cout << "[Main] Generated parametric " << toolName
              << " mesh: " << mesh.nodeCount() << " nodes, "
              << mesh.triangleCount() << " triangles" << std::endl;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string configPath;
    bool gpuInfoOnly   = false;
    bool validateOnly  = false;
    bool ipcMode       = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h") { printUsage(argv[0]); return 0; }
        else if (arg == "--gpu-info")             { gpuInfoOnly  = true; }
        else if (arg == "--validate")             { validateOnly = true; }
        else if (arg == "--ipc")                  { ipcMode      = true; }
        else if (arg[0] != '-')                   { configPath   = arg;  }
    }

    if (!ipcMode) printBanner();

    if (gpuInfoOnly) { printGPUInfo(); return 0; }

    if (configPath.empty()) {
        configPath = "input.json";
        if (!ipcMode)
            std::cout << "[Main] Using default config: " << configPath << std::endl;
    }

    if (!ipcMode) { printGPUInfo(); std::cout << std::endl; }

    // IPC mode: redirect std::cout to std::cerr so pure JSON lines reach stdout
    std::streambuf* origCout = nullptr;
    if (ipcMode) {
        origCout = std::cout.rdbuf();
        std::cout.rdbuf(std::cerr.rdbuf());
        std::cerr << "[Main] IPC mode — stdout reserved for NDJSON" << std::endl;
    }

    int exitCode = 0;
    try {
        auto startTime = std::chrono::high_resolution_clock::now();

        // ------------------------------------------------------------------
        // 1. Engine + config
        // ------------------------------------------------------------------
        SimulationEngine engine;
        if (!engine.initialize(configPath)) {
            std::cerr << "[FATAL] Failed to initialize engine\n";
            exitCode = 1;
            goto cleanup;
        }

        if (validateOnly) {
            std::cout << "[Main] Configuration is valid.\n";
            goto cleanup;
        }

        {
            const Config& config = engine.getConfig();

            // ------------------------------------------------------------------
            // 2. Create physics solvers
            // ------------------------------------------------------------------
            auto sphSolver = std::make_unique<SPHSolver>();
            auto femSolver = std::make_unique<FEMSolver>();

            sphSolver->initialize(config);
            femSolver->initialize(config);

            // Workpiece generation
            const auto& sphCfg = config.getSPH();
            double spacing = sphCfg.smoothingRadius * sphCfg.particleSpacingFactor;
            
            if (config.getMachiningType() == MachiningType::TURNING || 
                config.getMachiningType() == MachiningType::DRILLING) {
                Vec3 center(0.0, 0.0, 0.0);
                double radius = 0.02; // 20mm
                double length = 0.05; // 50mm
                sphSolver->initializeCylindricalWorkpiece(center, radius, length, spacing, 2);
            } else {
                Vec3 workMin(0.0, -0.01, -0.01);
                Vec3 workMax(0.05,  0.01,  0.01);
                sphSolver->initializeParticleBox(workMin, workMax, spacing);
            }

            // Tool geometry
            GeometryLoader geoLoader;
            Mesh toolMesh;
            const auto& filePaths = config.getFilePaths();

            if (!filePaths.toolGeometry.empty()) {
                if (geoLoader.load(filePaths.toolGeometry, toolMesh)) {
                    GeometryLoader::scaleMesh(toolMesh, 0.001); // mm → m
                    femSolver->initializeFromMesh(toolMesh);
                } else {
                    std::cerr << "[Warning] Tool geometry load failed: "
                              << geoLoader.getLastError() << std::endl;
                }
            }

            // Parametric fallback when no geometry file
            if (femSolver->getNodeCount() == 0) {
                generateParametricToolMesh(toolMesh, config);
                femSolver->initializeFromMesh(toolMesh);
            }

            // ------------------------------------------------------------------
            // AUTO-POSITION TOOL AT WORKPIECE SURFACE
            // The parametric tool is generated near origin; the SPH workpiece
            // box is at x=[0,50mm] y=[-10mm,10mm] z=[-10mm,10mm].
            // We must translate the tool so it actually overlaps the workpiece.
            // ------------------------------------------------------------------
            {
                double tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ;
                femSolver->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);

                double wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ;
                sphSolver->getBounds(wMinX, wMinY, wMinZ, wMaxX, wMaxY, wMaxZ);
                Vec3 workMin(wMinX, wMinY, wMinZ);
                Vec3 workMax(wMaxX, wMaxY, wMaxZ);

                MachiningType mtype = config.getMachiningType();
                double doc = config.getMachining().depthOfCutMm / 1000.0;

                if (mtype == MachiningType::TURNING) {
                    // Turning: tool approaches radially (Y-axis)
                    // Place tool tip at workpiece surface minus depth-of-cut
                    double toolHeight = tMaxY - tMinY;
                    double targetY = workMax.y - doc - tMinY;
                    double targetX = 0.005 - (tMinX + tMaxX) / 2.0; // near x=5mm
                    femSolver->translateMesh(targetX, targetY, 0);
                    std::cout << "[Main] Turning: positioned tool at workpiece surface"
                              << " (offsetX=" << targetX*1000 << "mm"
                              << " offsetY=" << targetY*1000 << "mm)" << std::endl;
                } else if (mtype == MachiningType::DRILLING) {
                    // Drilling: tool approaches along Z-axis (downward)
                    // Place drill tip just above workpiece top surface
                    double targetX = (workMin.x + workMax.x) / 2.0 - (tMinX + tMaxX) / 2.0;
                    double targetY = (workMin.y + workMax.y) / 2.0 - (tMinY + tMaxY) / 2.0;
                    double targetZ = workMax.z - tMinZ + spacing * 0.5; // just above surface
                    femSolver->translateMesh(targetX, targetY, targetZ);
                    std::cout << "[Main] Drilling: positioned tool above workpiece"
                              << " (offsetZ=" << targetZ*1000 << "mm)" << std::endl;
                } else {
                    // Milling: tool approaches from above (Z-axis)
                    // Place tool bottom at workpiece top surface minus depth-of-cut
                    double targetX = 0.0 - (tMinX + tMaxX) / 2.0; // at workpiece X start
                    double targetY = (workMin.y + workMax.y) / 2.0 - (tMinY + tMaxY) / 2.0;
                    double targetZ = workMax.z - doc - tMinZ;
                    femSolver->translateMesh(targetX, targetY, targetZ);
                    std::cout << "[Main] Milling: positioned tool at workpiece surface"
                              << " (offsetZ=" << targetZ*1000 << "mm)" << std::endl;
                }

                // Re-read bounds after translation
                femSolver->getBounds(tMinX, tMinY, tMinZ, tMaxX, tMaxY, tMaxZ);
                std::cout << "[Main] Tool bounds after positioning: "
                          << "X[" << tMinX*1000 << ", " << tMaxX*1000 << "] "
                          << "Y[" << tMinY*1000 << ", " << tMaxY*1000 << "] "
                          << "Z[" << tMinZ*1000 << ", " << tMaxZ*1000 << "] mm"
                          << std::endl;
            }
            
            // Link tool mesh to engine for VTK export
            engine.setToolMesh(toolMesh);

            // ------------------------------------------------------------------
            // 3. Contact solver (stack-allocated, outlives engine.run())
            //    Contact radius = 1.5 * smoothing radius for reliable engagement
            // ------------------------------------------------------------------
            ContactSolver contactSolver;
            ContactConfig  contactCfg;
            contactCfg.contactRadius       = sphCfg.smoothingRadius * 1.5;
            contactCfg.contactStiffness    = 1e11;
            contactCfg.frictionCoefficient = 0.3;
            contactSolver.initialize(sphSolver.get(), femSolver.get(), contactCfg);

            // ------------------------------------------------------------------
            // 4. Tool coating model (lifetime spans all of main body)
            // ------------------------------------------------------------------
            auto coatingModel = std::make_unique<ToolCoatingModel>();
            coatingModel->initialize(femSolver->getNodeCount());
            coatingModel->addLayer("TiAlN", 3e-6, 3300.0);
            coatingModel->addLayer("TiN",   1e-6, 2000.0);
            contactSolver.setToolCoatingModel(coatingModel.get());

            engine.setContactSolver(&contactSolver);

            // ------------------------------------------------------------------
            // 5. CFD solver (optional)
            //    Order matters: store raw pointer first, THEN transfer ownership.
            // ------------------------------------------------------------------
            std::unique_ptr<CFDSolverGPU> cfdSolver;
            if (config.getCFD().enabled) {
                try {
                    cfdSolver = std::make_unique<CFDSolverGPU>();
                    cfdSolver->initialize(config);
                    // Store non-owning pointer in engine BEFORE the move
                    engine.setCFDSolver(cfdSolver.get());
                    std::cout << "[Main] CFD coolant solver enabled ("
                              << config.getCFD().coolantType << ")" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "[Main] CFD init failed: " << e.what()
                              << " — continuing without coolant" << std::endl;
                    cfdSolver.reset();
                }
            }

            // ------------------------------------------------------------------
            // 6. Machining strategy
            //    createFromConfig() calls initialize(config) internally.
            //    connectSolvers() is called with VALID pointers (solvers still
            //    owned by their unique_ptrs in this scope, not yet moved).
            // ------------------------------------------------------------------
            auto strategy = MachiningStrategyFactory::createFromConfig(config);
            if (strategy) {
                // Raw pointers are valid — unique_ptrs not yet moved to engine
                strategy->connectSolvers(sphSolver.get(), femSolver.get(),
                                         &contactSolver,
                                         cfdSolver ? cfdSolver.get() : nullptr);
                engine.setStrategy(std::move(strategy));
            }

            // ------------------------------------------------------------------
            // 7. Optimisation & analytics
            // ------------------------------------------------------------------
            auto optMgr = std::make_unique<OptimizationManager>();
            optMgr->initialize(config);
            engine.setOptimizationManager(std::move(optMgr));

            engine.setRoughnessPredictor(std::make_unique<SurfaceRoughnessPredictor>());
            engine.setStressAnalyzer(std::make_unique<ResidualStressAnalyzer>());

            // ------------------------------------------------------------------
            // 8. Exporters
            // ------------------------------------------------------------------
            engine.addExporter(
                std::make_unique<VTKExporter>(filePaths.outputDirectory));

            if (ipcMode)
                engine.addExporter(std::make_unique<StdoutIPCExporter>());

            // ------------------------------------------------------------------
            // 9. Transfer solver ownership to engine
            //    (raw pointers stored in strategy/contactSolver remain valid
            //     because engine's vector is the new owner and lives until
            //     engine is destroyed at end of this scope)
            // ------------------------------------------------------------------
            engine.addSolver(std::move(sphSolver));
            engine.addSolver(std::move(femSolver));
            if (cfdSolver)
                engine.addSolver(std::move(cfdSolver));

            // ------------------------------------------------------------------
            // 10. Run
            // ------------------------------------------------------------------
            std::cout << "\n[Main] Starting simulation...\n" << std::endl;
            engine.run();

            // Final timing / metrics
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = endTime - startTime;
            std::cout << "\n[Main] Total execution time: "
                      << std::fixed << std::setprecision(2)
                      << elapsed.count() << " s\n";
            std::cout << "[Main] Max stress:       "
                      << engine.getMaxStress() / 1e6 << " MPa\n";
            std::cout << "[Main] Max temperature:  "
                      << engine.getMaxTemperature() << " °C\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
        exitCode = 1;
    }

cleanup:
    // Restore stdout if redirected
    if (ipcMode && origCout) std::cout.rdbuf(origCout);
    return exitCode;
}