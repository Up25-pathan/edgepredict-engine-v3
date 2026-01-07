#include "ToolGenerator.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <fstream>

// OpenCASCADE Includes
#include <opencascade/BRepPrimAPI_MakeCylinder.hxx>
#include <opencascade/BRepPrimAPI_MakeCone.hxx>
#include <opencascade/BRepAlgoAPI_Fuse.hxx>
#include <opencascade/STEPControl_Writer.hxx>
#include <opencascade/Interface_Static.hxx>
#include <opencascade/gp_Ax2.hxx>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ToolGenerator::ToolGenerator() {}
ToolGenerator::~ToolGenerator() {}

TopoDS_Shape ToolGenerator::generate_tool(const json& specs) {
    std::string type = specs.value("type", "drill");
    
    if (type == "drill") {
        double D = specs.value("diameter_mm", 10.0);
        double L = specs.value("length_mm", 100.0);
        double helix = specs.value("helix_angle_deg", 30.0);
        double tip = specs.value("tip_angle_deg", 118.0);
        return generate_twist_drill(D, L, helix, tip);
    } 
    else if (type == "end_mill") {
        double D = specs.value("diameter_mm", 10.0);
        double L = specs.value("length_mm", 50.0);
        int flutes = specs.value("flute_count", 4);
        return generate_end_mill(D, L, flutes);
    }
    else {
        throw std::runtime_error("Unknown tool type: " + type);
    }
}

TopoDS_Shape ToolGenerator::generate_twist_drill(double D, double L, double helix_angle, double tip_angle) {
    std::cout << "[ToolGenerator] Generating Twist Drill (D=" << D << "mm, L=" << L << "mm)..." << std::endl;

    // 1. Create Shank (Cylinder)
    // Axis aligned with Z
    gp_Ax2 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    TopoDS_Shape shank = BRepPrimAPI_MakeCylinder(axis, D/2.0, L).Shape();

    // 2. Create Tip (Cone)
    double radius = D / 2.0;
    double angle_rad = (tip_angle * M_PI / 180.0) / 2.0;
    // Height = Radius / tan(half_tip_angle)
    double height = radius / std::tan(angle_rad);
    
    // Attach tip at Z=0, pointing downwards (-Z)
    gp_Ax2 tip_axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, -1));
    TopoDS_Shape tip = BRepPrimAPI_MakeCone(tip_axis, radius, 0.0, height).Shape();

    // 3. Fuse Shank + Tip
    TopoDS_Shape drill = BRepAlgoAPI_Fuse(shank, tip).Shape();
    
    return drill;
}

TopoDS_Shape ToolGenerator::generate_end_mill(double D, double L, int flutes) {
    std::cout << "[ToolGenerator] Generating End Mill..." << std::endl;
    // Basic Flat End Mill = Cylinder
    return BRepPrimAPI_MakeCylinder(D/2.0, L).Shape();
}

void ToolGenerator::export_step(const TopoDS_Shape& shape, const std::string& filename) {
    STEPControl_Writer writer;
    Interface_Static::SetCVal("write.step.schema", "AP214");
    
    if (writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone) {
        throw std::runtime_error("Error transferring shape to STEP writer");
    }
    
    if (writer.Write(filename.c_str()) != IFSelect_RetDone) {
        throw std::runtime_error("Error writing STEP file: " + filename);
    }
    std::cout << "[ToolGenerator] Saved geometry to " << filename << std::endl;
}