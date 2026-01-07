#ifndef TOOL_GENERATOR_H
#define TOOL_GENERATOR_H

#include "json.hpp"
#include <string>
#include <opencascade/TopoDS_Shape.hxx>

using json = nlohmann::json;

class ToolGenerator {
public:
    ToolGenerator();
    ~ToolGenerator();

    // Main entry point: Reads JSON specs -> Returns CAD Shape
    TopoDS_Shape generate_tool(const json& specs);

    // Saves the CAD shape to a .step file
    void export_step(const TopoDS_Shape& shape, const std::string& filename);

private:
    TopoDS_Shape generate_twist_drill(double diameter, double length, double helix_angle, double tip_angle);
    TopoDS_Shape generate_end_mill(double diameter, double length, int flutes);
};

#endif // TOOL_GENERATOR_H