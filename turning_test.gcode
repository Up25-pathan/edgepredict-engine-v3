; Turning Test (OD Roughing)
; Insert: Sandvik CNMG 12 04 08
; Operation: Longitudinal Turning

G21      ; Metric units
G90      ; Absolute positioning
G18      ; XZ Plane selection (Lathe)

; Setup
S800 M03 ; Start spindle CW at 800 RPM
G00 X60.0 Z5.0 ; Rapid to safe home

; Approach
G00 X48.0      ; Approach cut diameter (Radius 24mm)
G00 Z2.0       ; Approach Z face

; Cut
G01 Z-50.0 F0.2 ; Turn to Z-50 (Feed 0.2 mm/rev)

; Retract
G01 X55.0 F0.5  ; Retract radially
G00 Z5.0        ; Rapid return Z

M05             ; Spindle stop
M30             ; End program
