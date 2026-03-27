; Sandvik DE10 Milling Test
; Tool: 10mm End Mill
; Material: Ti-6Al-4V
; Operation: Slot Milling

G21      ; Metric units
G90      ; Absolute positioning
G17      ; XY Plane selection

; Setup
S800 M03 ; Start spindle CW at 800 RPM
G00 Z10.0 ; Rapid to safe height

; Approach
G00 X-12.0 Y0.0 ; Rapid to start position (outside workpiece)
G00 Z1.0        ; Rapid to approach height

; Plunge
G01 Z-1.5 F50   ; Feed enter to cut depth (1.5mm)

; Cut
G01 X50.0 F200  ; Main cutting pass (Feed 200 mm/min)

; Retract
G00 Z10.0       ; Rapid retract
M05             ; Spindle stop
M30             ; End program
