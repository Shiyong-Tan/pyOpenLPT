# Narrower tracking side panel

The fixed width of the right-hand Tracking/VSC control tabs is reduced from
370 px to 320 px.  This returns 50 px to the image, plot, and execution-log
area while retaining a fixed control width, so switching between Run Tracking
and Check Tracking does not move the main visualization.

No control behavior, configuration value, numerical operation, or LPT output is
changed.

The view was instantiated with Qt's off-screen platform in the OpenLPT Python
environment.  The resulting tab widget reported matching 320 px minimum and
maximum widths.
