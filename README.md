# Splat Color Grade

A LichtFeld Studio plugin for color grading Gaussian splats: adjust saturation, hue, exposure and more for selected Gaussians or a splat, with a live preview in the viewport


## Usage

1. Select Gaussians with the usual selection tools (brush, rectangle, polygon)
   To grade whole splats instead, uncheck *Selected Gaussians only* and select splat nodes in the scene tree
2. Open the *Color Grading* tab and adjust the sliders, or type exact values into the fields next to them
3. Click *Commit* to keep the change, or *Discard* to revert to the original colors

The selection is frozen while a preview is active. Commit or discard first, then select a new area.

## Notes

- Only the base color (SH degree 0) is changed. View-dependent color (SH degrees 1–3) stays as it is, so strong hue shifts may still show traces of the original color from some angles
- Tested with LichtFeld Studio 0.5.5 (nightly)
