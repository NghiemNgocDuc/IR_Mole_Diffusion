# Movable 3D molecules with a color key

`viewer.html` contains an actual interactive 3D renderer. A PNG or SVG is a
still image and cannot rotate. Download the HTML from Unity and open it locally
in Chrome or Edge rather than opening the PNG or the OnDemand HTML preview.

## Update the existing prediction viewer

Upload `outputs/unity_3d_viewer_update.zip` to `~/diffusion` on Unity, then run:

```bash
cd ~/diffusion
unzip -o unity_3d_viewer_update.zip
module load conda/latest
conda run -n base --no-capture-output python make_viewer.py \
  --prediction_dir predictions/example_66285483
```

Download `predictions/example_66285483/viewer.html` and open it on your computer.
No GPU job or new training is needed. Open the newly downloaded HTML if an older
copy is still open in your browser.

- Drag to rotate in all directions; wheel or pinch to zoom.
- Use Move left/right/up/down to pan, or Auto-rotate to spin continuously.
- Use Full screen for a larger view and Reset view to recenter.
- Atoms are colored balls without element letters. The color key is directly
  below the model: gray carbon, blue nitrogen, red oxygen, pale hydrogen, and
  green fluorine, showing only elements present in the selected molecule.
- Click an atom to read its element and coordinates below the model. Measure
  distance draws a line and reports the distance below, without text overlays.
- Select any saved candidate/conformer and raw or final coordinates.
- The 2D drawing and SMILES are available in a collapsed sidebar section.

For saved predictions, the exporter preserves the actual saved model geometry.
It does not rebuild a conformer to make the prediction look more convincing.
Some molecular rings are naturally almost planar; rotation reveals their
geometry rather than artificially bending it.

To regenerate the optional static picture with its new outside color key:

```bash
conda run -n base --no-capture-output python plot_prediction.py \
  --prediction_dir predictions/example_66285483 --graph_index 2 \
  --output predictions/example_66285483/candidate3.png
```

## Generate a 3D conformer from a 2D molecular structure

If you have a known molecular graph encoded as SMILES, you can generate
plausible 3D conformers. For example, this standalone ethanol demonstration
needs only RDKit and NumPy, with no model checkpoint, IR spectrum, or GPU:

```bash
conda run -n base --no-capture-output python make_viewer.py \
  --smiles 'CCO' --output predictions/ethanol_generated.html
```

Download and open the generated HTML. Its display identifies the coordinates
as generated from the supplied structure. A 2D graph does not specify a unique
3D conformation; this mode gives generated estimates rather than experimental
coordinates or a result from the trained IR-to-3D model. A picture of a 2D
structure must first be converted to a correct graph/SMILES; this command does
not interpret image pixels. Unspecified stereochemistry is not determined by
the drawing or by this viewer.

The offline renderer, scripts, and guide are included in the update. The
manifest verifies their hashes; trained weights and datasets are not included.
