# Measured lens calibration

Use this when estimated field of view is insufficient, particularly near image
edges. This measures the lens; camera room positions/angles still need the camera
layout and T-pose calibration. A bad lens file can make tracking worse.

1. Print [the 25 mm checkerboard](review/simulation/checkerboard-25mm.svg) on A4
   landscape paper at 100% scale, without fit-to-page. Measure a square and mount
   the print on a rigid flat surface. It has 9x6 inner corners.
2. Collect 15-30 sharp JPG/PNG frames with the entire board visible. Move it across
   the center, edges and corners; vary distance and tilt in both directions. Keep
   the board flat. Repeated nearly identical frames are rejected.
3. Use frames from the **same native video capture mode used for tracking**, before
   the app's image-rotation correction. A phone photo mode can have different
   optics/cropping from its browser video mode. Preserve full frame dimensions.
   Keep zoom, stabilization, cropping and focus settings consistent. The app can
   detect an aspect-ratio change, but cannot detect every same-ratio crop/zoom change.
4. From the repository, run:

   ```powershell
   .venv\Scripts\python.exe scripts/calibrate_lens.py --images C:\path\to\camera-1-frames --output C:\path\to\camera-1-lens.json
   ```

   The tool requires at least 12 distinct board detections spread across the image,
   reserves every fourth view for validation, and rejects poor reprojection fits.
   It does not overwrite an existing calibration file. It supports ordinary
   pinhole/Brown lenses, not an OpenCV fisheye model.
5. Open **Camera layout**, click that camera's **Import lens…**, select its JSON,
   then **Apply layout** and **Save**. Repeat separately for each camera. Stop and
   restart tracking so the new profile takes effect, then recalibrate the rig.

**Clear lens** returns a camera to estimated FOV. Both camera-layout modes support
measured lenses. The saved focal lengths override the FOV field while present.

When the app reports camera disagreement, first check whether a camera moved,
whether the correct lens was assigned, and whether capture delays are correct.
Restore the camera layout or correct its settings, then use **Recalibrate & align**.
Simply clicking Recalibrate does not repair incorrect manually entered coordinates.

Measured calibration does not guarantee reliable tracking. The current rendered
acceptance result and remaining limitations are in the
[hardening report](review/simulation/HARDENING.md).
