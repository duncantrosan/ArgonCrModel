
"""
Static image (no animation): hemispherical argon plasma sitting on a flat
ceramic disc, purple + translucent, with
 
  * large arrows  : flux to the ceramic (flat face, Bohm/wall loss)
  * small arrows  : ambipolar diffusion out through the open curved face
 
Render a single frame:
    manim -s -qh plasma_diffusion.py PlasmaDiffusion
 
Options:  -t   transparent background
          -r 2400,1600   custom pixel size
Output lands in media/images/plasma_diffusion/PlasmaDiffusion*.png
"""
import numpy as np
from manim import *
 
# ---- palette --------------------------------------------------------------
BG          = "#07070E"
PLASMA_OUT  = "#B48CFF"   # outer shell
PLASMA_CORE = "#DCC0FF"   # bright core
CERAMIC     = "#D9D4C7"
ARROW_WALL  = "#FFC857"
ARROW_OUT   = "#7FDBFF"
 
 
class PlasmaDiffusion(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=62 * DEGREES, theta=-55 * DEGREES)
 
        R  = 2.0     # plasma radius
        Rc = 3.3     # ceramic radius
        hc = 0.25    # ceramic thickness
 
        # ---- ceramic: flat cylinder, top face at z = 0 -------------------
        ceramic = Cylinder(
            radius=Rc, height=hc, direction=OUT, resolution=(2, 64),
            checkerboard_colors=False, fill_color=CERAMIC, fill_opacity=1,
            stroke_width=0,
        )
        ceramic.set_fill(CERAMIC, opacity=1).set_stroke(width=0)
        ceramic.shift(-hc / 2 * OUT)
 
        # ---- plasma: two nested translucent hemispheres (emissive look) --
        def hemisphere(r, color, opacity, res=(48, 16)):
            s = Surface(
                lambda u, v: r * np.array([np.cos(u) * np.sin(v),
                                           np.sin(u) * np.sin(v),
                                           np.cos(v)]),
                u_range=[0, TAU], v_range=[0, PI / 2], resolution=res,
                checkerboard_colors=False, fill_color=color,
                fill_opacity=opacity, stroke_width=0,
            )
            s.set_fill(color, opacity=opacity).set_stroke(width=0)
            return s
 
        plasma = VGroup(
            hemisphere(R, PLASMA_OUT, 0.26),
            hemisphere(0.72 * R, PLASMA_CORE, 0.30, res=(32, 12)),
        )
 
        # ---- large arrows: flux to the ceramic (flat face) ---------------
        wall_arrows = VGroup()
        pts = [(0.0, 0.0)] + [(0.95 * np.cos(a), 0.95 * np.sin(a))
                              for a in np.linspace(0, TAU, 6, endpoint=False)]
        for x, y in pts:
            wall_arrows.add(Arrow3D(
                np.array([x, y, 0.4]), np.array([x, y, 0.06]),
                thickness=0.015, height=0.12, base_radius=0.12,
                color=ARROW_WALL,
            ))
 
        # ---- small arrows: diffusion out through the curved face ---------
        out_arrows = VGroup()
        rings = [(25, 4, 45), (55, 8, 0), (80, 8, 22.5)]   # (polar deg, n, az offset deg)
        for v_deg, n_az, az0 in rings:
            v = v_deg * DEGREES
            for a in np.linspace(0, TAU, n_az, endpoint=False) + az0 * DEGREES:
                n_hat = np.array([np.cos(a) * np.sin(v),
                                  np.sin(a) * np.sin(v),
                                  np.cos(v)])
                out_arrows.add(Arrow3D(
                    0.98 * R * n_hat, 1.32 * R * n_hat,
                    thickness=0.014, height=0.14, base_radius=0.05,
                    color=ARROW_OUT,
                ))
        out_arrows.add(Arrow3D(                                 # apex
            np.array([0, 0, 0.98 * R]), np.array([0, 0, 1.32 * R]),
            thickness=0.014, height=0.14, base_radius=0.05, color=ARROW_OUT,
        ))
 
        self.add(ceramic, plasma, wall_arrows, out_arrows)
 
        # ---- 2D overlays (fixed in frame) --------------------------------
        def legend_row(color, tex, width):
            arr = Arrow(LEFT * 0.45, RIGHT * 0.45, buff=0, color=color,
                stroke_width=width, max_tip_length_to_length_ratio=0.3)
            return VGroup(arr, Tex(tex, font_size=30)).arrange(RIGHT, buff=0.2)

        def legend_row(color, markup, width):
            arr = Arrow(LEFT * 0.45, RIGHT * 0.45, buff=0, color=color,
                        stroke_width=width, max_tip_length_to_length_ratio=0.3)
            return VGroup(arr, MarkupText(markup, font_size=26)).arrange(RIGHT, buff=0.2)

        legend = VGroup(
            legend_row(ARROW_WALL,
                "flux to ceramic (flat face, Bohm)   "
                "ν<sub>ceramic</sub> = h<sub>ℓ</sub> u<sub>B</sub> · A/V", 6),
            legend_row(ARROW_OUT,
                "ambipolar diffusion out   "
                "ν<sub>D</sub> = D<sub>a</sub>/Λ²", 3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(DL, buff=0.4)
        lab_plasma  = Text("Ar plasma", font_size=30, color=PLASMA_CORE).to_edge(UP, buff=0.6).shift(RIGHT * 2.5)
        lab_ceramic = Text("ceramic",   font_size=30, color=CERAMIC).to_edge(RIGHT, buff=0.4).shift(DOWN * 1.3)
 
        self.add_fixed_in_frame_mobjects(legend, lab_plasma, lab_ceramic)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt, matplotlib.image as mpimg
    with tempconfig({"save_last_frame": True, "quality": "high_quality",
                     "format": "png"}):
        scene = PlasmaDiffusion()
        scene.render()
    img = mpimg.imread(scene.renderer.file_writer.image_file_path)
    plt.figure(figsize=(12, 6.75)); plt.imshow(img); plt.axis("off"); plt.show()