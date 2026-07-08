# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:05:47 2026

@author: dptro
"""

"""
Radiation Trapping / Escape-Cone Geometry
------------------------------------------
Manim Community Edition script.

Scene:
  - A flat "wall" representing the solid-air interface (z = 0 plane, solid below).
  - A translucent hemisphere of radius R sitting on the wall (z >= 0), representing
    the space of possible ray directions from the origin O.
  - Vector 1 (blue):  O -> P, where P is the point on the hemisphere at spherical
                       coordinates (r, theta, phi)  [theta measured from +z, phi from +x].
  - Vector 2 (red):   P -> Q, leaving the hemisphere surface at P, with its OWN
                       local spherical angles (theta', phi') measured relative to
                       the outward normal at P (which is just the direction of
                       Vector 1, since P sits on a sphere centered at O).
                       This is exactly the local escape-cone construction used in
                       radiation-trapping diagrams (ray hits interface, then leaves
                       at some angle relative to the local normal).

Run with, e.g.:
    manim -pqh radiation_trapping.py RadiationTrapping

Tweak the parameters in the CONFIG section below to match your figure.
"""

from manim import *
import numpy as np

# ----------------------------- CONFIG -----------------------------------
R = 3.0                 # hemisphere / vector-1 radius
THETA = 55 * DEGREES    # polar angle of vector 1, from +z axis
PHI = 40 * DEGREES      # azimuthal angle of vector 1, from +x axis

THETA_P = 35 * DEGREES  # local polar angle of vector 2, from the outward normal at P
PHI_P = 130 * DEGREES   # local azimuthal angle of vector 2, around that normal

VEC2_LEN = 1.8          # length of the second (escape) vector
WALL_HALF = R * 1.6     # half-width of the solid wall slab
ARC_R_GLOBAL = 0.9      # radius used to draw the theta/phi angle arcs at the origin
ARC_R_LOCAL = 0.55      # radius used to draw the theta'/phi' angle arcs at P
# --------------------------------------------------------------------------


def sph_to_cart(r, theta, phi):
    """Standard physics convention: theta from +z, phi from +x in xy-plane."""
    return np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ])


def local_frame(normal):
    """Build an orthonormal (e1, e2, e3=normal) frame given a unit normal vector."""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0, 0, 1]) if abs(normal[2]) < 0.9 else np.array([1, 0, 0])
    e1 = np.cross(helper, normal)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2, normal


class RadiationTrapping(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES, distance=10)

        # ---------------- axes ----------------
        axes = ThreeDAxes(
            x_range=[-WALL_HALF, WALL_HALF, 1],
            y_range=[-WALL_HALF, WALL_HALF, 1],
            z_range=[0, R * 1.3, 1],
            x_length=8, y_length=8, z_length=5,
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")

        # ---------------- solid wall (interface) ----------------
        wall = Prism(dimensions=[2 * WALL_HALF, 2 * WALL_HALF, 0.25])
        wall.set_fill(GREY, opacity=0.6)
        wall.set_stroke(GREY_D, width=1)
        wall.shift(IN * 0.125)  # top face sits exactly at z = 0

        # ---------------- hemisphere (direction sphere) ----------------
        hemisphere = Surface(
            lambda u, v: sph_to_cart(R, u, v),
            u_range=[0, PI / 2],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.15,
            stroke_opacity=0.15,
        )
        hemisphere.set_fill(BLUE_E)
        hemisphere.set_stroke(BLUE_E, width=0.5)

        origin_dot = Dot3D(ORIGIN, color=WHITE, radius=0.06)

        # ---------------- Vector 1: O -> P (r, theta, phi) ----------------
        P = sph_to_cart(R, THETA, PHI)
        vec1 = Arrow3D(start=ORIGIN, end=P, color=BLUE, thickness=0.02, height=0.25)
        P_dot = Dot3D(P, color=BLUE, radius=0.06)

        # arc for theta (in the plane containing z-axis and vector 1)
        theta_arc_pts = [
            sph_to_cart(ARC_R_GLOBAL, t, PHI) for t in np.linspace(0, THETA, 30)
        ]
        theta_arc = VMobject(color=YELLOW).set_points_as_corners(theta_arc_pts)

        # arc for phi (in the xy-plane, from +x axis to phi)
        phi_arc_pts = [
            np.array([ARC_R_GLOBAL * np.cos(p), ARC_R_GLOBAL * np.sin(p), 0])
            for p in np.linspace(0, PHI, 30)
        ]
        phi_arc = VMobject(color=GREEN).set_points_as_corners(phi_arc_pts)

        theta_label = MathTex(r"\theta", color=YELLOW).scale(0.7)
        theta_label.move_to(sph_to_cart(ARC_R_GLOBAL + 0.35, THETA / 2, PHI))

        phi_label = MathTex(r"\phi", color=GREEN).scale(0.7)
        phi_label.move_to(
            np.array([
                (ARC_R_GLOBAL + 0.35) * np.cos(PHI / 2),
                (ARC_R_GLOBAL + 0.35) * np.sin(PHI / 2),
                0,
            ])
        )

        r_label = MathTex("r", color=BLUE).scale(0.7)
        r_label.move_to(P / 2 + np.array([0.2, 0.2, 0]))

        # ---------------- Vector 2: P -> Q, local angles (theta', phi') ----------------
        e1, e2, normal = local_frame(P)  # normal == direction of vector 1
        local_dir = (
            np.sin(THETA_P) * np.cos(PHI_P) * e1
            + np.sin(THETA_P) * np.sin(PHI_P) * e2
            + np.cos(THETA_P) * normal
        )
        Q = P + VEC2_LEN * local_dir
        vec2 = Arrow3D(start=P, end=Q, color=RED, thickness=0.02, height=0.25)

        # local dashed normal at P, to serve as the theta' reference axis
        normal_ref = DashedLine(
            P, P + ARC_R_LOCAL * 1.6 * normal, color=WHITE, stroke_width=2
        )

        # local theta' arc (plane containing normal and local_dir)
        # build via rotating 'normal' toward the projection of local_dir in the
        # (e1,e2,normal) frame -- i.e. sweep from normal to local_dir directly,
        # since both already lie in that plane.
        theta_p_arc_pts = []
        proj_dir = np.sin(1.0) * 0  # unused, placeholder for clarity
        in_plane_horiz = np.sin(THETA_P) * np.cos(PHI_P) * e1 + np.sin(THETA_P) * np.sin(PHI_P) * e2
        horiz_norm = np.linalg.norm(in_plane_horiz)
        horiz_unit = in_plane_horiz / horiz_norm if horiz_norm > 1e-6 else e1
        for t in np.linspace(0, THETA_P, 24):
            pt = P + ARC_R_LOCAL * (np.sin(t) * horiz_unit + np.cos(t) * normal)
            theta_p_arc_pts.append(pt)
        theta_p_arc = VMobject(color=ORANGE).set_points_as_corners(theta_p_arc_pts)

        # local phi' arc (plane spanned by e1,e2 at P, small circle around normal)
        phi_p_arc_pts = [
            P + (ARC_R_LOCAL * 0.6) * (np.cos(p) * e1 + np.sin(p) * e2)
            for p in np.linspace(0, PHI_P, 30)
        ]
        phi_p_arc = VMobject(color=PURPLE).set_points_as_corners(phi_p_arc_pts)

        theta_p_label = MathTex(r"\theta'", color=ORANGE).scale(0.6)
        theta_p_label.move_to(
            P + ARC_R_LOCAL * 1.3 * (np.sin(THETA_P / 2) * horiz_unit + np.cos(THETA_P / 2) * normal)
        )

        phi_p_label = MathTex(r"\phi'", color=PURPLE).scale(0.6)
        phi_p_label.move_to(
            P + (ARC_R_LOCAL * 0.9) * (np.cos(PHI_P / 2) * e1 + np.sin(PHI_P / 2) * e2)
        )

        Q_dot = Dot3D(Q, color=RED, radius=0.05)

        # ---------------- legend (fixed to frame) ----------------
        legend = VGroup(
            MathTex(r"\text{Vector 1: } O \to P(r,\theta,\phi)", color=BLUE),
            MathTex(r"\text{Vector 2: escapes } P \text{ at } (\theta',\phi')", color=RED),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.55)
        legend.to_corner(UL)

        # ---------------- build up the scene ----------------
        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(wall), FadeIn(axes), FadeIn(axes_labels))
        self.play(Create(hemisphere), FadeIn(origin_dot))
        self.play(GrowArrow(vec1), FadeIn(P_dot))
        self.play(Create(theta_arc), Create(phi_arc), Write(theta_label), Write(phi_label), Write(r_label))
        self.wait(0.5)
        self.play(Create(normal_ref))
        self.play(GrowArrow(vec2), FadeIn(Q_dot))
        self.play(
            Create(theta_p_arc), Create(phi_p_arc),
            Write(theta_p_label), Write(phi_p_label),
        )

        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        self.wait(1)