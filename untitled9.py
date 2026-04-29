# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:21:04 2026

@author: najla
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="PX4 X500 UAV Simulation", layout="wide")

st.title("PX4 X500 UAV Simulation")
st.write(
    "A quadrotor model with 3D flight animation, hover-trim thrust, "
    "damped altitude and yaw control, and visual roll/pitch tilt."
)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def smooth_scalar(state, command, tau, dt):
    if tau <= 1e-6:
        return command
    alpha = 1.0 - np.exp(-dt / tau)
    return state + alpha * (command - state)


def smooth_angle(state, command, tau, dt):
    if tau <= 1e-6:
        return command
    alpha = 1.0 - np.exp(-dt / tau)
    return wrap_angle(state + alpha * wrap_angle(command - state))


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    r_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return r_z @ r_y @ r_x


def transform_points(points, center, roll, pitch, yaw):
    rotation = rotation_matrix(roll, pitch, yaw)
    return (rotation @ points.T).T + center


def transform_segments(segments, center, roll, pitch, yaw):
    x_vals = []
    y_vals = []
    z_vals = []

    for segment in segments:
        pts = transform_points(segment, center, roll, pitch, yaw)
        x_vals.extend([float(v) for v in pts[:, 0]])
        y_vals.extend([float(v) for v in pts[:, 1]])
        z_vals.extend([float(v) for v in pts[:, 2]])
        x_vals.append(None)
        y_vals.append(None)
        z_vals.append(None)

    if x_vals:
        x_vals.pop()
        y_vals.pop()
        z_vals.pop()

    return x_vals, y_vals, z_vals


def quad_points_3d(xc, yc, zc, roll, pitch, yaw, arm=0.55):
    center = np.array([xc, yc, zc], dtype=float)

    arm1 = transform_points(
        np.array([[-arm, 0.0, 0.0], [arm, 0.0, 0.0]], dtype=float),
        center,
        roll,
        pitch,
        yaw,
    )
    arm2 = transform_points(
        np.array([[0.0, -arm, 0.0], [0.0, arm, 0.0]], dtype=float),
        center,
        roll,
        pitch,
        yaw,
    )
    motors = transform_points(
        np.array(
            [
                [arm, 0.0, 0.0],
                [-arm, 0.0, 0.0],
                [0.0, arm, 0.0],
                [0.0, -arm, 0.0],
            ],
            dtype=float,
        ),
        center,
        roll,
        pitch,
        yaw,
    )
    nose = transform_points(
        np.array([[0.0, 0.0, 0.0], [0.82 * arm, 0.0, 0.0]], dtype=float),
        center,
        roll,
        pitch,
        yaw,
    )

    gear_segments = [
        np.array([[-0.20, -0.18, 0.0], [-0.20, -0.18, -0.26]], dtype=float),
        np.array([[0.20, -0.18, 0.0], [0.20, -0.18, -0.26]], dtype=float),
        np.array([[-0.20, 0.18, 0.0], [-0.20, 0.18, -0.26]], dtype=float),
        np.array([[0.20, 0.18, 0.0], [0.20, 0.18, -0.26]], dtype=float),
        np.array([[-0.30, -0.18, -0.26], [0.30, -0.18, -0.26]], dtype=float),
        np.array([[-0.30, 0.18, -0.26], [0.30, 0.18, -0.26]], dtype=float),
    ]
    gear_x, gear_y, gear_z = transform_segments(gear_segments, center, roll, pitch, yaw)

    return {
        "arm1": arm1,
        "arm2": arm2,
        "motors": motors,
        "nose": nose,
        "gear_x": gear_x,
        "gear_y": gear_y,
        "gear_z": gear_z,
    }


def simulate(
    mass,
    k_thrust,
    k_yaw,
    izz,
    vertical_damping,
    altitude_cmd,
    yaw_cmd,
    forward_speed,
    alt_wn,
    alt_zeta,
    yaw_wn,
    yaw_zeta,
    motor_tau,
    ref_tau,
    mode,
):
    dt = 0.02
    total_time = 20.0
    gravity = 9.81
    max_omega = 1000.0
    max_z_accel_up = 5.0
    max_z_accel_down = 4.0
    max_yaw_accel = 4.0
    xy_response = 1.45
    xy_damping = 0.45
    tilt_tau = 0.18
    max_visual_tilt = 0.40
    t = np.arange(0.0, total_time + dt, dt)

    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    yaw = 0.0
    yaw_rate = 0.0
    roll_vis = 0.0
    pitch_vis = 0.0

    z_ref_state = 0.0
    yaw_ref_state = 0.0

    hover_omega_sq = mass * gravity / (4.0 * k_thrust)
    hover_omega = min(np.sqrt(max(hover_omega_sq, 0.0)), max_omega)
    omega = np.full(4, hover_omega, dtype=float)

    motor_alpha = 1.0 - np.exp(-dt / max(motor_tau, 1e-6))
    max_total_thrust = 4.0 * k_thrust * max_omega**2
    max_yaw_torque = 4.0 * k_yaw * max_omega**2

    x_hist, y_hist, z_hist = [], [], []
    yaw_hist, roll_hist, pitch_hist = [], [], []
    zr_hist, yr_hist = [], []
    vz_hist, yaw_rate_hist = [], []
    m1_hist, m2_hist, m3_hist, m4_hist = [], [], [], []

    for time in t:
        if mode == "Hover":
            z_ref_cmd = altitude_cmd
            yaw_ref_cmd = yaw_cmd
            forward_cmd = 0.0
        elif mode == "Altitude Step":
            z_ref_cmd = altitude_cmd if time >= 3.0 else 0.0
            yaw_ref_cmd = 0.0
            forward_cmd = 0.0
        elif mode == "Yaw Response":
            z_ref_cmd = altitude_cmd
            yaw_ref_cmd = yaw_cmd if time >= 3.0 else 0.0
            forward_cmd = 0.0
        else:
            z_ref_cmd = altitude_cmd
            yaw_ref_cmd = yaw_cmd
            forward_cmd = forward_speed

        z_ref_state = smooth_scalar(z_ref_state, z_ref_cmd, ref_tau, dt)
        yaw_ref_state = smooth_angle(yaw_ref_state, yaw_ref_cmd, ref_tau, dt)

        z_error = z_ref_state - z
        z_accel_cmd = alt_wn**2 * z_error - 2.0 * alt_zeta * alt_wn * vz
        z_accel_cmd = np.clip(z_accel_cmd, -max_z_accel_down, max_z_accel_up)

        thrust_cmd = mass * (gravity + z_accel_cmd + vertical_damping * vz)
        thrust_cmd = np.clip(thrust_cmd, 0.0, max_total_thrust)

        yaw_error = wrap_angle(yaw_ref_state - yaw)
        yaw_accel_cmd = yaw_wn**2 * yaw_error - 2.0 * yaw_zeta * yaw_wn * yaw_rate
        yaw_accel_cmd = np.clip(yaw_accel_cmd, -max_yaw_accel, max_yaw_accel)

        yaw_torque_cmd = np.clip(izz * yaw_accel_cmd, -max_yaw_torque, max_yaw_torque)

        base_omega_sq = thrust_cmd / (4.0 * k_thrust)
        yaw_delta_sq = yaw_torque_cmd / (4.0 * k_yaw)
        omega_sq_cmd = np.array(
            [
                base_omega_sq + yaw_delta_sq,
                base_omega_sq - yaw_delta_sq,
                base_omega_sq + yaw_delta_sq,
                base_omega_sq - yaw_delta_sq,
            ]
        )
        omega_sq_cmd = np.clip(omega_sq_cmd, 0.0, max_omega**2)
        omega_cmd = np.sqrt(omega_sq_cmd)
        omega += motor_alpha * (omega_cmd - omega)

        vx_cmd = forward_cmd * np.cos(yaw)
        vy_cmd = forward_cmd * np.sin(yaw)
        x_accel = xy_response * (vx_cmd - vx) - xy_damping * vx
        y_accel = xy_response * (vy_cmd - vy) - xy_damping * vy
        vx += x_accel * dt
        vy += y_accel * dt
        x += vx * dt
        y += vy * dt

        body_forward_acc = x_accel * np.cos(yaw) + y_accel * np.sin(yaw)
        body_lateral_acc = -x_accel * np.sin(yaw) + y_accel * np.cos(yaw)
        pitch_cmd = np.clip(-np.arctan2(body_forward_acc, gravity), -max_visual_tilt, max_visual_tilt)
        roll_cmd = np.clip(np.arctan2(body_lateral_acc, gravity), -max_visual_tilt, max_visual_tilt)
        roll_vis = smooth_scalar(roll_vis, roll_cmd, tilt_tau, dt)
        pitch_vis = smooth_scalar(pitch_vis, pitch_cmd, tilt_tau, dt)

        total_thrust = k_thrust * np.sum(omega**2)
        z_ddot = total_thrust / mass - gravity - vertical_damping * vz
        vz += z_ddot * dt
        z += vz * dt
        if z < 0.0:
            z = 0.0
            if vz < 0.0:
                vz = 0.0

        yaw_torque = k_yaw * (
            omega[0] ** 2
            - omega[1] ** 2
            + omega[2] ** 2
            - omega[3] ** 2
        )
        yaw_ddot = yaw_torque / izz
        yaw_rate += yaw_ddot * dt
        yaw += yaw_rate * dt
        yaw = wrap_angle(yaw)

        x_hist.append(x)
        y_hist.append(y)
        z_hist.append(z)
        yaw_hist.append(yaw)
        roll_hist.append(roll_vis)
        pitch_hist.append(pitch_vis)
        zr_hist.append(z_ref_state)
        yr_hist.append(yaw_ref_state)
        vz_hist.append(vz)
        yaw_rate_hist.append(yaw_rate)
        m1_hist.append(omega[0])
        m2_hist.append(omega[1])
        m3_hist.append(omega[2])
        m4_hist.append(omega[3])

    return {
        "t": t,
        "x": np.array(x_hist),
        "y": np.array(y_hist),
        "z": np.array(z_hist),
        "yaw": np.array(yaw_hist),
        "roll": np.array(roll_hist),
        "pitch": np.array(pitch_hist),
        "zr": np.array(zr_hist),
        "yr": np.array(yr_hist),
        "vz": np.array(vz_hist),
        "yaw_rate": np.array(yaw_rate_hist),
        "m1": np.array(m1_hist),
        "m2": np.array(m2_hist),
        "m3": np.array(m3_hist),
        "m4": np.array(m4_hist),
        "hover_omega": hover_omega,
        "max_omega": max_omega,
    }


st.header("Flight Inputs")

col1, col2 = st.columns(2)

with col1:
    altitude_cmd = st.number_input("Target Altitude (m)", value=8.0)
    yaw_cmd = st.number_input("Target Yaw (rad)", value=0.5)

with col2:
    mode = st.selectbox(
        "Mode",
        ["Hover", "Altitude Step", "Yaw Response", "Forward Flight"]
    )
    mass = st.number_input("Mass (kg)", value=1.20)

# --------------------------------------------------
# ADVANCED (HIDDEN — SAME VARIABLES, SAME CODE)
# --------------------------------------------------
with st.expander("Advanced Parameters (optional)"):

    k_thrust = st.number_input("Thrust Coefficient", value=0.00001, format="%.6f")
    k_yaw = st.number_input("Yaw Moment Coefficient", value=0.0000003, format="%.7f")

    izz = st.number_input("Yaw Inertia Izz", value=0.03)
    vertical_damping = st.number_input("Vertical Damping", value=0.35)
    motor_tau = st.number_input("Motor Time Constant (s)", value=0.12)

    forward_speed = st.number_input("Forward Speed Command (m/s)", value=1.2)

    alt_wn = st.number_input("Altitude Natural Frequency", value=1.25)
    alt_zeta = st.number_input("Altitude Damping Ratio", value=1.15)

    yaw_wn = st.number_input("Yaw Natural Frequency", value=1.8)
    yaw_zeta = st.number_input("Yaw Damping Ratio", value=1.05)

    ref_tau = st.number_input("Reference Smoothing (s)", value=0.30)

run = st.button("Run Simulation")

hover_omega_preview = np.sqrt(max(mass * 9.81 / (4.0 * k_thrust), 0.0))
if hover_omega_preview > 1000.0:
    st.warning(
        "The current mass/thrust settings require a hover rotor speed above the model limit "
        "of 1000 rad/s. Increase thrust coefficient or reduce mass for better realism."
    )


if run:
    res = simulate(
        mass=mass,
        k_thrust=k_thrust,
        k_yaw=k_yaw,
        izz=izz,
        vertical_damping=vertical_damping,
        altitude_cmd=altitude_cmd,
        yaw_cmd=yaw_cmd,
        forward_speed=forward_speed,
        alt_wn=alt_wn,
        alt_zeta=alt_zeta,
        yaw_wn=yaw_wn,
        yaw_zeta=yaw_zeta,
        motor_tau=motor_tau,
        ref_tau=ref_tau,
        mode=mode,
    )
    st.success("Simulation Completed")

    t = res["t"]
    x = res["x"]
    y = res["y"]
    z = res["z"]
    yaw = res["yaw"]
    roll = res["roll"]
    pitch = res["pitch"]

    xmin = float(min(np.min(x), 0.0) - 2.0)
    xmax = float(max(np.max(x), 0.0) + 2.0)
    ymin = float(min(np.min(y), 0.0) - 2.0)
    ymax = float(max(np.max(y), 0.0) + 2.0)
    zmax = float(max(np.max(z) + 2.0, altitude_cmd + 2.0, 3.0))

    frame_step = 4
    frame_indices = list(range(0, len(t), frame_step))
    if frame_indices[-1] != len(t) - 1:
        frame_indices.append(len(t) - 1)

    frames = []
    for i in frame_indices:
        geom = quad_points_3d(x[i], y[i], z[i], roll[i], pitch[i], yaw[i])
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x[: i + 1],
                        y=y[: i + 1],
                        z=z[: i + 1],
                        mode="lines",
                        line=dict(width=6, color="#4fa3ff"),
                        name="Flight Path",
                    ),
                    go.Scatter3d(
                        x=x[: i + 1],
                        y=y[: i + 1],
                        z=np.zeros(i + 1),
                        mode="lines",
                        line=dict(width=4, color="rgba(148, 163, 184, 0.55)"),
                        name="Ground Track",
                    ),
                    go.Scatter3d(
                        x=geom["arm1"][:, 0],
                        y=geom["arm1"][:, 1],
                        z=geom["arm1"][:, 2],
                        mode="lines",
                        line=dict(width=10, color="#bfc7d5"),
                        name="Arm 1",
                    ),
                    go.Scatter3d(
                        x=geom["arm2"][:, 0],
                        y=geom["arm2"][:, 1],
                        z=geom["arm2"][:, 2],
                        mode="lines",
                        line=dict(width=10, color="#bfc7d5"),
                        name="Arm 2",
                    ),
                    go.Scatter3d(
                        x=geom["gear_x"],
                        y=geom["gear_y"],
                        z=geom["gear_z"],
                        mode="lines",
                        line=dict(width=5, color="#7dd3fc"),
                        name="Landing Gear",
                    ),
                    go.Scatter3d(
                        x=geom["motors"][:, 0],
                        y=geom["motors"][:, 1],
                        z=geom["motors"][:, 2],
                        mode="markers",
                        marker=dict(
                            size=7,
                            color=["#ef4444", "#22c55e", "#22c55e", "#ef4444"],
                        ),
                        name="Motors",
                    ),
                    go.Scatter3d(
                        x=geom["nose"][:, 0],
                        y=geom["nose"][:, 1],
                        z=geom["nose"][:, 2],
                        mode="lines",
                        line=dict(width=8, color="#f59e0b"),
                        name="Nose",
                    ),
                ],
                name=str(i),
            )
        )

    geom0 = quad_points_3d(x[0], y[0], z[0], roll[0], pitch[0], yaw[0])

    fig_view = go.Figure(
        data=[
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="lines",
                line=dict(width=6, color="#4fa3ff"),
                name="Flight Path",
            ),
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[0.0],
                mode="lines",
                line=dict(width=4, color="rgba(148, 163, 184, 0.55)"),
                name="Ground Track",
            ),
            go.Scatter3d(
                x=geom0["arm1"][:, 0],
                y=geom0["arm1"][:, 1],
                z=geom0["arm1"][:, 2],
                mode="lines",
                line=dict(width=10, color="#bfc7d5"),
                name="Arm 1",
            ),
            go.Scatter3d(
                x=geom0["arm2"][:, 0],
                y=geom0["arm2"][:, 1],
                z=geom0["arm2"][:, 2],
                mode="lines",
                line=dict(width=10, color="#bfc7d5"),
                name="Arm 2",
            ),
            go.Scatter3d(
                x=geom0["gear_x"],
                y=geom0["gear_y"],
                z=geom0["gear_z"],
                mode="lines",
                line=dict(width=5, color="#7dd3fc"),
                name="Landing Gear",
            ),
            go.Scatter3d(
                x=geom0["motors"][:, 0],
                y=geom0["motors"][:, 1],
                z=geom0["motors"][:, 2],
                mode="markers",
                marker=dict(
                    size=7,
                    color=["#ef4444", "#22c55e", "#22c55e", "#ef4444"],
                ),
                name="Motors",
            ),
            go.Scatter3d(
                x=geom0["nose"][:, 0],
                y=geom0["nose"][:, 1],
                z=geom0["nose"][:, 2],
                mode="lines",
                line=dict(width=8, color="#f59e0b"),
                name="Nose",
            ),
        ],
        frames=frames,
    )

    fig_view.update_layout(
        template="plotly_dark",
        height=680,
        showlegend=False,
        title="PX4 X500 UAV 3D Flight View",
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis=dict(
                title="X Position [m]",
                range=[xmin, xmax],
                backgroundcolor="#0f172a",
                gridcolor="#334155",
            ),
            yaxis=dict(
                title="Y Position [m]",
                range=[ymin, ymax],
                backgroundcolor="#0f172a",
                gridcolor="#334155",
            ),
            zaxis=dict(
                title="Altitude [m]",
                range=[0.0, zmax],
                backgroundcolor="#0f172a",
                gridcolor="#334155",
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=1.40, z=0.95)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 35, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
                direction="left",
                x=0.36,
                y=1.05,
                showactive=True,
            )
        ],
    )

    st.plotly_chart(fig_view, use_container_width=True)

    horizontal_range = np.max(np.sqrt(x**2 + y**2))
    peak_tilt_deg = np.degrees(np.max(np.sqrt(roll**2 + pitch**2)))

    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric("Hover Rotor Speed", f"{res['hover_omega']:.1f} rad/s")
    with metric2:
        st.metric("Peak Altitude", f"{np.max(z):.2f} m")
    with metric3:
        st.metric("Horizontal Range", f"{horizontal_range:.2f} m")
    with metric4:
        st.metric("Peak Visual Tilt", f"{peak_tilt_deg:.1f} deg")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_resp = go.Figure()
        fig_resp.add_trace(go.Scatter(x=t, y=res["z"], name="Altitude"))
        fig_resp.add_trace(go.Scatter(x=t, y=res["zr"], name="Altitude Target", line=dict(dash="dash")))
        fig_resp.update_layout(
            title="Altitude Response",
            template="plotly_dark",
            xaxis_title="Time [s]",
            yaxis_title="Altitude [m]",
        )
        st.plotly_chart(fig_resp, use_container_width=True)

    with col_b:
        fig_yaw = go.Figure()
        fig_yaw.add_trace(go.Scatter(x=t, y=res["yaw"], name="Yaw"))
        fig_yaw.add_trace(go.Scatter(x=t, y=res["yr"], name="Yaw Target", line=dict(dash="dot")))
        fig_yaw.update_layout(
            title="Yaw Response",
            template="plotly_dark",
            xaxis_title="Time [s]",
            yaxis_title="Yaw [rad]",
        )
        st.plotly_chart(fig_yaw, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig_track = go.Figure()
        fig_track.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Ground Track",
                line=dict(width=3, color="#4fa3ff"),
            )
        )
        fig_track.add_trace(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[y[0], y[-1]],
                mode="markers+text",
                text=["Start", "End"],
                textposition="top center",
                marker=dict(size=10, color=["#22c55e", "#ef4444"]),
                name="Markers",
            )
        )
        fig_track.update_layout(
            title="Top-Down Ground Track",
            template="plotly_dark",
            xaxis_title="X Position [m]",
            yaxis_title="Y Position [m]",
        )
        fig_track.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_track, use_container_width=True)

    with col_d:
        fig_mot = go.Figure()
        fig_mot.add_trace(go.Scatter(x=t, y=res["m1"], name="Motor 1"))
        fig_mot.add_trace(go.Scatter(x=t, y=res["m2"], name="Motor 2"))
        fig_mot.add_trace(go.Scatter(x=t, y=res["m3"], name="Motor 3"))
        fig_mot.add_trace(go.Scatter(x=t, y=res["m4"], name="Motor 4"))
        fig_mot.update_layout(
            title="Motor Speeds",
            template="plotly_dark",
            xaxis_title="Time [s]",
            yaxis_title="Rotor Speed [rad/s]",
        )
        st.plotly_chart(fig_mot, use_container_width=True)

    st.markdown("### What happens in the 3D model")
    st.markdown(
        "- The drone moves through **x, y, and z**, so yaw steers the horizontal path instead of only rotating the top view.\n"
        "- The 3D body uses **roll, pitch, and yaw** for visualization, making acceleration and turning look more like actual flight."
    )