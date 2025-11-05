# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="TB Within-Host Phase Portrait", layout="wide")
st.title("TB Within-Host Dynamics: Phase Plane")

# --- Sidebar sliders
st.sidebar.header("Model Parameters")
beta_b = st.sidebar.slider("β_b (immune suppressing TB)", 0.1, 5.0, 1.0, 0.1)
beta_i = st.sidebar.slider("β_i (TB suppressing immune)", 0.1, 5.0, 1.0, 0.1)
eta_b = st.sidebar.slider("η_b (TB self-limiting)", 0.5, 2.0, 1.5, 0.05)
eta_i = st.sidebar.slider("η_i (immune self-limiting)", 0.5, 2.0, 1.5, 0.05)

# --- Define the system
def tb_ode(t, y):
    b, i = y
    dbdt = beta_b * b * (eta_b * (1 - b) - i)
    didt = beta_i * i * (eta_i * (1 - i) - b)
    return [dbdt, didt]

# --- Vector field
b_vals = np.linspace(0, 1.2, 20)
i_vals = np.linspace(0, 1.2, 20)
B, I = np.meshgrid(b_vals, i_vals)

dB = beta_b * B * (eta_b * (1 - B) - I)
dI = beta_i * I * (eta_i * (1 - I) - B)

# Normalize arrows for visualization
norm = np.sqrt(dB**2 + dI**2)
dB /= norm
dI /= norm

fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(B, I, dB, dI, color="gray", alpha=0.6)

# --- Add trajectories from initial conditions
initial_conditions = [
    [0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9],
    [0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]
]

for y0 in initial_conditions:
    sol = solve_ivp(tb_ode, [0, 50], y0, t_eval=np.linspace(0, 50, 500))
    ax.plot(sol.y[0], sol.y[1], lw=1.8)

ax.set_xlabel("b(t) - scaled TB")
ax.set_ylabel("i(t) - scaled immune response")
ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.2])
ax.set_title("Phase Plane with Vector Field and Trajectories")
st.pyplot(fig)
