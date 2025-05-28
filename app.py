# production_line_simulator.py

import streamlit as st
import os
import json
import simpy
import pandas as pd
from collections import defaultdict

# ========== Configuration ==========
SAVE_DIR = "simulations"
USERNAME = "aksh.fii"
PASSWORD = "foxy123"
os.makedirs(SAVE_DIR, exist_ok=True)
st.set_page_config(page_title="Production Line Simulator", layout="wide")

# ========== Session State Setup ==========
for key in ["authenticated", "page", "simulation_data"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "simulation_data" else False if key == "authenticated" else "login"

# ========== Pages ==========

def login_page():
    st.title("🔐 Login")
    user = st.text_input("User ID")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.authenticated = True
            st.session_state.page = "main"
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

def main_page():
    st.title("📊 Simulation Portal")
    st.write("Choose an option:")
    col1, col2 = st.columns(2)
    if col1.button("➕ New Simulation"):
        st.session_state.page = "new"
    if col2.button("📂 Open Simulation"):
        st.session_state.page = "open"

def new_simulation():
    st.title("➕ New Simulation Setup")
    station_groups = []
    num_groups = st.number_input("How many station groups?", min_value=1, step=1)

    for i in range(num_groups):
        with st.expander(f"Station Group {i + 1}"):
            group_name = st.text_input(f"Group Name {i + 1}", key=f"group_name_{i}").strip().upper()
            num_eq = st.number_input(f"Number of Equipment in {group_name or f'Group {i+1}'}", min_value=1, step=1, key=f"eq_count_{i}")
            eq_dict = {}
            for j in range(num_eq):
                eq_name = f"{group_name}_EQ{j+1}"
                cycle_time = st.number_input(f"Cycle Time for {eq_name} (sec)", min_value=0.1, key=f"ct_{i}_{j}")
                eq_dict[eq_name] = cycle_time
            station_groups.append({"group_name": group_name, "equipment": eq_dict})

    st.subheader("Connections Between Groups")
    group_names = [g["group_name"] for g in station_groups]
    connections = []
    from_stations = {}
    for from_group in group_names:
        to_groups = st.multiselect(f"{from_group} connects to:", group_names, key=f"conn_{from_group}")
        connections.extend([(from_group, to) for to in to_groups])
        for to in to_groups:
            from_stations.setdefault(to, []).append(from_group)

    duration = st.number_input("Simulation Duration (seconds)", min_value=10, value=100, step=10)
    sim_name = st.text_input("Simulation Name", value="simulation_summary").strip() or "simulation_summary"

    if st.button("▶️ Run Simulation"):
        run_result = run_simulation_backend(station_groups, connections, from_stations, duration)
        summary = generate_summary_table(run_result)
        st.success("Simulation complete.")
        st.dataframe(summary)
        st.download_button("📥 Download Summary CSV", summary.to_csv(index=False), file_name=f"{sim_name}.csv", mime="text/csv")

        with st.expander("📁 Save Simulation"):
            save_as = st.text_input("Filename", value=sim_name)
            if st.button("💾 Save"):
                with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
                    json.dump({
                        "station_groups": station_groups,
                        "connections": connections,
                        "from_stations": from_stations,
                        "duration": duration
                    }, f, indent=2)
                st.success(f"Saved as {save_as}.json")

def open_simulation():
    st.title("📂 Open Simulation")
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]
    if not files:
        st.warning("No simulations found.")
        return
    selected = st.selectbox("Choose file:", files)
    if st.button("Open"):
        with open(os.path.join(SAVE_DIR, selected), "r") as f:
            data = json.load(f)
        st.session_state.simulation_data = data
        st.session_state.page = "edit"

def edit_simulation():
    data = st.session_state.simulation_data
    st.title("✏️ Edit & Rerun Simulation")
    st.json(data)
    duration = st.number_input("Simulation Duration (seconds)", value=data["duration"], step=10)
    if st.button("▶️ Run Simulation Again"):
        run_result = run_simulation_backend(data["station_groups"], data["connections"], data["from_stations"], duration)
        summary = generate_summary_table(run_result)
        st.success("Simulation complete.")
        st.dataframe(summary)
        st.download_button("📥 Download Summary CSV", summary.to_csv(index=False), file_name="rerun_summary.csv", mime="text/csv")

# ========== Simulation Backend ==========

def run_simulation_backend(station_groups_data, connections_list, from_stations_dict, duration):
    env = simpy.Environment()
    sim = FactorySim(env, station_groups_data, connections_list, from_stations_dict, duration)
    env.process(sim.run())
    env.run(until=duration)
    return sim

class FactorySim:
    def __init__(self, env, station_groups_data, connections, from_stations, duration):
        self.env = env
        self.duration = duration
        self.connections = defaultdict(list)
        for src, dst in connections:
            self.connections[src].append(dst)
        self.from_stations = from_stations
        self.buffers = defaultdict(lambda: simpy.Store(env))
        self.resources = {}
        self.cycle_times = {}
        self.throughput = defaultdict(int)
        self.group_eq_map = defaultdict(list)
        self.board_id = 0

        for group in station_groups_data:
            g_name = group["group_name"]
            for eq_name, ct in group["equipment"].items():
                self.resources[eq_name] = simpy.Resource(env, capacity=1)
                self.cycle_times[eq_name] = ct
                self.group_eq_map[g_name].append(eq_name)

    def run(self):
        for eq in self.resources:
            self.env.process(self.worker(eq))
        self.env.process(self.feeder())
    
    def feeder(self):
        start_groups = [g for g in self.group_eq_map if g not in self.from_stations]
        while self.env.now < self.duration:
            for g in start_groups:
                board = f"B{self.board_id}"
                self.board_id += 1
                yield self.buffers[g].put(board)
            yield self.env.timeout(1)

    def worker(self, eq):
        group = next(g for g, eqs in self.group_eq_map.items() if eq in eqs)
        while True:
            board = yield self.buffers[group].get()
            with self.resources[eq].request() as req:
                yield req
                yield self.env.timeout(self.cycle_times[eq])
            self.throughput[eq] += 1
            for next_group in self.connections.get(group, []):
                yield self.buffers[next_group].put(board)

def generate_summary_table(sim):
    data = [{"Equipment": eq, "Throughput": count, "Cycle Time": sim.cycle_times[eq]} for eq, count in sim.throughput.items()]
    return pd.DataFrame(data)

# ========== Page Routing ==========
if not st.session_state.authenticated:
    login_page()
elif st.session_state.page == "main":
    main_page()
elif st.session_state.page == "new":
    new_simulation()
elif st.session_state.page == "open":
    open_simulation()
elif st.session_state.page == "edit":
    edit_simulation()
