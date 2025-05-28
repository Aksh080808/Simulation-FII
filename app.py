# production_line_simulator.py

import streamlit as st
import os
import json
import simpy
import pandas as pd
from collections import defaultdict
from io import BytesIO

# ========== Configuration ==========
SAVE_DIR = "simulations"
USERNAME = "aksh.fii"
PASSWORD = "foxy123"
os.makedirs(SAVE_DIR, exist_ok=True)
st.set_page_config(page_title="Production Line Simulator", layout="wide")

# ========== Session State Setup ==========
for key in ["authenticated", "page", "simulation_data", "group_names", "connections", "from_stations"]:
    if key not in st.session_state:
        if key == "authenticated":
            st.session_state[key] = False
        elif key == "page":
            st.session_state[key] = "login"
        elif key in ["connections", "from_stations"]:
            st.session_state[key] = {}
        elif key == "group_names":
            st.session_state[key] = []
        else:
            st.session_state[key] = None

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

    # Step 1: Enter station groups
    st.header("Step 1: Define Station Groups")
    num_groups = st.number_input("How many station groups?", min_value=1, step=1, key="num_groups_new")
    group_names = []
    valid_groups = {}

    for i in range(num_groups):
        with st.expander(f"Station Group {i + 1}"):
            group_name = st.text_input(f"Group Name {i + 1}", key=f"group_name_{i}").strip().upper()
            if group_name:
                num_eq = st.number_input(f"Number of Equipment in {group_name}", min_value=1, step=1, key=f"eq_count_{i}")
                eq_dict = {}
                for j in range(num_eq):
                    eq_name = f"{group_name}_EQ{j+1}"
                    cycle_time = st.number_input(f"Cycle Time for {eq_name} (sec)", min_value=0.1, key=f"ct_{i}_{j}")
                    eq_dict[eq_name] = cycle_time
                valid_groups[group_name] = eq_dict
                group_names.append(group_name)
            else:
                group_names.append("")

    st.session_state.group_names = group_names

    # Step 2: Connections with START/STOP logic
    st.header("Step 2: Connect Stations")
    if "from_stations" not in st.session_state:
        st.session_state.from_stations = {}
    if "connections" not in st.session_state:
        st.session_state.connections = {}

    for i, name in enumerate(group_names):
        if not name:
            continue
        with st.expander(f"{name} Connections"):
            from_options = ['START'] + [g for g in group_names if g and g != name]
            to_options = ['STOP'] + [g for g in group_names if g and g != name]

            from_selected = st.multiselect(f"{name} receives from:", from_options, key=f"from_{i}")
            to_selected = st.multiselect(f"{name} sends to:", to_options, key=f"to_{i}")

            # If 'START' selected, means no from_stations (start of line)
            st.session_state.from_stations[name] = [] if "START" in from_selected else from_selected
            # If 'STOP' selected, means no connections forward (end of line)
            st.session_state.connections[name] = [] if "STOP" in to_selected else to_selected

    # Simulation duration and name
    duration = st.number_input("Simulation Duration (seconds)", min_value=10, value=100, step=10, key="sim_duration_new")
    sim_name = st.text_input("Simulation Name", value="simulation_summary", key="sim_name_new").strip()
    if not sim_name:
        sim_name = "simulation_summary"

    # Run Simulation button
    if st.button("▶️ Run Simulation"):
        # Prepare station_groups_data from valid_groups dict
        station_groups_data = []
        for g_name, eqs in valid_groups.items():
            station_groups_data.append({"group_name": g_name, "equipment": eqs})

        run_result = run_simulation_backend(
            station_groups_data,
            [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            st.session_state.from_stations,
            duration,
        )

        # Show summary table with WIP and utilization
        show_detailed_summary(run_result, valid_groups, st.session_state.from_stations, duration)

        # Save simulation data option
        with st.expander("📁 Save Simulation"):
            save_as = st.text_input("Filename", value=sim_name, key="save_filename")
            if st.button("💾 Save Simulation"):
                # Save all needed data to JSON file
                data_to_save = {
                    "station_groups": station_groups_data,
                    "connections": [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
                    "from_stations": st.session_state.from_stations,
                    "duration": duration,
                    "simulation_name": save_as,
                    "valid_groups": valid_groups,  # Save also the eq dict
                }
                with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
                    json.dump(data_to_save, f, indent=2)
                st.success(f"Saved simulation as {save_as}.json")

def open_simulation():
    st.title("📂 Open Simulation")
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]

    if not files:
        st.warning("No simulations found.")
        return

    # Show saved simulation names (read from inside JSON)
    display_names = []
    file_map = {}
    for f in files:
        try:
            with open(os.path.join(SAVE_DIR, f), "r") as jf:
                data = json.load(jf)
                display_name = data.get("simulation_name", f[:-5])
                display_names.append(display_name)
                file_map[display_name] = f
        except Exception:
            display_names.append(f[:-5])
            file_map[f[:-5]] = f

    selected_name = st.selectbox("Choose simulation to open:", display_names)
    if st.button("Open Selected Simulation"):
        filename = file_map[selected_name]
        with open(os.path.join(SAVE_DIR, filename), "r") as f:
            data = json.load(f)
        st.session_state.simulation_data = data
        st.session_state.page = "edit"

def edit_simulation():
    data = st.session_state.simulation_data
    st.title(f"✏️ Edit & Rerun Simulation: {data.get('simulation_name', 'Unnamed')}")
    st.json(data)
    duration = st.number_input("Simulation Duration (seconds)", value=data.get("duration", 100), step=10, key="edit_duration")

    if st.button("▶️ Run Simulation Again"):
        run_result = run_simulation_backend(
            data["station_groups"],
            data["connections"],
            data["from_stations"],
            duration,
        )
        # We need valid_groups for summary - rebuild from station_groups
        valid_groups = {g["group_name"]: g["equipment"] for g in data["station_groups"]}
        show_detailed_summary(run_result, valid_groups, data["from_stations"], duration)

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
        self.throughput_in = defaultdict(int)
        self.throughput_out = defaultdict(int)
        self.equipment_busy_time = defaultdict(float)
        self.group_eq_map = defaultdict(list)
        self.board_id = 0

        # Create resources and map eq to group
        for group in station_groups_data:
            g_name = group["group_name"]
            for eq_name, ct in group["equipment"].items():
                self.resources[eq_name] = simpy.Resource(env, capacity=1)
                self.cycle_times[eq_name] = ct
                self.group_eq_map[g_name].append(eq_name)

    def run(self):
        # Start feeder processes for start groups
        start_groups = [g for g in self.group_eq_map if g not in self.from_stations or not self.from_stations[g]]
        for g in start_groups:
            self.env.process(self.feeder(g))

        # Start workers for all equipment
        for eq in self.resources:
            self.env.process(self.worker(eq))

    def feeder(self, group):
        while self.env.now < self.duration:
            board = f"B{self.board_id}"
            self.board_id += 1
            yield self.buffers[group].put(board)
            yield self.env.timeout(1)  # Feed one board per second (adjust if needed)

    def worker(self, eq):
        group = next(g for g, eqs in self.group_eq_map.items() if eq in eqs)
        while True:
            board = yield self.buffers[group].get()
            self.throughput_in[eq] += 1
            with self.resources[eq].request() as req:
                start_busy = self.env.now
                yield req
                yield self.env.timeout(self.cycle_times[eq])
                end_busy = self.env.now
                self.equipment_busy_time[eq] += (end_busy - start_busy)
            self.throughput_out[eq] += 1
            for next_group in self.connections.get(group, []):
                yield self.buffers[next_group].put(board)

def show_detailed_summary(sim, valid_groups, from_stations, sim_time):
    st.markdown("---")
    st.subheader("📊 Simulation Results Summary")

    groups = list(valid_groups.keys())
    agg = defaultdict(lambda: {'in': 0, 'out': 0, 'busy': 0, 'count': 0, 'cycle_times': [], 'wip': 0})

    for group in groups:
        eqs = valid_groups[group]
        for eq in eqs:
            agg[group]['in'] += sim.throughput_in.get(eq, 0)
            agg[group]['out'] += sim.throughput_out.get(eq, 0)
            agg[group]['busy'] += sim.equipment_busy_time.get(eq, 0)
            agg[group]['cycle_times'].append(sim.cycle_times.get(eq, 0))
            agg[group]['count'] += 1

        prev_out = sum(
            sim.throughput_out.get(eq, 0)
            for g in from_stations.get(group, [])
            for eq in valid_groups.get(g, {})
        )
        curr_in = agg[group]['in']
        agg[group]['wip'] = max(0, prev_out - curr_in)

    # Prepare DataFrame
    df = pd.DataFrame([{
        "Station Group": g,
        "Boards In": agg[g]['in'],
        "Boards Out": agg[g]['out'],
        "WIP": agg[g]['wip'],
        "Number of Equipment": agg[g]['count'],
        "Cycle Times (sec)": ", ".join(str(round(ct, 1)) for ct in agg[g]['cycle_times']),
        "Utilization (%)": round((agg[g]['busy'] / (sim_time * agg[g]['count'])) * 100, 1) if agg[g]['count'] > 0 else 0
    } for g in groups])

    st.dataframe(df, use_container_width=True)

    # Excel download
    towrite = BytesIO()
    df.to_excel(towrite, index=False, sheet_name="Summary")
    towrite.seek(0)
    st.download_button(
        "📥 Download Summary Excel",
        data=towrite,
        file_name="simulation_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ========== Main Control ==========

def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        if st.session_state.page == "main":
            main_page()
        elif st.session_state.page == "new":
            new_simulation()
        elif st.session_state.page == "open":
            open_simulation()
        elif st.session_state.page == "edit":
            edit_simulation()

if __name__ == "__main__":
    main()
