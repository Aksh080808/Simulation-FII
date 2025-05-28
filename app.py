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

    # Save simulation data option BEFORE running simulation
    st.header("Save your simulation setup")
    save_as = st.text_input("Filename to save current inputs", value=sim_name, key="save_filename")
    if st.button("💾 Save Current Setup"):
        # Save all needed data to JSON file
        data_to_save = {
            "station_groups": [{"group_name": g, "equipment": valid_groups[g]} for g in valid_groups],
            "connections": [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            "from_stations": st.session_state.from_stations,
            "duration": duration,
            "simulation_name": save_as,
            "valid_groups": valid_groups,  # Save also the eq dict
        }
        with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
            json.dump(data_to_save, f, indent=2)
        st.success(f"Saved simulation as {save_as}.json")

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
    env.process(sim.run())  # run() must be a generator
    env.run(until=duration)
    return sim

class FactorySim:
    def __init__(self, env, station_groups_data, connections, from_stations, duration):
        self.env = env
        self.duration = duration

        # Store topology
        self.station_groups = {}
        self.connections = defaultdict(list)
        self.from_stations = from_stations  # dict[group_name] = list of from groups (or empty)

        # Create SimPy resources per equipment
        for group in station_groups_data:
            group_name = group["group_name"]
            equipment = group["equipment"]
            self.station_groups[group_name] = {
                "resources": {},
                "cycle_times": {},
                "throughput": 0,
                "completed": 0,
                "queue_times": [],
                "active_count": 0,
            }
            for eq_name, cycle_time in equipment.items():
                res = simpy.Resource(env, capacity=1)
                self.station_groups[group_name]["resources"][eq_name] = res
                self.station_groups[group_name]["cycle_times"][eq_name] = cycle_time

        # Build connection map: from -> [to]
        for src, dst in connections:
            self.connections[src].append(dst)

        # Data tracking
        self.throughput_per_group = defaultdict(int)
        self.start_time = env.now

    def run(self):
        # Start feeders at groups with no from_stations (start of line)
        for group_name in self.station_groups:
            if not self.from_stations.get(group_name):
                self.env.process(self.feeder(group_name))

        # Start workers for all equipment in all groups
        for group_name, group in self.station_groups.items():
            for eq_name in group["resources"]:
                self.env.process(self.worker(group_name, eq_name))

        # Run simulation until duration
        while self.env.now < self.duration:
            yield self.env.timeout(1)

    def feeder(self, group_name):
        """Continuously feed boards into a station group."""
        group = self.station_groups[group_name]
        while True:
            # Feed boards at intervals determined by the min cycle time of this group (rough guess)
            min_ct = min(group["cycle_times"].values())
            yield self.env.timeout(min_ct)

            # Immediately start processing on the first equipment in the group
            for eq_name, resource in group["resources"].items():
                # Try request resource if free
                if resource.count < resource.capacity:
                    # Start processing a board
                    self.env.process(self.process_board(group_name, eq_name))
                    break

    def worker(self, group_name, eq_name):
        """Worker process that continuously tries to process boards."""
        group = self.station_groups[group_name]
        resource = group["resources"][eq_name]
        cycle_time = group["cycle_times"][eq_name]

        while True:
            with resource.request() as req:
                yield req
                # Process one board for cycle_time
                yield self.env.timeout(cycle_time)
                # Increment throughput count
                group["completed"] += 1
                self.throughput_per_group[group_name] += 1

    def process_board(self, group_name, eq_name):
        """Process a board through one equipment and pass it downstream."""
        group = self.station_groups[group_name]
        resource = group["resources"][eq_name]
        cycle_time = group["cycle_times"][eq_name]

        with resource.request() as req:
            yield req
            yield self.env.timeout(cycle_time)
            group["completed"] += 1
            self.throughput_per_group[group_name] += 1

            # Pass board downstream
            downstream = self.connections.get(group_name, [])
            for next_group in downstream:
                # Feed next group
                self.env.process(self.process_board(next_group, next(iter(self.station_groups[next_group]["resources"]))))

# ========== Result Display ==========

def show_detailed_summary(sim, valid_groups, from_stations, duration):
    st.header("📈 Simulation Summary")

    data = []
    for group_name, group_data in sim.station_groups.items():
        completed = group_data["completed"]
        eq_cycle_times = valid_groups[group_name].values()
        avg_cycle = sum(eq_cycle_times) / len(eq_cycle_times)
        utilization = (completed * avg_cycle) / duration if duration > 0 else 0
        from_list = from_stations.get(group_name, [])
        data.append({
            "Station Group": group_name,
            "Throughput (completed units)": completed,
            "Avg Cycle Time (sec)": round(avg_cycle, 2),
            "Utilization (%)": round(utilization * 100, 2),
            "From Stations": ", ".join(from_list) if from_list else "START"
        })
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Download button
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Summary CSV", csv_data, "simulation_summary.csv", "text/csv")

# ========== Main App Logic ==========

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
        else:
            st.session_state.page = "main"
            main_page()

if __name__ == "__main__":
    main()
