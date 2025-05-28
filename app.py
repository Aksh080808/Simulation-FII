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

            st.session_state.from_stations[name] = [] if "START" in from_selected else from_selected
            st.session_state.connections[name] = [] if "STOP" in to_selected else to_selected

    # Step 3: Duration and Save
    duration = st.number_input("Simulation Duration (seconds)", min_value=10, value=100, step=10, key="sim_duration_new")
    sim_name = st.text_input("Simulation Name", value="simulation_summary", key="sim_name_new").strip()
    if not sim_name:
        sim_name = "simulation_summary"

    st.header("Save your simulation setup")
    save_as = st.text_input("Filename to save current inputs", value=sim_name, key="save_filename")
    if st.button("💾 Save Current Setup"):
        data_to_save = {
            "station_groups": [{"group_name": g, "equipment": valid_groups[g]} for g in valid_groups],
            "connections": [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            "from_stations": st.session_state.from_stations,
            "duration": duration,
            "simulation_name": save_as,
            "valid_groups": valid_groups,
        }
        with open(os.path.join(SAVE_DIR, f"{save_as}.json"), "w") as f:
            json.dump(data_to_save, f, indent=2)
        st.success(f"Saved simulation as {save_as}.json")

    if st.button("▶️ Run Simulation"):
        station_groups_data = [{"group_name": g, "equipment": valid_groups[g]} for g in valid_groups]
        run_result = run_simulation_backend(
            station_groups_data,
            [(src, dst) for src, tos in st.session_state.connections.items() for dst in tos],
            st.session_state.from_stations,
            duration,
        )
        show_detailed_summary(run_result, valid_groups, st.session_state.from_stations, duration)

def open_simulation():
    st.title("📂 Open Simulation")
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")]

    if not files:
        st.warning("No simulations found.")
        return

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
        valid_groups = {g["group_name"]: g["equipment"] for g in data["station_groups"]}
        show_detailed_summary(run_result, valid_groups, data["from_stations"], duration)

# ========== Simulation Backend ==========

def run_simulation_backend(station_groups_data, connections_list, from_stations_dict, duration):
    env = simpy.Environment()
    station_groups = {g["group_name"]: g["equipment"] for g in station_groups_data}
    sim = FactorySimulation(env, station_groups, duration, dict(connections_list), from_stations_dict)
    env.process(sim.run())
    env.run(until=duration)
    return sim

# ========== Simulation Class ==========

class FactorySimulation:
    def __init__(self, env, station_groups, duration, connections, from_stations):
        self.env = env
        self.station_groups = station_groups
        self.connections = connections
        self.from_stations = from_stations
        self.duration = duration

        self.buffers = defaultdict(lambda: simpy.Store(env))
        self.resources = {eq: simpy.Resource(env, capacity=1) for group in station_groups.values() for eq in group}
        self.cycle_times = {eq: ct for group in station_groups.values() for eq, ct in group.items()}
        self.equipment_to_group = {eq: group for group, eqs in station_groups.items() for eq in eqs}
        self.throughput_in = defaultdict(int)
        self.throughput_out = defaultdict(int)
        self.wip_over_time = defaultdict(list)
        self.time_points = []
        self.equipment_busy_time = defaultdict(float)
        self.board_id = 1
        self.wip_interval = 5
        env.process(self.track_wip())

    def equipment_worker(self, eq):
        group = self.equipment_to_group[eq]
        while True:
            board = yield self.buffers[group].get()
            self.throughput_in[eq] += 1
            with self.resources[eq].request() as req:
                yield req
                start = self.env.now
                yield self.env.timeout(self.cycle_times[eq])
                end = self.env.now
                self.equipment_busy_time[eq] += (end - start)
            self.throughput_out[eq] += 1
            for tgt in self.connections.get(group, []):
                yield self.buffers[tgt].put(board)

    def track_wip(self):
        while True:
            snapshot = {}
            for group in self.station_groups:
                in_count = sum(self.throughput_in[eq] for eq in self.station_groups[group])
                out_count = sum(self.throughput_out[eq] for eq in self.station_groups[group])
                snapshot[group] = max(0, in_count - out_count)
            self.time_points.append(self.env.now)
            for group, wip in snapshot.items():
                self.wip_over_time[group].append(wip)
            yield self.env.timeout(self.wip_interval)

    def feeder(self):
        while True:
            for group, sources in self.from_stations.items():
                if not sources:
                    yield self.buffers[group].put(f"B{self.board_id}")
                    self.board_id += 1
            yield self.env.timeout(1)

    def run(self):
     for group in self.station_groups:
        for eq in self.station_groups[group]:
            self.env.process(self.equipment_worker(eq))
        self.env.process(self.feeder())
        yield self.env.timeout(0)  # ✅ makes this a generator



# ========== Summary Display ==========

def show_detailed_summary(sim, valid_groups, from_stations, duration):
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

        prev_out = sum(sim.throughput_out.get(eq, 0) for g in from_stations.get(group, []) for eq in valid_groups.get(g, []))
        curr_in = agg[group]['in']
        agg[group]['wip'] = max(0, prev_out - curr_in)

    df = pd.DataFrame([{
        "Station Group": g,
        "Boards In": agg[g]['in'],
        "Boards Out": agg[g]['out'],
        "WIP": agg[g]['wip'],
        "Number of Equipment": agg[g]['count'],
        "Cycle Times (s)": ", ".join([f"{ct:.1f}" for ct in agg[g]['cycle_times']]),
        "Utilization (%)": round(100 * agg[g]['busy'] / (agg[g]['count'] * duration), 1) if agg[g]['count'] else 0.0
    } for g in groups])

    st.dataframe(df)

    # Optional: Download Summary as Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(sim.wip_over_time).to_excel(writer, sheet_name="WIP_Over_Time", index=False)
    output.seek(0)
    st.download_button("📥 Download Summary as Excel", data=output, file_name="simulation_results.xlsx")

    # Optional: Show WIP Chart
    st.markdown("### 📈 WIP Over Time")
    wip_df = pd.DataFrame(sim.wip_over_time)
    wip_df["Time"] = sim.time_points
    wip_df = wip_df.set_index("Time")
    st.line_chart(wip_df)

# ========== Page Navigation ==========
if st.session_state.page == "login":
    login_page()
elif not st.session_state.authenticated:
    login_page()
elif st.session_state.page == "main":
    main_page()
elif st.session_state.page == "new":
    new_simulation()
elif st.session_state.page == "open":
    open_simulation()
elif st.session_state.page == "edit":
    edit_simulation()
 
