
# Autonomous Vehicle Control via Control Barrier Functions (CBF-QP) & Stanley Steering

An autonomous driving framework combining **Control Barrier Functions (CBF)** solved via Quadratic Programming (QP) for longitudinal safety with a **Stanley Controller** for lateral lane keeping and lane changing across CommonRoad XML scenarios.

---

##  Repository Structure

```text
cbf_qp/
├── scenarios/                  # CommonRoad XML scenario files
│   ├── USA_US101-9_1_T-1.xml
│   ├── ZAM_Zip-1_32_I-1-1.xml
│   └── ZAM_Zip-1_64_T-1.xml
├── src/                        # Core system modules
│   ├── __init__.py
│   ├── behavior_planner.py    # State machine for high-level decision making
│   ├── cbf_solver.py          # CBF-QP safety solver (Analytical & CVXPY)
│   ├── lateral_controller.py  # Stanley steering controller & trajectory extraction
│   ├── radar.py               # Field-of-View perception & lead vehicle tracking
│   ├── scenario_loader.py     # CommonRoad XML parser & ego state setup
│   ├── utils.py               # Animation build utilities & file cleanup
│   ├── vehicle_dynamics.py    # Kinematic bicycle model & vehicle geometry
│   └── visualizer.py          # Render engine & GIF frame generation
├── run_usa.py                  # Entry point for US-101 highway scenario
├── run_zam.py                  # Entry point for ZAM Zip merge scenario
├── requirements.txt            # Project dependencies
└── README.md
```
##  System Architecture & Pipeline Workflow

The simulation loop processes every step through four sequential pipelines:

```text
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│  1. Perception  │ ──► │ 2. Behavior Plan  │ ──► │ 3. CBF-QP Control │ ──► │ 4. Dynamics & Render │
└─────────────────┘     └───────────────────┘     └───────────────────┘     └──────────────────────┘
   (Radar FOV)           (Finite State Machine)     (Longitudinal Safety)     (Kinematic Bicycle Model)
```
### 1. Radar Perception (`src/radar.py`)
* Filters dynamic obstacles using a forward-facing body-frame radar cone specified by maximum range $R_{\text{max}}$ and field-of-view angle $\theta_{\text{FOV}}$.
* Transforms surrounding obstacle coordinates into the ego-vehicle local coordinate frame to track the primary lead vehicle in the current driving corridor.

### 2. High-Level Behavior Planning (`src/behavior_planner.py`)
* Manages state transitions across `LANE_KEEP`, `GAP_SEARCH`, and `LANE_CHANGE`.
* Queries road lanelets to extract and resample smooth reference trajectory waypoints for target paths.

### 3. Safety-Critical Control (`src/cbf_solver.py` & `src/lateral_controller.py`)
* **Longitudinal Controller (CBF-QP):** Enforces dynamic safety distance using the Control Barrier Function:

  $$h(x) = \Delta x - (d_{\text{min}} + v_{\text{ego}} \cdot \tau) \ge 0$$

  Solves a Quadratic Program (or analytical closed-form fallback) to guarantee safe acceleration control inputs ($u$) while aiming for a desired cruising velocity $v_{\text{des}}$.

* **Lateral Controller (Stanley Method):** Computes front-wheel steering angle ($\delta$) using cross-track error ($e_y$) and heading error ($\psi_{\text{e}}$):

  $$\delta(t) = \psi_{\text{e}}(t) + \arctan\left(\frac{k \cdot e_y(t)}{v(t) + k_{\text{soft}}}\right)$$

### 4. Dynamics & Visualization (`src/vehicle_dynamics.py` & `src/visualizer.py`)
* Propagates vehicle state $(x, y, \psi, v)$ using a non-linear Kinematic Bicycle Model.
* Evaluates polygon geometry for collision detection and renders output frames into animated `.gif` files.

---

##  Environment Setup

### Prerequisites
* **Python 3.9+**
* `pip` package manager

### 1. Clone the Repository
git clone [https://github.com/your-username/cbf_qp.git](https://github.com/your-username/cbf_qp.git)

cd cbf_qp
### 2. Create and Activate a Virtual Environment
* **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
  
* **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
### 3. Install Dependencies
```bash
 pip install --upgrade pip
 pip install -r requirements.txt
```
##  Execution Instructions

Run either scenario directly from the project root directory:

### Run USA Highway Scenario (Gap Search & Dynamic Lane Change)
Simulates dynamic lane selection and safe distance maintenance on a highway track:
```bash
python run_usa.py
```
* **Output:** Saves `usa_us101_radar_cbf.gif` in the project root directory.

### Run ZAM Zip-Merge Scenario (Track Following & Safe Braking)
Simulates map-following trajectory tracking with strict CBF safety distance control behind lead traffic:
```bash
python run_zam.py
```
* **Output:** Saves `zam_zip_radar_cbf.gif` in the project root directory.

---

##  Verification & Results

Upon completion, the execution log confirms simulation state metrics:

```text
Loaded Scenario: USA_US101-9_1_T-1
Ego Initial Position: (212.50, -42.10)
...
Simulation Complete! Output saved to: 'usa_us101_radar_cbf.gif'
🛡️ SAFE! CBF Safety Control maintained nominal distance.
```
