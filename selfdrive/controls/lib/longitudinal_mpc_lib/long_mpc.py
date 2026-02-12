#!/usr/bin/env python3
import os
import time
import numpy as np
from cereal import log
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.realtime import DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.conversions import Conversions as CV

# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.car.gm.values import BOLT_REGEN_DECEL_BP, BOLT_REGEN_DECEL_V

if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

# ===== VOACC SPEED-BASED TUNING PARAMETERS =====
# City: Emergency-responsive | Highway: Rubber-banding prevention
# Speed ranges: [0-35, 35-55, 55-70, 70+ mph]

# SPEED BREAKPOINTS (mph)
SPEED_BREAKPOINTS = [0, 35, 55, 70]  # 4 ranges: 0-35, 35-55, 55-70, 70+

# ===== CHANGE THESE VALUES FOR DIFFERENT SPEEDS =====

# RESPONSIVENESS TO LEAD CARS (Lower = More responsive, Higher = More stable)
# [City Emergency, Urban Hwy, Rural Hwy, High Speed]
X_EGO_OBSTACLE_COSTS = [3.0, 3.0, 2.5, 2.0]  # Less aggressive at low speeds, closer to original

# JERK CONTROL (Lower = More jerky/responsive, Higher = Smoother/conservative)
# [City Emergency, Urban Hwy, Rural Hwy, High Speed]
J_EGO_COSTS = [5.5, 5.25, 5.0, 4.5]  # Slightly increased for smoother ride

# ACCELERATION CHANGE PENALTIES (Lower = More responsive, Higher = Smoother)
# [City Emergency, Urban Hwy, Rural Hwy, High Speed]
A_CHANGE_COSTS = [250, 240, 220, 200]  # Increased for smoother acceleration changes

# SMOOTHING FILTERS - Speed-adaptive for optimal responsiveness
# Lower = More responsive, Higher = Smoother
LEAD_FILTER_TIME_LOW = 0.8  # Under 40 mph: Fast response for city emergency braking
LEAD_FILTER_TIME_HIGH = 1.5  # Over 40 mph: Smoother 1.5s (was 1.2s) for relaxed following
SPEED_FILTER_THRESHOLD = 40 * CV.MPH_TO_MS  # 40 mph threshold

# DISTANCE ADAPTATION STRENGTH (How much penalties increase when close to lead)
# [City, Urban Hwy, Rural Hwy, High Speed]
DIST_ADAPTS = [0.04, 0.06, 0.06, 0.05]  # Balanced across speeds

# ===== END TUNING PARAMETERS =====


# Function to get parameter value based on current speed
def get_speed_based_param(speed_mph, param_array):
  """Get parameter value based on current speed using smooth interpolation"""
  return np.interp(speed_mph, SPEED_BREAKPOINTS, param_array)


# Current active values (set based on speed)
X_EGO_OBSTACLE_COST = 2.75
J_EGO_COST = 5.5
A_CHANGE_COST = 250.0
LEAD_FILTER_TIME = 2.0
DIST_ADAPT = 0.06

X_EGO_COST = 0.0
V_EGO_COST = 0.0
A_EGO_COST = 0.0
DANGER_ZONE_COST = 100.0
CRASH_DISTANCE = 0.25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'
# Default lead acceleration decay set to 50% at 1s
LEAD_ACCEL_TAU = 1.5


# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N + 1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.0])
COMFORT_BRAKE = 2.5


def get_jerk_factor(
  aggressive_jerk_acceleration=0.5,
  aggressive_jerk_danger=0.5,
  aggressive_jerk_speed=0.5,
  standard_jerk_acceleration=1.0,
  standard_jerk_danger=1.0,
  standard_jerk_speed=1.0,
  relaxed_jerk_acceleration=1.0,
  relaxed_jerk_danger=1.0,
  relaxed_jerk_speed=1.0,
  custom_personalities=False,
  personality=log.LongitudinalPersonality.standard,
):
  if custom_personalities:
    if personality == log.LongitudinalPersonality.relaxed:
      return relaxed_jerk_acceleration, relaxed_jerk_danger, relaxed_jerk_speed
    elif personality == log.LongitudinalPersonality.standard:
      return standard_jerk_acceleration, standard_jerk_danger, standard_jerk_speed
    elif personality == log.LongitudinalPersonality.aggressive:
      return aggressive_jerk_acceleration, aggressive_jerk_danger, aggressive_jerk_speed
    else:
      raise NotImplementedError("Longitudinal personality not supported")
  else:
    if personality == log.LongitudinalPersonality.relaxed:
      return 1.0, 1.0, 1.0
    elif personality == log.LongitudinalPersonality.standard:
      return 1.0, 1.0, 1.0
    elif personality == log.LongitudinalPersonality.aggressive:
      return 0.5, 0.5, 0.5
    else:
      raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(
  aggressive_follow=1.25, standard_follow=1.45, relaxed_follow=1.75, custom_personalities=False, personality=log.LongitudinalPersonality.standard
):
  if custom_personalities:
    if personality == log.LongitudinalPersonality.relaxed:
      return relaxed_follow
    elif personality == log.LongitudinalPersonality.standard:
      return standard_follow
    elif personality == log.LongitudinalPersonality.aggressive:
      return aggressive_follow
    else:
      raise NotImplementedError("Longitudinal personality not supported")
  else:
    if personality == log.LongitudinalPersonality.relaxed:
      return 1.75
    elif personality == log.LongitudinalPersonality.standard:
      return 1.45
    elif personality == log.LongitudinalPersonality.aggressive:
      return 1.25
    else:
      raise NotImplementedError("Longitudinal personality not supported")


def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)


def get_safe_obstacle_distance(v_ego, t_follow):
  from openpilot.common.params import Params

  params = Params()
  stop_str = params.get("StopDistance", encoding="utf8")
  stop_distance = float(stop_str) if stop_str else 6.0
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + stop_distance


def desired_follow_distance(v_ego, v_lead, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return get_safe_obstacle_distance(v_ego, t_follow) - get_stopped_equivalence_factor(v_lead)


def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  prev_a = SX.sym('prev_a')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  prev_a = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  ocp.cost.yref = np.zeros((COST_DIM,))
  ocp.cost.yref_e = np.zeros((COST_E_DIM,))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  accel_change = a_ego - prev_a
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.0), x_ego, v_ego, a_ego, accel_change, j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego, (a_ego - a_min), (a_max - a_ego), ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.0))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])

  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4 * np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, mode='acc', dt=DT_MDL):
    self.mode = mode
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.source = SOURCES[2]
    # Initialize smoothing filters with default time constants
    self.current_filter_time = LEAD_FILTER_TIME_LOW
    self.lead_a_filter = FirstOrderFilter(0.0, self.current_filter_time, self.dt)
    self.lead_v_filter = FirstOrderFilter(0.0, self.current_filter_time, self.dt)
    # Slew-limited filter factor to avoid abrupt 0.50↔1.00 jumps
    self.filter_time_factor = 1.0
    self.slew_per_sec = 1.0
    # Instance variables to avoid global modifications
    self.current_x_ego_cost = X_EGO_OBSTACLE_COSTS[0]
    self.current_j_ego_cost = J_EGO_COSTS[0]
    self.current_a_change_cost = A_CHANGE_COSTS[0]
    self.current_dist_adapt = DIST_ADAPTS[0]
    # Initialize acceleration limits to prevent AttributeError
    self.cruise_min_a = ACCEL_MIN
    self.max_a = 1.2  # Default max acceleration
    # Lead speed matching: smooth transition for cruise_obstacle factor
    self.cruise_obstacle_factor = 1.0  # 1.0 = normal, higher = less cruise pull
    self.reset()

  def reset(self):
    # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.solver.reset()
    # self.solver.options_set('print_level', 2)
    self.v_solution = np.zeros(N + 1)
    self.a_solution = np.zeros(N + 1)
    self.prev_a = np.array(self.a_solution)
    self.j_solution = np.zeros(N)
    self.yref = np.zeros((N + 1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N + 1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.params = np.zeros((N + 1, PARAM_DIM))
    for i in range(N + 1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4, 4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(
    self,
    acceleration_jerk=1.0,
    danger_jerk=1.0,
    speed_jerk=1.0,
    prev_accel_constraint=True,
    personality=log.LongitudinalPersonality.standard,
    v_ego=0.0,
    lead_dist=50.0,
    uncertainty=0.0,
    accel_reengage=False,
    panic_bypass=False,
    lead_v_rel=0.0,
    has_lead=False,
    hard_brake_prob=0.0,
    model_confidence=2,  # 0=red, 1=yellow, 2=green
    lane_changing=False,
    lead_future_dist=-1.0,  # Predicted lead distance 2s ahead, -1 if unavailable
    model_desired_accel=0.0,  # Model's desired acceleration for regen check
    is_bolt_regen=True,  # Whether this is a Bolt EV with regen paddle
  ):
    # Update parameters based on current speed with interpolation for smooth scaling
    speed_mph = v_ego * CV.MS_TO_MPH  # Convert m/s to mph

    # Use speed-based parameters for smooth scaling across all breakpoints
    self.current_x_ego_cost = get_speed_based_param(speed_mph, X_EGO_OBSTACLE_COSTS)

    # Early Braking: Scale cost based on closing speed (only when lead exists and closing)
    # #1: Skip when no lead, #2: Only apply when actually closing on lead
    # -5.0 m/s (~18kph closing) -> 2.0x Cost (Coast early)
    # -2.0 m/s (~7kph closing) -> 1.0x Cost (Normal)
    if has_lead and lead_v_rel < -0.1:
      close_factor = interp(lead_v_rel, [-5.0, -2.0], [2.0, 1.0])
      self.current_x_ego_cost *= close_factor

    self.current_j_ego_cost = get_speed_based_param(speed_mph, J_EGO_COSTS)
    self.current_a_change_cost = get_speed_based_param(speed_mph, A_CHANGE_COSTS)

    # For dist_adapt, start from 0.0 under low speeds while enabling full smooth transitions
    dist_adapt_array = [0.0, DIST_ADAPTS[1], DIST_ADAPTS[2], DIST_ADAPTS[3]]
    self.current_dist_adapt = get_speed_based_param(speed_mph, dist_adapt_array)

    # Update filter time constants with interp and recreate filters if needed
    # Dynamic TTC Smoothing combined with speed-based logic
    # #3: TTC upper clipping (100.0), #4: Speed+TTC combined filter
    # Base filter from speed with improved low-speed response
    # 0 mph → 0.0s (instant), 12 mph → 0.8s, 70 mph → 1.5s
    base_filter = interp(speed_mph, [0, 12, 70], [0.0, LEAD_FILTER_TIME_LOW, LEAD_FILTER_TIME_HIGH])

    # Speed-proportional safety distance: max(15m, 1 second of travel)
    # At 70 mph (~31 m/s): 31m safety distance
    # At 30 mph (~13 m/s): 15m safety distance (minimum)
    safety_dist = max(15.0, v_ego * 1.0)

    # Adjust safety distance based on model confidence
    # red=0: +30%, yellow=1: +15%, green=2: no change
    if model_confidence == 0:  # red - low confidence
      safety_dist *= 1.3
    elif model_confidence == 1:  # yellow - medium confidence
      safety_dist *= 1.15

    # Bolt EV Regen Capability Override: If model wants more decel than regen can provide, instant response
    # This is critical for the Bolt EV which can only brake with pedal and regen paddle
    if is_bolt_regen and model_desired_accel < 0:
      # Get max regen deceleration at current speed (negative value)
      max_regen_decel = interp(v_ego, BOLT_REGEN_DECEL_BP, BOLT_REGEN_DECEL_V)
      # If model wants more deceleration than regen can provide (with 20% safety margin)
      if model_desired_accel < max_regen_decel * 0.8:  # e.g., -1.4 * 0.8 = -1.12
        self.current_filter_time = 0.0
    # Model Hard Brake Override: If model predicts >30% probability of 3m/s² hard braking, instant response
    elif hard_brake_prob > 0.3:
      self.current_filter_time = 0.0

    # Dynamic MPC Cost Adjustment based on Regen Safety Margin (User Request)
    # "Flexible Deceleration": If we have plenty of margin to stop using regen even in worst-case,
    # act smoother (lower obstacle cost, higher jerk cost).
    if is_bolt_regen and has_lead and lead_v_rel < -0.1:
      # Calculate Max Regen Decel at current speed
      max_regen_decel = interp(v_ego, BOLT_REGEN_DECEL_BP, BOLT_REGEN_DECEL_V) # negative value
      
      # 1. Ego Stopping Distance at Max Regen
      t_stop_ego = v_ego / -max_regen_decel
      ego_stop_dist = 0.5 * v_ego * t_stop_ego
      
      # 2. Lead Stopping Distance (Worst Case: Lead brakes efficiently or stops)
      # User logic: consider current lead acceleration
      # If lead is braking (a < 0), project it. If accelerating, assume constant speed (conservative? no, worse is braking)
      # Actually, worse case is lead slamming brakes. But user liked "current lead behavior" logic + buffer.
      # Let's use the same logic as warning: Current Lead Accel clamped to 0 velocity.
      lead_v = v_ego + lead_v_rel # approx
      # We don't have lead_a passed to set_weights explicitly, but we have internal filtered lead_a
      lead_a = self.lead_a_filter.x
      
      lead_travel_dist = 0.0
      if lead_a < 0:
        t_stop_lead = -lead_v / lead_a
        if t_stop_lead < t_stop_ego:
           lead_travel_dist = 0.5 * lead_v * t_stop_lead
        else:
           lead_travel_dist = (lead_v * t_stop_ego) + (0.5 * lead_a * t_stop_ego**2)
      else:
        # Lead not braking -> assume constant speed or slight decel? 
        # User request: "flexible... as long as regen can handle worst case"
        # Let's assume constant speed for the "flexible" check, as we want to be permissive when safe.
        # If we assume lead slams brakes, we will never be "safe" enough to relax.
        lead_travel_dist = lead_v * t_stop_ego
      
      # 3. Predicted Gap after stop
      # current gap + lead_travel - ego_travel
      predicted_gap = (lead_dist + lead_travel_dist) - ego_stop_dist
      
      # 4. Safety Margin (Gap - 4.0m buffer)
      regen_safety_margin = predicted_gap - 4.0
      
      # 5. Modulate Costs
      # If margin is HUGE (> 15m), we can be very soft.
      # If margin is TIGHT (< 5m), we must be firm.
      if regen_safety_margin > 10.0:
        # Safe: Reduce obstacle cost (allow getting closer), Increase jerk cost (enforce smoothness)
        # Scale obstacle cost down to 20%?
        safe_factor = interp(regen_safety_margin, [10.0, 30.0], [0.5, 0.1])
        self.current_x_ego_cost *= safe_factor
        
        # Scale jerk cost up to 5x?
        jerk_factor = interp(regen_safety_margin, [10.0, 30.0], [2.0, 5.0])
        acceleration_jerk *= jerk_factor
        speed_jerk *= jerk_factor
        
      elif regen_safety_margin < 5.0:
        # Unsafe: Increase obstacle cost (strict), Decrease jerk cost (allow reaction)
        unsafe_factor = interp(regen_safety_margin, [0.0, 5.0], [5.0, 1.0])
        self.current_x_ego_cost *= unsafe_factor
        
        # Allow quick reaction
        jerk_relax = interp(regen_safety_margin, [0.0, 5.0], [0.1, 1.0])
        acceleration_jerk *= jerk_relax
        speed_jerk *= jerk_relax

    # Safety Override: Instant response only when within safety distance AND closing on lead
    # If close but not closing (following at same speed), use normal filter for comfort
    elif has_lead and lead_dist < safety_dist and lead_v_rel < -0.1:
      self.current_filter_time = 0.0
    # TTC-based filter scaling (only when lead exists and closing)
    # #1: Division by zero protection with min lead_dist
    elif has_lead and lead_v_rel < -0.1 and lead_dist > 0.5:
      # Use predicted future distance if available (more conservative for safety)
      effective_dist = lead_dist
      if lead_future_dist > 0:
        effective_dist = min(lead_dist, lead_future_dist)
      ttc = min(max(effective_dist, 0.5) / -lead_v_rel, 100.0)  # #3: Clamp TTC to [0, 100]
      # TTC < 2.5s: Filter 0.0 (Instant Safety)
      # TTC > 5.0s: Filter 1.2s (Max Smoothness)
      ttc_based_filter = interp(ttc, [2.5, 5.0], [0.0, LEAD_FILTER_TIME_HIGH])
      # Use min of base_filter and ttc_based_filter to prevent sluggishness
      # This ensures we don't jump to a higher filter time (laggy) than the speed-based tuning
      self.current_filter_time = min(base_filter, ttc_based_filter)
    else:
      self.current_filter_time = base_filter

    # Lane changing protection: maintain minimum filter to prevent jerky behavior
    if lane_changing and self.current_filter_time < 0.5:
      self.current_filter_time = 0.5
    if abs(self.current_filter_time - getattr(self, 'prev_filter_time', 0)) > 0.1:  # Only update if significant change
      # Recreate filters with new time constant while preserving current values
      current_a = self.lead_a_filter.x if hasattr(self.lead_a_filter, 'x') else 0.0
      current_v = self.lead_v_filter.x if hasattr(self.lead_v_filter, 'x') else 0.0
      self.lead_a_filter = FirstOrderFilter(current_a, self.current_filter_time, self.dt)
      self.lead_v_filter = FirstOrderFilter(current_v, self.current_filter_time, self.dt)
      self.prev_filter_time = self.current_filter_time

    # Adaptive jerk factors for distance with interp scaling
    dist_factor = 1.0 + self.current_dist_adapt * (20.0 / max(lead_dist, 5.0))
    acceleration_jerk *= dist_factor
    danger_jerk *= dist_factor
    speed_jerk *= dist_factor

    # Scene complexity adjustment based on model uncertainty
    prev_filter_time_factor = getattr(self, 'prev_filter_time_factor', 1.0)
    # Target factor from uncertainty
    if uncertainty <= 0.45:
      tgt_factor = 1.0
    elif uncertainty >= 0.70:
      tgt_factor = 0.0
    else:
      tgt_factor = float(np.interp(uncertainty, [0.45, 0.70], [1.0, 0.30]))

    # if accel_reengage:
    #   tgt_factor = min(tgt_factor, 0.5)

    # Hard bypass of smoothing when approaching fast or magnitude trips
    if panic_bypass:
      tgt_factor = 0.0

    # Slew-limit changes to avoid step-wise filter jumps
    max_step = self.slew_per_sec * self.dt
    delta = np.clip(tgt_factor - self.filter_time_factor, -max_step, max_step)
    self.filter_time_factor += float(delta)
    filter_time_factor = float(self.filter_time_factor)

    # When uncertainty is moderately elevated, allow accel but cap jerk by increasing jerk cost
    if 0.45 <= uncertainty < 0.60:
      scale = float(np.interp(uncertainty, [0.45, 0.60], [1.2, 1.5]))
      speed_jerk *= scale

    if self.mode == 'acc':
      a_change_cost = acceleration_jerk if prev_accel_constraint else 0
      cost_weights = [self.current_x_ego_cost, X_EGO_COST, V_EGO_COST, A_EGO_COST, a_change_cost, speed_jerk]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, danger_jerk]
    elif self.mode == 'blended':
      a_change_cost = 40.0 if prev_accel_constraint else 0
      cost_weights = [0.0, 0.1, 0.2, 5.0, a_change_cost, 1.0]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, danger_jerk]
    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
    self.set_cost_weights(cost_weights, constraint_cost_weights)

    # Adjust filter time constants for complex scenes
    if abs(filter_time_factor - getattr(self, 'prev_filter_time_factor', 1.0)) > 0.05:
      current_a = self.lead_a_filter.x if hasattr(self.lead_a_filter, 'x') else 0.0
      current_v = self.lead_v_filter.x if hasattr(self.lead_v_filter, 'x') else 0.0
      new_filter_time = self.current_filter_time * filter_time_factor
      self.lead_a_filter = FirstOrderFilter(current_a, new_filter_time, self.dt)
      self.lead_v_filter = FirstOrderFilter(current_v, new_filter_time, self.dt)
      self.prev_filter_time_factor = filter_time_factor

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.0:  # probably only helps if v < v_prev
      for i in range(N + 1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, v_ego=0.0):
    speed_mph = v_ego * CV.MS_TO_MPH
    bp = [0, 20, 35]
    exp_weight = interp(speed_mph, bp, [1.0, 1.0, 0.0])  # Full exp at <20, blend to constant at 35

    if exp_weight > 0:
      # Exponential decay component
      a_lead_traj_exp = a_lead * np.exp(-a_lead_tau * (T_IDXS**2) / 2.0)
      v_lead_traj_exp = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj_exp), 0.0, 1e8)
      x_lead_traj_exp = x_lead + np.cumsum(T_DIFFS * v_lead_traj_exp)
    else:
      x_lead_traj_exp = np.zeros_like(T_IDXS)
      v_lead_traj_exp = np.zeros_like(T_IDXS)

    # Constant acceleration component
    v_lead_traj_const = np.clip(v_lead + a_lead * T_IDXS, 0.0, 1e8)
    x_lead_traj_const = x_lead + v_lead * T_IDXS + 0.5 * a_lead * T_IDXS**2

    # Blend based on weight
    v_lead_traj = exp_weight * v_lead_traj_exp + (1 - exp_weight) * v_lead_traj_const
    x_lead_traj = exp_weight * x_lead_traj_exp + (1 - exp_weight) * x_lead_traj_const

    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead, tracking_lead=True):
    v_ego = self.x0[1]
    if lead is not None and lead.status and tracking_lead:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead) / 2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = clip(x_lead, min_x_lead, 1e8)
    v_lead = clip(v_lead, 0.0, 1e8)
    a_lead = clip(a_lead, -10.0, 5.0)
    # Apply smoothing filters with interp scaling
    self.lead_a_filter.update(a_lead)
    self.lead_v_filter.update(v_lead)
    a_lead = self.lead_a_filter.x
    v_lead = self.lead_v_filter.x
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, v_ego)
    return lead_xv

  def set_accel_limits(self, min_a, max_a):
    # TODO this sets a max accel limit, but the minimum limit is only for cruise decel
    # needs refactor
    self.cruise_min_a = min_a
    self.max_a = max_a

  def update(self, lead_one, lead_two, v_cruise, x, v, a, j, t_follow, tracking_lead, personality=log.LongitudinalPersonality.standard, stable_lead=False):
    v_ego = self.x0[1]
    self.status = lead_one.status and tracking_lead or lead_two.status

    lead_xv_0 = self.process_lead(lead_one, tracking_lead)
    lead_xv_1 = self.process_lead(lead_two, v_ego)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:, 0] + get_stopped_equivalence_factor(lead_xv_0[:, 1])
    lead_1_obstacle = lead_xv_1[:, 0] + get_stopped_equivalence_factor(lead_xv_1[:, 1])

    self.params[:, 0] = ACCEL_MIN
    # negative accel constraint causes problems because negative speed is not allowed
    self.params[:, 1] = max(0.0, self.max_a)

    # Update in ACC mode or ACC/e2e blend
    if self.mode == 'acc':
      self.params[:, 5] = LEAD_DANGER_FACTOR

      # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
      # when the leads are no factor.
      v_lower = v_ego + (T_IDXS * self.cruise_min_a * 1.05)
      # TODO does this make sense when max_a is negative?
      v_upper = v_ego + (T_IDXS * self.max_a * 1.05)
      v_cruise_clipped = np.clip(v_cruise * np.ones(N + 1), v_lower, v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow)

      # Lead Speed Matching: When stably following a lead (distance maintained),
      # push cruise_obstacle further away to reduce the "pull" toward set speed.
      # This prevents oscillation between lead following and cruise acceleration.
      # Use smooth transition to avoid abrupt changes.
      target_factor = 1.5 if (stable_lead and tracking_lead) else 1.0
      # Slew rate: ~0.5 per second for smooth transition (takes ~1s to reach target)
      max_change = 0.5 * self.dt
      if self.cruise_obstacle_factor < target_factor:
        self.cruise_obstacle_factor = min(self.cruise_obstacle_factor + max_change, target_factor)
      else:
        self.cruise_obstacle_factor = max(self.cruise_obstacle_factor - max_change, target_factor)
      cruise_obstacle = cruise_obstacle * self.cruise_obstacle_factor

      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      self.source = SOURCES[np.argmin(x_obstacles[0])]

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
      self.params[:, 5] = 1.0

      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])
      x = np.cumsum(np.insert(xforward, 0, x[0]))

      x_and_cruise = np.column_stack([x, cruise_target])
      x = np.min(x_and_cruise, axis=1)

      self.source = 'e2e' if x_and_cruise[1, 0] < x_and_cruise[1, 1] else 'cruise'

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    self.yref[:, 1] = x
    self.yref[:, 2] = v
    self.yref[:, 3] = a
    self.yref[:, 5] = j
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params[:, 2] = np.min(x_obstacles, axis=1)
    self.params[:, 3] = np.copy(self.prev_a)
    self.params[:, 4] = t_follow

    self.run()
    lead_probability = lead_one.modelProb
    if np.any(lead_xv_0[FCW_IDXS, 0] - self.x_sol[FCW_IDXS, 0] < CRASH_DISTANCE) and lead_probability > 0.9:
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

    # Check if it got within lead comfort range
    # TODO This should be done cleaner
    if self.mode == 'blended':
      if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:, 1], t_follow)) - self.x_sol[:, 0] < 0.0):
        self.source = 'lead0'
      if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:, 1], t_follow)) - self.x_sol[:, 0] < 0.0) and (lead_1_obstacle[0] - lead_0_obstacle[0]):
        self.source = 'lead1'

  def run(self):
    # t0 = time.monotonic()
    # reset = 0
    for i in range(N + 1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    # qp_iter = self.solver.get_stats('statistics')[-1][-1] # SQP_RTI specific
    # print(f"long_mpc timings: tot {self.solve_time:.2e}, qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e}, \
    # integrator {self.time_integrator:.2e}, qp_iter {qp_iter}")
    # res = self.solver.get_residuals()
    # print(f"long_mpc residuals: {res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}")
    # self.solver.print_statistics()

    for i in range(N + 1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:, 1]
    self.a_solution = self.x_sol[:, 2]
    self.j_solution = self.u_sol[:, 0]

    self.prev_a = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    if self.solution_status != 0:
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(time.monotonic() - t0):.2e} qp {self.time_qp_solution:.2e}, \
    # lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
