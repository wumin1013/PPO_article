from __future__ import annotations

from typing import Dict, Tuple, Any, List
from dataclasses import dataclass, field
import math

import numpy as np

@dataclass
class RewardContext:
    """Unified context for reward calculation."""
    # Core tracking
    contour_error: float
    progress: float
    velocity: float
    heading_error: float
    
    # Motion state
    jerk: float
    angular_jerk: float
    angular_acc: float = 0.0
    
    # KCM / Constraints
    kcm_intervention: float = 0.0
    
    # Turn info (from self.turn_info)
    corner_mask: bool = False
    turn_sign: int = 0
    
    # Termination / Status
    stall_triggered: bool = False
    lap_completed: bool = False
    is_closed: bool = False
    end_distance: float = 0.0  # Optional, not used in P0 reward logic but in signature
    
    # P4.0 Extra Status (for legacy P0 logic compatibility)
    p4_status: Dict[str, float] = field(default_factory=dict)
    
    # Legacy / Optional
    du_theta_u: float = 0.0
    du_v_u: float = 0.0
    
    # Corridor (Legacy P3.1/P5.2 compat)
    corridor_status: Dict[str, Any] = field(default_factory=dict)

    # Phase 31(v3.2): far-scale lookahead geometry (rad)
    lookahead_delta_theta: float = 0.0
    # Learnable lookahead control status
    lookahead_dist: float = 0.0
    lookahead_dist_norm: float = 0.0
    region_weight: float = 0.0


class RewardCalculator:
    """P0 reward: progress-dominant with tracking/heading/time penalties."""

    def __init__(
        self,
        weights: Dict[str, float],
        max_vel: float,
        max_ang_vel: float | None,
        half_epsilon: float,
        max_jerk: float,
        max_ang_jerk: float,
        max_ang_acc: float | None = None,
        safe_ratio: float | None = None,  # 兼容旧参数，逻辑中不再使用
    ):
        self.weights = weights or {}
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel
        self.max_ang_acc = max_ang_acc
        self.max_jerk = max_jerk
        self.max_ang_jerk = max_ang_jerk
        self.half_epsilon = max(half_epsilon, 1e-6)
        self.last_progress = 0.0
        self._cornerness_ema = 0.0

    def reset(self) -> None:
        self.last_progress = 0.0
        self._cornerness_ema = 0.0

    @staticmethod
    def _safe_abs(value: Any, *, fallback: float = 0.0) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not math.isfinite(v):
            return float(fallback)
        return abs(v)

    def _get_cornerness_cfg(self) -> Dict[str, Any]:
        cfg = self.weights.get("cornerness", {})
        return cfg if isinstance(cfg, dict) else {}

    def _compute_cornerness(self, ctx: RewardContext) -> Tuple[float, float]:
        cfg = self._get_cornerness_cfg()
        if not bool(cfg.get("enabled", False)):
            self._cornerness_ema = 0.0
            return 0.0, 0.0

        theta0_deg = float(cfg.get("theta0_deg", 30.0))
        theta0 = max(math.radians(theta0_deg), 1e-6)
        ema_tau_steps = float(cfg.get("ema_tau_steps", 10.0))
        ema_tau_steps = max(1.0, ema_tau_steps)
        alpha = 1.0 / ema_tau_steps

        delta_theta = self._safe_abs(ctx.lookahead_delta_theta, fallback=0.0)
        c_raw = float(delta_theta / theta0)
        self._cornerness_ema = float((1.0 - alpha) * float(self._cornerness_ema) + alpha * c_raw)
        c = float(np.clip(self._cornerness_ema, 0.0, 1.0))
        return c, c_raw

    def calculate_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        """Dispatch to minimal or legacy reward calculation."""
        # v2.0: Check minimal mode flag
        if self.weights.get("minimal_mode", False):
            return self._calculate_minimal_reward(ctx)
        return self._calculate_legacy_reward(ctx)

    def _calculate_minimal_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        """v3.0 Minimal reward: progress + tube reward + boundary/time/completion."""
        w_s = float(self.weights.get("w_s", 20.0))
        p4_cfg = self.weights.get("p4", {})
        time_penalty = float(p4_cfg.get("time_penalty", -0.02))
        stall_penalty = float(p4_cfg.get("stall_penalty", -8.0))
        boundary_cfg = self.weights.get("boundary", {})
        completion_cfg = self.weights.get("completion", {})

        tube_cfg = self.weights.get("tube", {})
        tube_enabled = bool(tube_cfg.get("enabled", True))
        tube_ratio = float(tube_cfg.get("ratio", 0.5))
        tube_tolerance = tube_ratio * self.half_epsilon

        # 1. Progress reward
        progress_now = float(ctx.progress)
        progress_diff = max(0.0, progress_now - float(self.last_progress))
        r_progress = w_s * progress_diff

        # 2. Contour penalty (tube reward)
        r_contour = 0.0
        contour_error = abs(float(ctx.contour_error))
        if tube_enabled:
            if contour_error < tube_tolerance:
                r_contour = 0.0
            else:
                excess = contour_error - tube_tolerance
                r_contour = -10.0 * (excess / self.half_epsilon) ** 2

        # 3. Boundary penalty (hard constraint)
        r_boundary = 0.0
        if boundary_cfg.get("enabled", False):
            if contour_error > float(self.half_epsilon):
                r_boundary = float(boundary_cfg.get("penalty", -100.0))

        # 4. Time penalty
        r_time = time_penalty

        # 5. Completion reward
        r_completion = 0.0
        if completion_cfg.get("enabled", False) and ctx.lap_completed:
            r_completion = float(completion_cfg.get("reward", 50.0))

        # 6. Stall penalty
        r_stall = 0.0
        if ctx.stall_triggered:
            r_stall = stall_penalty

        total = r_progress + r_contour + r_boundary + r_time + r_completion + r_stall
        self.last_progress = progress_now

        return total, {
            "progress_diff": float(progress_diff),
            "r_progress": float(r_progress),
            "r_contour": float(r_contour),
            "r_boundary": float(r_boundary),
            "r_time": float(r_time),
            "r_completion": float(r_completion),
            "r_stall": float(r_stall),
            "total": float(total),
        }

    def _calculate_legacy_reward(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        """P0: progress-dominant reward with pure penalties."""
        # Unpack weights
        w_s = abs(float(self.weights.get("w_s", 20.0)))
        w_e = abs(float(self.weights.get("w_e", 5.0)))
        w_tau = abs(float(self.weights.get("w_tau", 2.0)))
        w_t = abs(float(self.weights.get("w_t", 1.0)))
        w_smooth = abs(float(self.weights.get("w_smooth", 0.0)))
        w_ang_acc = abs(float(self.weights.get("w_ang_acc", 0.0)))
        smooth_corner_only = bool(self.weights.get("smooth_corner_only", False))
        track_deadzone_ratio = float(self.weights.get("track_deadzone_ratio", 0.0))
        track_outside_weight = float(self.weights.get("track_outside_weight", 1.0))
        corner_w_tau_scale = float(self.weights.get("corner_w_tau_scale", 1.0))
        
        # P6.1 Action Rate Penalty (if configured in weights)
        # Note: In Phase 20 Cleanup, P6.1 is being removed from env, but reward logic might be kept if configured.
        # Check if caller passed du values or if they are in p4_status
        du_enabled = bool(self.weights.get("p6_1", {}).get("du_enabled", False))
        
        # P4.0 Configs (Passed via weights or ctx?)
        # Originally passed as kwargs. Now we must look in self.weights['p4']
        p4_cfg = self.weights.get("p4", {})
        time_penalty = float(p4_cfg.get("time_penalty", -0.01))
        stall_penalty = float(p4_cfg.get("stall_penalty", -8.0))
        
        # Corridor Configs
        corridor_cfg = self.weights.get("corridor", {})
        corridor_out_penalty_w = float(corridor_cfg.get("outside_penalty_weight", 20.0))
        # P7.1 Dir Pref
        corridor_dir_pref_w = float(corridor_cfg.get("dir_pref_weight", 0.0))
        
        # --- Logic Start ---

        if not (track_deadzone_ratio >= 0.0):
            track_deadzone_ratio = 0.0
        elif track_deadzone_ratio >= 1.0:
            track_deadzone_ratio = 0.999

        if not (track_outside_weight >= 1.0):
            track_outside_weight = 1.0

        if not (corner_w_tau_scale >= 0.0):
            corner_w_tau_scale = 1.0

        progress_now = float(ctx.progress)
        progress_diff = max(0.0, progress_now - float(self.last_progress))

        cornerness_cfg = self._get_cornerness_cfg()
        cornerness_enabled = bool(cornerness_cfg.get("enabled", False))
        cornerness, cornerness_raw = self._compute_cornerness(ctx)
        cornerness_power = float(cornerness_cfg.get("power", 2.0))
        if not math.isfinite(cornerness_power) or cornerness_power <= 0.0:
            cornerness_power = 2.0

        w_track_eff = float(w_e)
        w_smooth_dyn = 0.0
        smooth_term = 0.0
        r_corner_smooth = 0.0
        if cornerness_enabled:
            w_track0 = self._safe_abs(cornerness_cfg.get("w_track0", w_e), fallback=w_e)
            w_track_min = self._safe_abs(cornerness_cfg.get("w_track_min", 0.2 * w_e), fallback=0.2 * w_e)
            w_track_min = max(w_track_min, 1e-6)
            w_track_eff = float(w_track0 * ((1.0 - cornerness) ** cornerness_power) + w_track_min)

            w_smooth0_default = max(float(w_smooth), float(w_ang_acc), 0.0)
            w_smooth0 = self._safe_abs(cornerness_cfg.get("w_smooth0", w_smooth0_default), fallback=w_smooth0_default)
            w_smooth_dyn = float(w_smooth0 * (cornerness ** cornerness_power))

            smooth_source = str(cornerness_cfg.get("smooth_source", "omega")).lower()
            if smooth_source == "domega":
                smooth_term = self._safe_abs(ctx.angular_acc) / max(float(self.max_ang_acc or 1.0), 1e-6)
            elif smooth_source == "jerk":
                smooth_term = self._safe_abs(ctx.jerk) / max(float(self.max_jerk), 1e-6)
            elif smooth_source in {"ang_jerk", "omega_jerk"}:
                smooth_term = self._safe_abs(ctx.angular_jerk) / max(float(self.max_ang_jerk), 1e-6)
            else:
                omega_exec = 0.0
                if isinstance(ctx.p4_status, dict):
                    omega_exec = float(ctx.p4_status.get("omega_exec", 0.0))
                smooth_term = self._safe_abs(omega_exec) / max(float(self.max_ang_vel or 1.0), 1e-6)
            smooth_power = float(cornerness_cfg.get("smooth_power", 2.0))
            if not math.isfinite(smooth_power) or smooth_power <= 0.0:
                smooth_power = 2.0
            r_corner_smooth = -w_smooth_dyn * (smooth_term**smooth_power)

        error_abs = abs(float(ctx.contour_error))
        error_ratio = error_abs / max(float(self.half_epsilon), 1e-6)
        tau = abs(float(ctx.heading_error))

        # 1. Progress Reward
        r_progress = w_s * progress_diff
        
        # 2. Tracking Penalty
        if bool(ctx.corner_mask):
            deadzone = track_deadzone_ratio * float(self.half_epsilon)
            if error_abs <= deadzone:
                r_track = 0.0
            elif error_abs <= float(self.half_epsilon):
                denom = max(float(self.half_epsilon) - deadzone, 1e-6)
                scaled = (error_abs - deadzone) / denom
                r_track = -w_track_eff * (scaled**2)
            else:
                r_track = -w_track_eff * (error_ratio**2) * track_outside_weight
        else:
            r_track = -w_track_eff * (error_ratio**2)

        # 3. Direction Penalty
        w_tau_eff = w_tau * (corner_w_tau_scale if bool(ctx.corner_mask) else 1.0)
        r_dir = -w_tau_eff * (tau**2)
        
        # 4. Time Penalty (P4)
        r_time = time_penalty

        # 5. Smoothness Penalty
        r_smooth = 0.0
        keep_legacy_smooth = bool(cornerness_cfg.get("keep_legacy_smooth", False))
        if (not cornerness_enabled) or keep_legacy_smooth:
            if w_smooth > 0.0 and (not smooth_corner_only or bool(ctx.corner_mask)):
                jerk_ratio = abs(float(ctx.jerk)) / max(float(self.max_jerk), 1e-6)
                ang_jerk_ratio = abs(float(ctx.angular_jerk)) / max(float(self.max_ang_jerk), 1e-6)
                r_smooth = -w_smooth * (jerk_ratio**2 + ang_jerk_ratio**2)
            if bool(ctx.corner_mask) and w_ang_acc > 0.0 and self.max_ang_acc is not None:
                ang_acc_ratio = abs(float(ctx.angular_acc)) / max(float(self.max_ang_acc), 1e-6)
                r_smooth += -w_ang_acc * (ang_acc_ratio**2)
        if cornerness_enabled:
            r_smooth += float(r_corner_smooth)

        # 6. Du Penalty (Legacy P6.1)
        # If enabled in weights, use du values from ctx (assume they are set if needed)
        r_du = 0.0
        if du_enabled:
             # Need to get du_mode and w_du from weights
             p6_1_cfg = self.weights.get("p6_1", {})
             w_du = float(p6_1_cfg.get("w_du", 0.01))
             du_mode = str(p6_1_cfg.get("du_mode", "l1")).lower()
             
             # Try to get from p4_status if not explicitly in ctx fields (ctx has du_theta_u, du_v_u)
             du_theta = ctx.du_theta_u
             du_v = ctx.du_v_u
             if du_theta == 0.0 and du_v == 0.0 and "du_theta_u" in ctx.p4_status:
                 du_theta = ctx.p4_status.get("du_theta_u", 0.0)
                 du_v = ctx.p4_status.get("du_v_u", 0.0)
                 
             if du_mode == "l1":
                 r_du = -w_du * (abs(du_theta) + abs(du_v))
             else:
                 r_du = -w_du * (du_theta**2 + du_v**2)

        # 7. Stall Penalty (P4)
        r_stall = 0.0
        if ctx.stall_triggered:
            r_stall = stall_penalty
            
        # 8. Corridor / Dir Pref (Legacy P3.1/P7.1)
        r_corridor = 0.0
        # Check if enabled in weights
        if bool(corridor_cfg.get("enabled", False)):
            # If enabled, logic is complex. 
            # Phase 20 Cleanup: "Simplify, retain compat keys."
            # We will use what's in ctx.corridor_status
            outside_penalty_weight = float(corridor_cfg.get("outside_penalty_weight", 20.0))
            is_outside = ctx.corridor_status.get("is_outside", False)
            if is_outside:
                 # Simplified penalty if we don't have exact distances
                 r_corridor -= 1.0 * outside_penalty_weight # Rough approx if fields missing
        
        # P7.1 Dir Pref (simplified)
        if corridor_dir_pref_w > 0.0 and "turn_sign" in ctx.corridor_status:
             # Implementation skipped for cleanup unless strictly needed
             pass
        
        # 9. Learnable Lookahead Reward (corner/straight region-aware)
        r_lookahead = 0.0
        lookahead_cfg = self.weights.get("lookahead_reward", {})
        if isinstance(lookahead_cfg, dict) and bool(lookahead_cfg.get("enabled", False)):
            region = float(np.clip(float(ctx.region_weight), 0.0, 1.0))
            la_norm = float(np.clip(float(ctx.lookahead_dist_norm), 0.0, 1.0))
            p = float(lookahead_cfg.get("power", 2.0))
            if not math.isfinite(p) or p <= 0.0:
                p = 2.0

            straight_target = float(np.clip(float(lookahead_cfg.get("straight_target", 0.25)), 0.0, 1.0))
            corner_target = float(np.clip(float(lookahead_cfg.get("corner_target", 0.80)), 0.0, 1.0))
            w_straight = self._safe_abs(lookahead_cfg.get("w_straight", 0.5), fallback=0.5)
            w_corner = self._safe_abs(lookahead_cfg.get("w_corner", 1.0), fallback=1.0)

            err_straight = abs(la_norm - straight_target)
            err_corner = abs(la_norm - corner_target)
            r_lookahead = -((1.0 - region) * w_straight * (err_straight**p) + region * w_corner * (err_corner**p))

        total = float(r_progress + r_track + r_dir + r_time + r_smooth + r_du + r_stall + r_corridor + r_lookahead)

        self.last_progress = progress_now

        components = {
            "progress_diff": float(progress_diff),
            "r_progress": float(r_progress),
            "r_track": float(r_track),
            "r_dir": float(r_dir),
            "r_time": float(r_time),
            "r_smooth": float(r_smooth),
            "r_du": float(r_du),
            "r_stall": float(r_stall),
            "r_corridor": float(r_corridor),
            "r_lookahead": float(r_lookahead),
            "cornerness": float(cornerness),
            "cornerness_raw": float(cornerness_raw),
            "w_track_dyn": float(w_track_eff),
            "w_smooth_dyn": float(w_smooth_dyn),
            "smooth_term": float(smooth_term),
            "lookahead_dist": float(ctx.lookahead_dist),
            "lookahead_dist_norm": float(ctx.lookahead_dist_norm),
            "region_weight": float(ctx.region_weight),
            "total": float(total),
        }
        return total, components
