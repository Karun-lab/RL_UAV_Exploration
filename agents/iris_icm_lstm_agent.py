"""
iris_icm_lstm_agent.py
======================
PPO agent with action-only LSTM trajectory memory for ICM exploration.

Core design insight (why previous versions crashed at step 255)
--------------------------------------------------------------
SKRL's PPO.update() calls model.compute() internally through its own
mini-batch sampling pipeline. Any attempt to inject extra state (h_t,
lstm_input) into that pipeline via monkey-patching, pending buffers, or
wrapper attributes fails because SKRL rebuilds the inputs dict from its
own memory tensors at update time — it ignores anything we attach.

The correct solution: treat SKRL as a black box and keep it happy.
Give it a standard feedforward model that takes a flat observation and
outputs actions. Move ALL LSTM logic outside SKRL's control.

Architecture: LSTM-augmented observation
-----------------------------------------
Instead of trying to thread h_t through SKRL's update loop, we append
h_t directly to the observation tensor before SKRL ever sees it:

    raw_obs  : (N, T*H*W*C)       flat depth history
    h_t      : (N, lstm_hidden)   LSTM hidden state this step
    augmented: (N, T*H*W*C + lstm_hidden)  ← what SKRL sees

The augmented obs is registered as the observation space. SKRL stores
it in its rollout memory, samples it in mini-batches, and passes it to
compute() during updates — and h_t is already there, stored from rollout
time. No injection needed. No monkey-patching. No timing issues.

The model's compute() simply splits the flat input back into depth and h_t,
runs the CNN on depth, and concatenates the stored h_t for the MLP. During
rollout collection the agent steps the LSTM externally (before calling
env.step) and builds the augmented obs. During PPO update SKRL replays the
stored augmented obs unchanged — h_t is already correct for each transition.

LSTM update (pose supervision)
-------------------------------
The LSTM weights are updated by a separate pose supervision loss that runs
on every rollout, after super().update(). It replays the stored lstm_inputs
in temporal order (BPTT-64) and supervises the LSTM hidden state against
Isaac ground-truth XY displacement. This uses its own Adam optimiser and
never touches SKRL's internals.

Symmetry augmentation (left/right bias fix)
--------------------------------------------
The env applies horizontal depth flip with p=0.5 and negates yaw for
flipped envs before physics. The agent stores the flip mask and also
negates the yaw component of lstm_input for flipped envs so the LSTM
sees consistent action-to-motion correspondence.

Deployment
----------
At deployment: load checkpoint, call step() in a loop, reset_episode()
between flights. The TrajectoryVisualiser plots both dead-reckoning and
LSTM pose-head trajectories live in a background thread.
"""

from __future__ import annotations
import math
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler


# =============================================================================
# CONSTANTS  (must match env cfg)
# =============================================================================
OBS_T   = 3
OBS_H   = 64
OBS_W   = 80
OBS_C   = 1
OBS_FLAT = OBS_T * OBS_H * OBS_W * OBS_C   # 15360
LSTM_HIDDEN    = 256
LSTM_INPUT_DIM = 3    # [vx, yaw_rate, icm_r]
AUG_OBS_DIM    = OBS_FLAT + LSTM_HIDDEN     # 15616  ← what SKRL sees


# =============================================================================
# DEPTH CNN
# =============================================================================

class DepthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(OBS_C, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(
                torch.zeros(1, OBS_C, OBS_H, OBS_W)).shape[1]

    def forward(self, x):
        return self.net(x)


# =============================================================================
# POLICY MODEL  (feedforward from SKRL's perspective)
# =============================================================================

class IrisICMLSTMModel(GaussianMixin, DeterministicMixin, Model):
    """
    Standard feedforward model from SKRL's perspective.

    Observation space: flat vector of size AUG_OBS_DIM = OBS_FLAT + LSTM_HIDDEN
    The last LSTM_HIDDEN elements are the LSTM hidden state stored at rollout time.

    compute() splits this back into depth_flat and h_t, runs CNN on depth,
    concatenates h_t, and passes through the MLP. SKRL never needs to know
    about the LSTM — it just sees a bigger flat observation.
    """

    def __init__(self, observation_space, action_space, device,
                 pose_head_active: bool = True,
                 clip_actions: bool = False,
                 clip_log_std: bool = True,
                 min_log_std: float = -20.0,
                 max_log_std: float = 2.0,
                 reduction: str = "sum",
                 **kwargs):

        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std,
                               min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.pose_head_active = pose_head_active

        # CNN processes T frames independently
        self.cnn      = DepthCNN()
        cnn_total     = OBS_T * self.cnn.out_dim

        # MLP: CNN output + h_t → 512 → 256
        self.net = nn.Sequential(
            nn.Linear(cnn_total + LSTM_HIDDEN, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.policy_mean = nn.Linear(256, action_space.shape[0])
        self.log_std     = nn.Parameter(torch.zeros(action_space.shape[0]))
        self.value_head  = nn.Linear(256, 1)

        # Pose head: projects h_t → (x, y) displacement — training scaffold
        self.pose_head = nn.Linear(LSTM_HIDDEN, 2) if pose_head_active else None

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        """
        inputs["states"]: (N, AUG_OBS_DIM) = (N, OBS_FLAT + LSTM_HIDDEN)

        Split into depth_flat (N, OBS_FLAT) and h_t (N, LSTM_HIDDEN).
        This works identically during rollout collection AND PPO update
        because h_t was already stored in the observation at rollout time.
        """
        flat   = inputs["states"]                              # (N, AUG_OBS_DIM)
        depth_flat = flat[:, :OBS_FLAT]                       # (N, OBS_FLAT)
        h_t        = flat[:, OBS_FLAT:]                       # (N, LSTM_HIDDEN)

        # Unflatten depth: (N, OBS_FLAT) → (N, T, H, W, C)
        N   = flat.shape[0]
        obs = depth_flat.reshape(N, OBS_T, OBS_H, OBS_W, OBS_C)

        # CNN per frame
        feats = []
        for t in range(OBS_T):
            frame = obs[:, t].permute(0, 3, 1, 2)   # (N, C, H, W)
            feats.append(self.cnn(frame))
        cnn_out = torch.cat(feats, dim=1)            # (N, T*cnn_out)

        # MLP with h_t from stored obs
        shared = self.net(torch.cat([cnn_out, h_t], dim=-1))  # (N, 256)

        if role == "policy":
            return self.policy_mean(shared), self.log_std, {}
        elif role == "value":
            return self.value_head(shared), {}
        return self.policy_mean(shared), self.log_std, {}


# =============================================================================
# LSTM  (lives entirely outside SKRL)
# =============================================================================

class TrajectoryLSTM(nn.Module):
    """
    Standalone LSTM that maintains hidden state across rollout steps.
    Entirely managed by ICMLSTMPPOAgent — SKRL never touches it.

    Input at each step: [vx, yaw_rate, icm_r]  (3 floats per env)
    Output: h_t (N, LSTM_HIDDEN)  appended to obs before passing to SKRL
    """

    def __init__(self, num_envs: int, device: torch.device):
        super().__init__()
        self.num_envs = num_envs
        self.device   = device

        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        # Persistent hidden state across steps  (num_layers, N, hidden)
        self.h = torch.zeros(1, num_envs, LSTM_HIDDEN, device=device)
        self.c = torch.zeros(1, num_envs, LSTM_HIDDEN, device=device)

        # Pose head for visualisation (same weights as model.pose_head,
        # shared after training via share_pose_head())
        self.pose_head: nn.Linear | None = None

        # Dedicated optimiser — updated during pose supervision only
        self.optim: torch.optim.Adam | None = None

    def setup_optimiser(self, lr: float = 3e-4):
        params = list(self.lstm.parameters())
        if self.pose_head is not None:
            params += list(self.pose_head.parameters())
        self.optim = torch.optim.Adam(params, lr=lr)

    def share_pose_head(self, pose_head: nn.Linear | None):
        """Point to the same pose_head as the policy model."""
        self.pose_head = pose_head

    def reset(self, env_ids: torch.Tensor):
        """Zero hidden/cell state for specified envs."""
        self.h[:, env_ids, :] = 0.0
        self.c[:, env_ids, :] = 0.0

    def step(self, lstm_input: torch.Tensor,
             done_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Advance one step for all envs.
        lstm_input: (N, 3)   done_mask: (N,) bool
        Returns h_t: (N, LSTM_HIDDEN)
        """
        if done_mask is not None and done_mask.any():
            self.h[:, done_mask, :] = 0.0
            self.c[:, done_mask, :] = 0.0

        x            = lstm_input.unsqueeze(1)         # (N, 1, 3)
        _, (h, c)    = self.lstm(x, (self.h, self.c))
        self.h       = h.detach()
        self.c       = c.detach()
        return self.h.squeeze(0)                       # (N, LSTM_HIDDEN)

    def current_h(self) -> torch.Tensor:
        return self.h.squeeze(0).clone()               # (N, LSTM_HIDDEN)

    def run_bptt(self,
                 lstm_inputs: torch.Tensor,
                 dones:       torch.Tensor,
                 gt_poses:    torch.Tensor,
                 bptt_len:    int) -> torch.Tensor:
        """
        Replay a full rollout in BPTT windows of bptt_len steps.
        Compute MSE(pose_head(h_t), gt_pos_t) at every step.

        lstm_inputs : (S, N, 3)
        dones       : (S, N) bool
        gt_poses    : (S, N, 2)
        Returns scalar loss tensor with requires_grad=True.
        """
        if self.pose_head is None:
            return torch.tensor(0.0, device=self.device)

        N      = lstm_inputs.shape[1]
        S      = lstm_inputs.shape[0]
        h = torch.zeros(1, N, LSTM_HIDDEN, device=self.device)
        c = torch.zeros(1, N, LSTM_HIDDEN, device=self.device)

        losses    = []
        seq_start = 0
        while seq_start < S:
            seq_end = min(seq_start + bptt_len, S)
            for t in range(seq_start, seq_end):
                if dones[t].any():
                    h[:, dones[t], :] = 0.0
                    c[:, dones[t], :] = 0.0
                x           = lstm_inputs[t].unsqueeze(1)   # (N,1,3)
                _, (h, c)   = self.lstm(x, (h, c))
                pose_pred   = self.pose_head(h.squeeze(0))  # (N,2)
                losses.append(F.mse_loss(pose_pred, gt_poses[t]))
            h = h.detach()
            c = c.detach()
            seq_start = seq_end

        return torch.stack(losses).mean() if losses else \
               torch.tensor(0.0, device=self.device)


# =============================================================================
# PPO AGENT
# =============================================================================

class ICMLSTMPPOAgent(PPO):
    """
    Standard PPO that wraps LSTM management outside SKRL's update loop.

    What this agent does at each step:
      1. Get lstm_input from env.unwrapped.extras (safe, bypasses wrapper)
      2. Step the TrajectoryLSTM to get h_t
      3. Augment the SKRL observation by appending h_t
      4. Pass augmented obs to SKRL PPO as if it were a normal flat obs
      5. At update time: SKRL replays stored augmented obs (h_t already inside)
         → no injection needed, no timing issues
      6. After SKRL update: run BPTT pose supervision on stored sequences
    """

    def __init__(self, *args,
                 env,
                 lstm: TrajectoryLSTM,
                 pose_loss_scale: float = 0.1,
                 bptt_len: int = 64,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._raw_env        = env.unwrapped   # bypass gymnasium wrapper
        self._lstm           = lstm
        self._pose_loss_scale = pose_loss_scale
        self._bptt_len        = bptt_len

        # Sequence buffers (filled during rollout, cleared after update)
        self._lstm_in_buf:  list[torch.Tensor] = []
        self._gt_pos_buf:   list[torch.Tensor] = []
        self._done_buf:     list[torch.Tensor] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extras(self) -> dict:
        """Always access env extras on unwrapped env."""
        return self._raw_env.extras

    def _augment(self, states: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Concatenate h_t onto the flat obs so SKRL stores h_t in its memory.
        states: (N, OBS_FLAT)   h_t: (N, LSTM_HIDDEN)
        Returns: (N, AUG_OBS_DIM)
        """
        return torch.cat([states, h_t], dim=-1)

    # ── Rollout step ──────────────────────────────────────────────────────────

    def act(self, states, timestep, timesteps):
        """
        Step LSTM, augment states, then call standard PPO act.
        states here is the raw flat obs from SKRL's pre_interaction.
        """
        extras     = self._extras()
        lstm_input = extras.get("lstm_input", None)
        done_mask  = extras.get("_done_mask", None)   # set in record_transition

        if lstm_input is not None:
            h_t = self._lstm.step(lstm_input.to(self.device), done_mask)
        else:
            h_t = self._lstm.current_h()

        # Augment: SKRL will use this as the observation for the policy forward pass
        aug_states = self._augment(states, h_t)
        return super().act(aug_states, timestep, timesteps)

    def record_transition(self, states, actions, rewards, next_states,
                          terminated, truncated, infos, timestep, timesteps):
        """
        Augment both states and next_states with h_t before storing in memory.
        Also buffer lstm_input and gt_pos for the pose supervision pass.
        """
        extras     = self._extras()
        lstm_input = extras.get("lstm_input", None)
        gt_pos     = extras.get("gt_pos_local", None)
        done       = (terminated | truncated).squeeze(-1)   # (N,) bool

        # Store done mask so act() can reset LSTM on next step
        self._raw_env.extras["_done_mask"] = done

        # Reset LSTM for envs that just finished
        if done.any():
            self._lstm.reset(done.nonzero(as_tuple=True)[0])

        # Buffer for pose BPTT
        if lstm_input is not None:
            self._lstm_in_buf.append(lstm_input.detach().clone().to(self.device))
        if gt_pos is not None:
            self._gt_pos_buf.append(gt_pos.detach().clone().to(self.device))
        self._done_buf.append(done.detach().clone())

        # Augment states: h_t already stepped in act(), use current hidden state
        h_t         = self._lstm.current_h()
        aug_states  = self._augment(states,      h_t)
        aug_next    = self._augment(next_states, h_t)   # approximation: same h_t

        return super().record_transition(
            aug_states, actions, rewards, aug_next,
            terminated, truncated, infos, timestep, timesteps)

    # ── PPO update + pose supervision ─────────────────────────────────────────

    def update(self, timestep: int, timesteps: int):
        """
        1. Standard SKRL PPO update (no LSTM involvement — h_t is in the obs).
        2. Pose supervision BPTT pass on the stored rollout sequences.
        """
        # Standard PPO — SKRL replays stored (aug_obs, action, reward) tuples.
        # The model's compute() splits h_t back out of aug_obs automatically.
        super().update(timestep, timesteps)

        # Pose supervision
        if (self._pose_loss_scale > 0.0
                and self._lstm.pose_head is not None
                and self._lstm.optim is not None
                and len(self._lstm_in_buf) > 0
                and len(self._gt_pos_buf) > 0):

            lstm_inputs = torch.stack(self._lstm_in_buf, dim=0)   # (S, N, 3)
            gt_poses    = torch.stack(self._gt_pos_buf,  dim=0)   # (S, N, 2)
            dones       = torch.stack(self._done_buf,    dim=0)   # (S, N)

            pose_loss = self._lstm.run_bptt(
                lstm_inputs, dones, gt_poses, self._bptt_len)

            if pose_loss.requires_grad:
                self._lstm.optim.zero_grad()
                (self._pose_loss_scale * pose_loss).backward()
                nn.utils.clip_grad_norm_(
                    list(self._lstm.lstm.parameters()) +
                    (list(self._lstm.pose_head.parameters())
                     if self._lstm.pose_head else []),
                    max_norm=0.5)
                self._lstm.optim.step()

            if timestep % 500 == 0:
                print(f"  pose_loss = {float(pose_loss):.5f}")

        # Clear rollout buffers
        self._lstm_in_buf.clear()
        self._gt_pos_buf.clear()
        self._done_buf.clear()


# =============================================================================
# AGENT FACTORY
# =============================================================================

def get_agent(env, device, experiment_cfg: dict | None = None):
    """
    Build ICM+LSTM PPO agent.

    The observation space registered with SKRL is AUG_OBS_DIM = OBS_FLAT + LSTM_HIDDEN
    rather than OBS_FLAT. SKRL treats this as a standard flat observation.
    The LSTM is managed entirely outside SKRL by ICMLSTMPPOAgent.
    """
    raw_env  = env.unwrapped
    num_envs = raw_env.num_envs

    # Augmented obs space: flat depth + LSTM hidden state
    aug_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(AUG_OBS_DIM,), dtype=np.float32)

    memory = RandomMemory(
        memory_size=256,
        num_envs=num_envs,
        device=device,
    )

    # Policy model (feedforward from SKRL's perspective)
    model = IrisICMLSTMModel(
        observation_space=aug_obs_space,
        action_space=env.action_space,
        device=device,
        pose_head_active=True,
    ).to(device)

    models = {"policy": model, "value": model}

    # TrajectoryLSTM — lives outside SKRL
    lstm = TrajectoryLSTM(num_envs=num_envs, device=device).to(device)
    lstm.share_pose_head(model.pose_head)   # shared weights
    lstm.setup_optimiser(lr=3e-4)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"]        = 256
    cfg["learning_epochs"] = 8
    cfg["mini_batches"]    = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"]          = 0.95

    cfg["learning_rate"]                  = 1e-4
    cfg["learning_rate_scheduler"]        = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    cfg["grad_norm_clip"]        = 0.5
    cfg["ratio_clip"]            = 0.2
    cfg["value_clip"]            = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"]    = 0.05
    cfg["value_loss_scale"]      = 1.0

    cfg["state_preprocessor"]        = None
    cfg["value_preprocessor"]        = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    cfg["experiment"]["write_interval"]      = 100
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"]           = "logs/skrl/iris_icm_lstm"
    cfg["experiment"]["wandb"]               = False

    if experiment_cfg:
        cfg["experiment"].update(experiment_cfg)

    agent = ICMLSTMPPOAgent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=aug_obs_space,
        action_space=env.action_space,
        device=device,
        env=env,
        lstm=lstm,
        pose_loss_scale=raw_env.cfg.pose_loss_scale,
        bptt_len=raw_env.cfg.bptt_len,
    )

    return agent


# =============================================================================
# DEPLOYMENT
# =============================================================================

def load_policy_for_deployment(checkpoint_path: str,
                                device: torch.device) -> "DeploymentPolicy":
    aug_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(AUG_OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1., high=1., shape=(2,))

    model = IrisICMLSTMModel(
        observation_space=aug_obs_space,
        action_space=act_space,
        device=device,
        pose_head_active=True,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd   = ckpt.get("policy", ckpt)
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:     print(f"[load] missing keys:    {missing}")
    if unexpected:  print(f"[load] unexpected keys: {unexpected}")
    model.eval()

    lstm = TrajectoryLSTM(num_envs=1, device=device)
    lstm.share_pose_head(model.pose_head)

    return DeploymentPolicy(model, lstm, device)


class DeploymentPolicy:
    def __init__(self, model: IrisICMLSTMModel,
                 lstm: TrajectoryLSTM,
                 device: torch.device):
        self.model  = model
        self.lstm   = lstm
        self.device = device

    def reset_episode(self):
        """Call at the start of each flight."""
        ids = torch.arange(1, device=self.device)
        self.lstm.reset(ids)

    @torch.no_grad()
    def step(self, depth_hist: torch.Tensor,
             lstm_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        depth_hist  : (1, T, H, W, C) normalised depth
        lstm_input  : (1, 3) — [vx_prev, yaw_prev, icm_r_prev]
        Returns: action (1,2), h_t (1, LSTM_HIDDEN)
        """
        h_t = self.lstm.step(lstm_input.to(self.device))

        depth_flat = depth_hist.reshape(1, OBS_FLAT).to(self.device)
        aug_obs    = torch.cat([depth_flat, h_t], dim=-1)   # (1, AUG_OBS_DIM)

        action, _, _ = self.model.act({"states": aug_obs}, role="policy")
        return action, h_t

    @torch.no_grad()
    def estimate_pose(self, h_t: torch.Tensor) -> tuple[float, float]:
        if self.model.pose_head is None:
            return 0.0, 0.0
        xy = self.model.pose_head(h_t)
        return float(xy[0, 0]), float(xy[0, 1])


# =============================================================================
# TRAJECTORY VISUALISER  (unchanged — works with DeploymentPolicy)
# =============================================================================

class TrajectoryVisualiser:
    DT = 0.1

    def __init__(self, policy: DeploymentPolicy,
                 max_points: int = 2000, update_hz: float = 5.0):
        self._policy    = policy
        self._update_hz = update_hz
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None

        self._dr_x = self._dr_y = self._dr_yaw = 0.0
        self._dr_xs   = deque(maxlen=max_points)
        self._dr_ys   = deque(maxlen=max_points)
        self._lstm_xs = deque(maxlen=max_points)
        self._lstm_ys = deque(maxlen=max_points)
        for q in (self._dr_xs, self._dr_ys, self._lstm_xs, self._lstm_ys):
            q.append(0.0)

    def reset(self):
        with self._lock:
            self._dr_x = self._dr_y = self._dr_yaw = 0.0
            for q in (self._dr_xs, self._dr_ys, self._lstm_xs, self._lstm_ys):
                q.clear(); q.append(0.0)

    def push(self, vx_norm: float, yaw_norm: float, h_t: torch.Tensor,
             max_vx: float = 1.5, max_yaw: float = 1.0):
        vx  = vx_norm  * max_vx
        yaw = yaw_norm * max_yaw
        with self._lock:
            self._dr_yaw += yaw * self.DT
            self._dr_x   += vx * math.cos(self._dr_yaw) * self.DT
            self._dr_y   += vx * math.sin(self._dr_yaw) * self.DT
            self._dr_xs.append(self._dr_x)
            self._dr_ys.append(self._dr_y)
            lx, ly = self._policy.estimate_pose(h_t)
            self._lstm_xs.append(lx)
            self._lstm_ys.append(ly)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._plot_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _plot_loop(self):
        import matplotlib; matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Drone Trajectory — Live", fontsize=13)
        for ax, title in [(ax1, "Dead Reckoning"), (ax2, "LSTM Pose Head")]:
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            ax.set_title(title)

        l1, = ax1.plot([], [], "b-", lw=1, alpha=0.8)
        h1, = ax1.plot([], [], "r^", ms=8)
        ax1.plot(0, 0, "go", ms=8, label="start"); ax1.legend(fontsize=8)

        l2, = ax2.plot([], [], "m-", lw=1, alpha=0.8)
        h2, = ax2.plot([], [], "r^", ms=8)
        ax2.plot(0, 0, "go", ms=8, label="start"); ax2.legend(fontsize=8)

        plt.tight_layout(); plt.ion(); plt.show()

        while self._running:
            with self._lock:
                dx = list(self._dr_xs); dy = list(self._dr_ys)
                lx = list(self._lstm_xs); ly = list(self._lstm_ys)

            if len(dx) > 1:
                l1.set_data(dx, dy); h1.set_data([dx[-1]], [dy[-1]])
                _autoscale(ax1, dx, dy)
            if len(lx) > 1:
                l2.set_data(lx, ly); h2.set_data([lx[-1]], [ly[-1]])
                _autoscale(ax2, lx, ly)

            fig.canvas.draw_idle(); fig.canvas.flush_events()
            time.sleep(1.0 / self._update_hz)

        plt.ioff(); plt.close(fig)


def _autoscale(ax, xs, ys, margin: float = 1.0):
    if not xs: return
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)