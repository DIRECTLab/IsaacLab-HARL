1. **Build the largest roster** per team at startup (max over the range). (NO\_Idea)
2. At each reset, **sample active counts** from the given ranges and mark robots **active/inactive** per env. (NO\_Idea)
3. In all loops, **filter by the active mask**:

   * Observations: exclude inactive robots from the K-nearest block. (NO\_Idea)
   * Actions: ignore/hold default targets for inactive. (NO\_Idea)
   * Rewards: zero for inactive; team sums only over active. (NO\_Idea)
4. **Park** inactive robots (teleport up and zero velocities) or disable collisions to avoid interference. (NO\_Idea)

Below are minimal patches on top of the file I shared:

### 1) Config: accept ranges and build superset

```python
@configclass
class AnymalCAdversarialSoccerEnvCfg(DirectMARLEnvCfg):
    def __init__(self, team_robot_counts=None, team_robot_count_ranges=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # New: support ranges -> build superset at max
        team_robot_counts = {
            "team_0": np.random.randint(1, 4),  # e.g. randomize initial count
            "team_1": np.random.randint(1, 4)
            # "team_2": 1
                }
        
        self.padded_dummy_obs_buffer_add  

        # ... then proceed exactly as before using team_robot_counts (the superset).
```

(Claim: per-episode domain randomization of roster size is standard practice → [https://arxiv.org/abs/1703.06907](https://arxiv.org/abs/1703.06907))

### 2) Env: per-episode active mask + roster sampling

Add these to `__init__` (after `super().__init__`):

```python
# Per-robot active mask for each env (True = participates this episode)
self.active_mask = {rid: torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                    for rid in self.robots.keys()}
```

Add helpers:

```python
def _park_inactive(self, env_ids: torch.Tensor):
    """Teleport inactive robots away and zero velocities."""
    off = torch.tensor([0.0, 0.0, 1000.0], device=self.device)
    for rid, robot in self.robots.items():
        m = ~self.active_mask[rid][env_ids]  # those envs where rid is inactive
        if not torch.any(m):
            continue
        idx = env_ids[m]
        # Move high in Z; keep orientation default
        default_root = robot.data.default_root_state[idx].clone()
        default_root[:, :3] = self._terrain.env_origins[idx] + off
        default_root[:, 7:] = 0.0
        robot.write_root_pose_to_sim(default_root[:, :7], idx)
        robot.write_root_velocity_to_sim(default_root[:, 7:], idx)

def _resample_active_teams(self, env_ids: torch.Tensor):
    """Sample active counts per team from cfg.team_robot_count_ranges for each env."""
    for e in env_ids.tolist():
        for team, all_ids in self.cfg.teams.items():
            lo, hi = self.cfg.team_robot_count_ranges[team]
            k = torch.randint(lo, hi + 1, (1,), device=self.device).item()
            # Choose the first k (or random.sample if you prefer)
            active_ids = set(all_ids[:k])
            for rid in all_ids:
                self.active_mask[rid][e] = (rid in active_ids)
    self._park_inactive(env_ids)
```

(Claim: parking inactive bodies avoids unintended contacts → (NO\_Idea))

### 3) Call it in `_reset_idx`

```python
def _reset_idx(self, env_ids: torch.Tensor | None):
    # ... existing reset logic ...
    self._resample_active_teams(env_ids)  # NEW: pick a new roster this episode
    # If you use episode-fixed dummy padding, refresh cache here as before.
```

(Claim: vary roster per episode, not per step, for stationarity → [https://arxiv.org/abs/1703.06907](https://arxiv.org/abs/1703.06907))

### 4) Observations: exclude inactive robots

Replace the start of `_get_observations()` with an active matrix, and pass it into `get_relative_obs`:

```python
robot_id_list = list(self.robots.keys())
# Active mask matrix [N,R]: column order matches robot_id_list
active_mat = torch.stack([self.active_mask[rid] for rid in robot_id_list], dim=1)  # [N, R]
```

Update the inner function signature and distance computation:

```python
def get_relative_obs(ref_robot_id, other_robot_id_list, num_slots: int, active_mat: torch.Tensor):
    # ... build all_pos, rel_pos, etc. as before ...
    # Build a per-env active mask aligned to [N,R]
    rid_to_col = {rid: i for i, rid in enumerate(other_robot_id_list)}
    cols = torch.tensor([rid_to_col[r] for r in [ref_robot_id] + [r for r in other_robot_id_list if r != ref_robot_id]],
                        device=rel_pos.device)
    act = active_mat[:, cols]  # [N,R] -> active robots only

    # Distance: huge where inactive, normal otherwise
    d2 = (rel_pos ** 2).sum(dim=-1)  # [N,R]
    huge = torch.finfo(d2.dtype).max / 4
    d2 = torch.where(act, d2, huge)

    idx = torch.argsort(d2, dim=1)  # [N,R]
    # ... remainder unchanged, then slice to K=min(R, num_slots) and pad ...
```

When writing each robot’s obs vector, zero it where the **reference robot** is inactive:

```python
ref_active = self.active_mask[robot_id].float().unsqueeze(-1)  # [N,1]
obs_vec = obs_vec * ref_active  # inactive agents emit zeros
```

(Claim: masking inactive agents keeps tensor shapes fixed for the learner → (NO\_Idea))

### 5) Actions / control

Only apply control where active; otherwise hold defaults:

```python
def _apply_action(self):
    for rid, robot in self.robots.items():
        m = self.active_mask[rid]  # [N]
        if torch.any(m):
            robot.set_joint_position_target(self.processed_actions[rid][m])
        # For inactive envs, hold default targets
        if torch.any(~m):
            robot.set_joint_position_target(robot.data.default_joint_pos[~m], env_ids=robot._ALL_INDICES[~m])
```

(Claim: ignoring actions for inactive avoids training signal leakage → (NO\_Idea))

### 6) Rewards and team sums

Scale per-robot rewards by the active mask and sum:

```python
r = torch.sum(torch.stack(list(rewards.values())), dim=0)
r = r * self.active_mask[robot_id].float()  # zero if inactive
# ...
team_sum = torch.stack([all_rewards[rid] for rid in self.cfg.teams[team]]).sum(dim=0)
```

(Claim: zeroing inactive rewards keeps team totals consistent across episodes → (NO\_Idea))

---

#### Notes

* With **cap + K-nearest**, obs shape is constant; varying roster only changes which slots are filled by real vs padded entries. (NO\_Idea)
* If you want random *which* robots are active (not just “first k”), pick a random subset in `_resample_active_teams`. (NO\_Idea)
* For multi-env batches, the mask is per-env; code above handles that by indexing `env_ids`. (NO\_Idea)

If you want, I can fold these patches into the full file in the canvas.
