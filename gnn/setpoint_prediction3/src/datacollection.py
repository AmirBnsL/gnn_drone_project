"""
V3 Data Collection — Fixed velocity frame + APF + integral_pos_err + random shapes.

Key changes from V2 (temp_datacollection.py):
  1. state[2] is GLOBAL velocity → R.T @ state[2] = correct local body frame
  2. APF repulsive forces modify setpoints for obstacle + drone-drone avoidance
  3. integral_pos_err: rolling mean of local_pos_err (3 new features)
  4. Random obstacle shapes: cylinder, box, wall — placed along trajectories
"""
import json, os, multiprocessing
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from collections import deque
import numpy as np
import pybullet as p
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data, InMemoryDataset
from PyFlyt.core import Aviary

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
FORMATION_NAMES = ("a", "rectangle", "triangle", "random_cloud")
FORMATION_TO_ID = {name: idx for idx, name in enumerate(FORMATION_NAMES)}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

# ═══════════════════════════════════════════════════════════════════════════
# APF Obstacle + Drone Avoidance  (NEW in V3)
# ═══════════════════════════════════════════════════════════════════════════
def apf_repulsive_force(drone_xy, obj_xy, obj_radius, influence=3.0, gain=2.0):
    diff = drone_xy - obj_xy
    dist = np.linalg.norm(diff)
    surface = dist - obj_radius
    if surface > influence or surface < 0.01:
        return np.zeros(2)
    mag = gain * (1.0/surface - 1.0/influence) / (surface**2)
    return mag * diff / (dist + 1e-6)

def compute_apf_setpoint(drone_pos, original_sp, obstacles, other_positions,
                          drone_radius=0.3, obs_influence=3.0, drone_influence=2.0):
    force = np.zeros(2)
    for obs in obstacles:
        force += apf_repulsive_force(drone_pos[:2], obs[:2], obs[2], obs_influence, 2.0)
    for op in other_positions:
        force += apf_repulsive_force(drone_pos[:2], op[:2], drone_radius, drone_influence, 1.5)
    mag = np.linalg.norm(force)
    if mag > 2.0:
        force = force / mag * 2.0
    sp = original_sp.copy()
    sp[0] += force[0]
    sp[1] += force[1]
    return sp

# ═══════════════════════════════════════════════════════════════════════════
# Random Obstacle Shapes  (NEW in V3)
# ═══════════════════════════════════════════════════════════════════════════
def sample_trajectory_obstacles(rng, drone_positions, target_positions,
                                 max_obstacles=15, radius_range=(0.3, 2.0),
                                 drone_clearance=2.0, target_clearance=1.5):
    n_obs = int(rng.integers(3, max_obstacles + 1))
    n_drones = len(drone_positions)
    if n_drones == 0:
        return np.zeros((0, 3))
    spawn_xy = drone_positions[:, :2]
    target_xy = target_positions[:, :2] if target_positions.ndim == 2 else target_positions[:, :2]
    accepted = []
    for _ in range(n_obs):
        for _ in range(200):
            idx = rng.integers(0, n_drones)
            s, e = drone_positions[idx], target_positions[idx]
            t = rng.uniform(0.15, 0.85)
            mx = s[0] + t * (e[0] - s[0])
            my = s[1] + t * (e[1] - s[1])
            td = np.array([e[0]-s[0], e[1]-s[1]])
            tl = np.linalg.norm(td)
            if tl > 0.1:
                perp = np.array([-td[1], td[0]]) / tl
                lat = rng.uniform(-3.0, 3.0)
                mx += lat * perp[0]; my += lat * perp[1]
            cr = rng.uniform(radius_range[0], radius_range[1])
            cand = np.array([mx, my])
            if np.any(np.linalg.norm(spawn_xy - cand, axis=1) < drone_clearance + cr):
                continue
            if np.any(np.linalg.norm(target_xy - cand, axis=1) < target_clearance + cr):
                continue
            if accepted and np.any(np.linalg.norm(np.array(accepted)[:,:2] - cand, axis=1) - np.array(accepted)[:,2] - cr < 1.0):
                continue
            accepted.append([mx, my, cr])
            break
    return np.array(accepted, dtype=np.float64) if accepted else np.zeros((0, 3))

def spawn_random_shapes(obstacles, rng, client_id):
    shapes = ["cylinder", "box", "wall"]
    ids = []
    for obs in obstacles:
        x, y, r = obs
        shape = rng.choice(shapes)
        if shape == "cylinder":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=10.0, physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=10.0, rgbaColor=[1,0,0,0.5], physicsClientId=client_id)
            ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x,y,5.0], physicsClientId=client_id))
        elif shape == "box":
            hw, hd = r*rng.uniform(0.5,1.5), r*rng.uniform(0.3,0.8)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw,hd,5.0], physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw,hd,5.0], rgbaColor=[0.8,0.4,0.1,0.5], physicsClientId=client_id)
            orn = p.getQuaternionFromEuler([0,0,rng.uniform(-np.pi,np.pi)])
            ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x,y,5.0], baseOrientation=orn, physicsClientId=client_id))
        else:  # wall
            hw, hd = r*rng.uniform(2.0,4.0), 0.15
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hw,hd,5.0], physicsClientId=client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw,hd,5.0], rgbaColor=[0.5,0.5,0.5,0.6], physicsClientId=client_id)
            orn = p.getQuaternionFromEuler([0,0,rng.uniform(-np.pi,np.pi)])
            ids.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[x,y,5.0], baseOrientation=orn, physicsClientId=client_id))
    return ids

# ═══════════════════════════════════════════════════════════════════════════
# Formations (unchanged from V2)
# ═══════════════════════════════════════════════════════════════════════════
def _formation_a_offsets(n, spacing=2.0):
    off = np.zeros((n,3))
    if n <= 1: return off
    nc = (n//5) if n > 5 else 0; nv = n - nc
    if nv % 2 == 0: nc += 1; nv -= 1
    off[0] = [0,0,0]
    for i in range(1, nv):
        lv = (i+1)//2; s = -1.0 if i%2==1 else 1.0
        off[i,0] = s*lv*spacing; off[i,1] = lv*spacing
    if nc > 0:
        ml = (nv+1)//2; mid = max(1, ml//2)
        lx, rx = -mid*spacing, mid*spacing; w = rx-lx
        for ci in range(nc):
            f = (ci+1)/(nc+1); off[nv+ci,0] = lx+f*w; off[nv+ci,1] = mid*spacing
    off[:,:2] -= np.mean(off[:,:2], axis=0)
    return off

def _formation_rectangle_offsets(n, spacing=2.0):
    off = np.zeros((n,3))
    if n == 0: return off
    side = n*spacing/4.0
    corners = [[-side/2,-side/2],[side/2,-side/2],[side/2,side/2],[-side/2,side/2]]
    for i in range(min(n,4)): off[i,:2] = corners[i]
    if n > 4:
        rem = n-4; dpe = [rem//4+(1 if e<rem%4 else 0) for e in range(4)]
        ci = 4
        for e in range(4):
            sc, ec = np.array(corners[e]), np.array(corners[(e+1)%4])
            for s in range(1, dpe[e]+1):
                off[ci,:2] = sc + s/(dpe[e]+1)*(ec-sc); ci += 1
    off[:,:2] -= np.mean(off[:,:2], axis=0)
    return off

def _formation_triangle_offsets(n, spacing=2.0):
    off = np.zeros((n,3))
    if n == 0: return off
    side = n*spacing/3.0; h = side*np.sqrt(3)/2.0
    corners = [[-side/2,-h/3],[side/2,-h/3],[0,2*h/3]]
    for i in range(min(n,3)): off[i,:2] = corners[i]
    if n > 3:
        rem = n-3; dpe = [rem//3+(1 if e<rem%3 else 0) for e in range(3)]
        ci = 3
        for e in range(3):
            sc, ec = np.array(corners[e]), np.array(corners[(e+1)%3])
            for s in range(1, dpe[e]+1):
                off[ci,:2] = sc + s/(dpe[e]+1)*(ec-sc); ci += 1
    off[:,:2] -= np.mean(off[:,:2], axis=0)
    return off

def generate_random_cloud_setpoints(n, rng, x_b=(-10,10), y_b=(-10,10), z_b=(1,9.5), min_d=1.5):
    accepted = []
    for _ in range(5000):
        if len(accepted) >= n: break
        c = np.array([rng.uniform(*x_b), rng.uniform(*y_b), rng.uniform(*z_b)])
        if not accepted or np.all(np.linalg.norm(np.array(accepted)-c, axis=1) >= min_d):
            accepted.append(c)
    if len(accepted) < n:
        raise RuntimeError(f"Could only place {len(accepted)}/{n} cloud setpoints")
    return np.array(accepted)

def apply_obstacle_avoidance(slots, obstacles, padding=1.0):
    if len(obstacles) == 0: return slots
    safe = np.copy(slots)
    for i in range(len(safe)):
        for obs in obstacles:
            diff = safe[i,:2] - obs[:2]; d = np.linalg.norm(diff)
            if d < obs[2]+padding:
                safe[i,:2] = obs[:2] + diff/(d+1e-6)*(obs[2]+padding)
    return safe

def build_setpoints(dtype, start_pos, start_orn, rng, obstacles=np.empty((0,3))):
    n = len(start_pos)
    if dtype in {"random_cloud","cloud"}:
        tgt = generate_random_cloud_setpoints(n, rng)
        gs = np.zeros((n,3)); gs[:,:2] = tgt[:,:2]
        ss = apply_obstacle_avoidance(gs, obstacles)
        dm = np.linalg.norm(start_pos[:,None,:2]-ss[None,:,:2], axis=2)
        _, ai = linear_sum_assignment(dm)
        sp = np.zeros((n,4))
        for i in range(n):
            sp[i,0]=ss[ai[i],0]; sp[i,1]=ss[ai[i],1]
            sp[i,2]=rng.uniform(-np.pi,np.pi); sp[i,3]=tgt[ai[i],2]
        return sp, ai, np.zeros((n,3))

    fc = np.mean(start_pos[:,:2], axis=0); alt = np.mean(start_pos[:,2])
    if dtype in {"a","formation_a"}: off = _formation_a_offsets(n)
    elif dtype in {"rectangle","formation_rectangle"}: off = _formation_rectangle_offsets(n)
    elif dtype in {"triangle","formation_triangle"}: off = _formation_triangle_offsets(n)
    else:
        sp = np.zeros((n,4)); sp[:,:2] = start_pos[:,:2]+rng.uniform(-5,5,size=(n,2))
        sp[:,2] = rng.uniform(-np.pi,np.pi,size=(n,)); sp[:,3] = rng.uniform(1,5,size=(n,))
        return sp, np.arange(n), np.zeros((n,3))

    gs = np.zeros((n,3)); gs[:,:2] = fc + off[:,:2]
    ss = apply_obstacle_avoidance(gs, obstacles)
    dm = np.linalg.norm(start_pos[:,None,:2]-ss[None,:,:2], axis=2)
    _, ai = linear_sum_assignment(dm)
    sp = np.zeros((n,4))
    for i in range(n):
        sp[i,:2] = ss[ai[i],:2]; sp[i,2] = 0.0; sp[i,3] = alt
    return sp, ai, off

# ═══════════════════════════════════════════════════════════════════════════
# Feature Building  (FIXED velocity frame + integral_pos_err)
# ═══════════════════════════════════════════════════════════════════════════
def build_drone_features(drone, setpoint, formation_one_hot, obstacles,
                          integral_pos_err, state_override=None):
    """Build 35-dim feature frame with CORRECT local velocity + integral_pos_err."""
    if state_override is None:
        state = drone.state
        global_pos = np.array(state[3], copy=True)
        global_euler = np.array(state[1], copy=True)
        global_lin_vel_raw = np.array(state[2], copy=True)   # GLOBAL frame
        global_ang_vel_raw = np.array(state[0], copy=True)   # GLOBAL frame
    else:
        global_pos, global_euler, global_lin_vel_raw, global_ang_vel_raw = state_override

    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(global_euler))).reshape(3,3)

    # FIX: Both velocities are GLOBAL from PyBullet — rotate to local body frame
    local_lin_vel = R.T @ global_lin_vel_raw
    local_ang_vel = R.T @ global_ang_vel_raw
    global_lin_vel = global_lin_vel_raw  # Keep global copy for edge features

    # LiDAR
    num_rays, max_range = 16, 5.0
    lidar = np.full(num_rays, max_range)
    if len(obstacles) > 0:
        dp = global_pos[:2]; yaw = global_euler[2]
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False) + yaw
        rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        oc, orr = obstacles[:,:2], obstacles[:,2]
        for i, ray in enumerate(rays):
            W = oc - dp; t = np.dot(W, ray); hm = t > 0
            if np.any(hm):
                Wh, th, rh = W[hm], t[hm], orr[hm]
                dsq = np.sum(Wh**2,axis=1) - th**2; v = dsq <= rh**2
                if np.any(v):
                    dd = th[v] - np.sqrt(rh[v]**2 - dsq[v]); dd = dd[dd>0]
                    if len(dd)>0: lidar[i] = min(max_range, np.min(dd))

    tgt_pos = np.array([setpoint[0], setpoint[1], setpoint[3]])
    tgt_yaw = setpoint[2]
    global_pos_err = tgt_pos - global_pos
    local_pos_err = R.T @ global_pos_err
    yaw_err = (tgt_yaw - global_euler[2] + np.pi) % (2*np.pi) - np.pi
    dist_floor = global_pos[2]
    dist_ceil = 10.0 - global_pos[2]

    # 35-dim frame: vel(3) + angvel(3) + lidar(16) + pos_err(3) + yaw(1) + floor/ceil(2) + integral(3) + formation(4)
    foh = formation_one_hot if formation_one_hot is not None else np.zeros(4)
    frame = np.concatenate([
        local_lin_vel, local_ang_vel, lidar, local_pos_err, [yaw_err],
        [dist_floor, dist_ceil], integral_pos_err, foh
    ])
    return frame, global_pos, global_euler, global_lin_vel, local_pos_err

def build_edges(positions, global_vels, eulers, comm_radius):
    edges, attrs = [], []
    n = len(positions)
    for i in range(n):
        Ri = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))).reshape(3,3)
        for j in range(n):
            if i == j: continue
            rp = positions[j] - positions[i]; d = np.linalg.norm(rp)
            if d <= comm_radius:
                rv = global_vels[j] - global_vels[i]
                edges.append([i,j])
                attrs.append(np.concatenate([Ri.T@rp, [d], Ri.T@rv]))
    return edges, attrs

def build_next_step_labels(env, active, cur_pos, cur_euler):
    labels = []
    for i, di in enumerate(active):
        np_ = np.array(env.drones[di].state[3], copy=True)
        ne_ = np.array(env.drones[di].state[1], copy=True)
        disp = np_ - cur_pos[i]
        R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(cur_euler[i]))).reshape(3,3)
        ld = R.T @ disp
        dy = (ne_[2] - cur_euler[i][2] + np.pi) % (2*np.pi) - np.pi
        labels.append(np.concatenate([ld, [dy]]).astype(np.float32))
    return labels

# ═══════════════════════════════════════════════════════════════════════════
# Multiprocessing Worker Function
# ═══════════════════════════════════════════════════════════════════════════
def simulate_episode(ep_idx, split, seed, max_steps, save_interval, max_obstacles, communication_radius, integral_window, graphical):
    rng = np.random.default_rng(seed)
    n_drones = int(rng.integers(10, 21))
    
    # Formation
    form_types = ["a","rectangle","triangle","random_cloud"]
    ep_type = str(rng.choice(form_types))
    
    # Spawn
    sp = rng.uniform(-10,10,size=(n_drones,3))
    sp[:,2] = rng.uniform(0.5,5.0,size=(n_drones,))
    so = np.zeros((n_drones,3)); so[:,2] = rng.uniform(-np.pi,np.pi,size=(n_drones,))
    
    setpoints, col_ind, offsets = build_setpoints(ep_type, sp, so, rng)
    
    # Trajectory-aware obstacles with random shapes
    tgt3 = np.column_stack([setpoints[:,0], setpoints[:,1], setpoints[:,3]])
    obstacles = sample_trajectory_obstacles(rng, sp, tgt3, max_obstacles)
    
    # Formation one-hot
    fid = FORMATION_TO_ID.get(ep_type, -1)
    foh = np.zeros(len(FORMATION_NAMES), dtype=np.float32)
    if fid >= 0: foh[fid] = 1.0
    
    env = Aviary(start_pos=sp, start_orn=so, drone_type="quadx", render=graphical)
    env.set_mode(7)
    client = env._client
    
    # Spawn random shapes
    if len(obstacles) > 0:
        spawn_random_shapes(obstacles, rng, client)
        env.register_all_new_bodies()
        
    active = list(range(n_drones))
    
    # APF-modified setpoints
    for i in active:
        others = [np.array(env.drones[j].state[3]) for j in active if j != i]
        mod_sp = compute_apf_setpoint(sp[i], setpoints[i], obstacles, others)
        env.set_setpoint(i, mod_sp)
        
    int_bufs = {i: deque(maxlen=integral_window) for i in range(n_drones)}
    prev_feats = {}
    graphs = []
    
    for step in range(max_steps):
        should_save = (step % save_interval == 0)
        cache = None; cur_pos = None; cur_euler = None
        
        if should_save:
            cache, cur_pos, cur_euler = [], [], []
            for di in active:
                st = env.drones[di].state
                gp = np.array(st[3], copy=True)
                ge = np.array(st[1], copy=True)
                gv = np.array(st[2], copy=True)
                av = np.array(st[0], copy=True)
                cache.append((gp, ge, gv, av))
                cur_pos.append(gp); cur_euler.append(ge)
                
        # APF-modified setpoints each step
        for i, di in enumerate(active):
            others = [np.array(env.drones[dj].state[3]) for dj in active if dj != di]
            mod_sp = compute_apf_setpoint(
                np.array(env.drones[di].state[3]), setpoints[i], obstacles, others)
            env.set_setpoint(di, mod_sp)
            
        env.step()
        
        if should_save:
            labels = build_next_step_labels(env, active, cur_pos, cur_euler)
            states, positions, eulers, gvels = [], [], [], []
            
            for i, di in enumerate(active):
                R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(cache[i][1]))).reshape(3,3)
                tgt_p = np.array([setpoints[i][0], setpoints[i][1], setpoints[i][3]])
                lpe = R.T @ (tgt_p - cache[i][0])
                int_bufs[di].append(lpe.copy())
                ipe = np.mean(list(int_bufs[di]), axis=0)
                
                feat, gp, ge, gv, _ = build_drone_features(
                    env.drones[di], setpoints[i], foh, obstacles, ipe, state_override=cache[i])
                states.append(feat)
                positions.append(gp); eulers.append(ge); gvels.append(gv)
                
            stacked = []
            for i, di in enumerate(active):
                prev = prev_feats.get(di, np.zeros_like(states[i]))
                stacked.append(np.concatenate([states[i], prev]))
            for i, di in enumerate(active):
                prev_feats[di] = states[i].copy()
                
            edges, eattrs = build_edges(np.array(positions), np.array(gvels), np.array(eulers), communication_radius)
            
            x = torch.as_tensor(np.array(stacked), dtype=torch.float32)
            target = torch.as_tensor(np.array(labels), dtype=torch.float32)
            ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long)
            ea = torch.as_tensor(np.array(eattrs), dtype=torch.float32) if eattrs else torch.empty((0,7), dtype=torch.float32)
            
            graph = Data(x=x, target=target, edge_index=ei, edge_attr=ea,
                         pos=torch.as_tensor(np.array(positions), dtype=torch.float32),
                         formation_id=torch.tensor([fid], dtype=torch.long),
                         episode_id=torch.tensor([ep_idx], dtype=torch.long),
                         step_idx=torch.tensor([step], dtype=torch.long),
                         num_drones=torch.tensor([len(active)], dtype=torch.long),
                         obstacles=torch.as_tensor(obstacles, dtype=torch.float32),
                         target_pos=torch.as_tensor(setpoints, dtype=torch.float32))
            graphs.append(graph)
            
    env.disconnect()
    return split, graphs

# ═══════════════════════════════════════════════════════════════════════════
# Main generation loop (Multiprocessed)
# ═══════════════════════════════════════════════════════════════════════════
def generate_dataset(
    num_episodes=10, max_steps=2000, dataset_name="setpoint_V3",
    dataset_type="mixed_formations", save_interval=10,
    max_obstacles=15, communication_radius=10.0,
    seed=12345, graphical=False,
    integral_window=10,
    split_ratios=(0.8, 0.1, 0.1),
    num_workers=None
):
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    datasets_dir = os.path.join(os.path.dirname(script_dir), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    prefix = os.path.join(datasets_dir, f"{dataset_name}_{dataset_type}")

    counts = {"train": int(num_episodes*split_ratios[0]),
              "val": int(num_episodes*split_ratios[1]),
              "test": num_episodes - int(num_episodes*split_ratios[0]) - int(num_episodes*split_ratios[1])}
    
    split_graphs = {s: [] for s in SPLIT_NAMES}
    
    # Prepare tasks
    tasks = []
    global_ep_idx = 0
    for split in SPLIT_NAMES:
        for ep in range(counts[split]):
            eseed = seed + SPLIT_SEED_OFFSETS[split] + ep
            tasks.append({
                'ep_idx': global_ep_idx,
                'split': split,
                'seed': eseed,
            })
            global_ep_idx += 1

    print(f"Starting V3 data generation with {num_workers} parallel workers...")
    
    # Execute in parallel
    worker_fn = partial(
        simulate_episode,
        max_steps=max_steps,
        save_interval=save_interval,
        max_obstacles=max_obstacles,
        communication_radius=communication_radius,
        integral_window=integral_window,
        graphical=graphical
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_fn, t['ep_idx'], t['split'], t['seed']) for t in tasks]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                split, graphs = future.result()
                split_graphs[split].extend(graphs)
                print(f"[{i+1}/{len(tasks)}] Finished episode for split '{split}' ({len(graphs)} frames)")
            except Exception as e:
                print(f"Episode failed: {e}")

    # Save datasets
    for split, graphs in split_graphs.items():
        if not graphs: continue
        data, slices = InMemoryDataset.collate(graphs)
        path = f"{prefix}_{split}.pt"
        torch.save({"data": data, "slices": slices, "split_name": split}, path)
        print(f"Saved {split} → {path} ({len(graphs)} graphs)")

    print("Done!")

if __name__ == "__main__":
    # Ensure this script runs properly on Windows when using multiprocessing
    multiprocessing.freeze_support()
    generate_dataset(num_episodes=10, num_workers=4) # Adjust episodes and workers as needed
