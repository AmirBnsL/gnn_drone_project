"""
V3 GNN Inference — Reduced goal_gain thanks to better training data.

Key changes from V2:
  1. goal_gain: 0.4 → 0.15 (model handles navigation, goal just provides final pull)
  2. pred_gain: 0.6 → 0.85 (trust the GNN more)
  3. Correct velocity frame: R.T @ state[2] for local body velocity
  4. integral_pos_err feature computed at inference
  5. --simp / --comp obstacle modes
  6. in_channels: 58 → 64 (for 35-dim raw frames)
"""
import argparse, os, time
from collections import deque
import numpy as np
import torch
import pybullet as p
from torch_geometric.data import Data
from PyFlyt.core import Aviary
from model import SetpointGATv2
from dataloader import DatasetNormalizer, engineer_x

DEFAULT_CONFIG = {
    "in_channels": 64,      # V3: 35-dim frames → 32 after engineering × 2 = 64
    "hidden_channels": 64,
    "out_channels": 4,
    "edge_dim": 7,
    "heads": 4,
    "num_layers": 3,
    "dropout": 0.0,
    "physics_hz": 240,
    "ctrl_hz": 48,
    "ctrl_every": 10,
    "max_steps": 2000,
    "comm_radius": 10.0,
    "arena_size": 10.0,
    "ceiling": 10.0,
    "warmup_steps": 120,
    "gain_max": 25.0,
    "d_half": 2.0,
    "yaw_gain": 3.0,
    "dt": 1.0 / 120.0,
    # V3: REDUCED goal_gain — GNN trained on correct data handles navigation
    "pred_gain": 0.85,
    "goal_gain": 0.15,
    "ramp_steps": 200,
    "max_step_xy": 2.0,
    "max_step_z": 1.0,
    "max_step_yaw": 0.4,
    "min_drones": 3,
    "max_drones": 6,
    "min_spawn_dist": 1.5,
    "max_obstacles": 15,
    "obs_radius_range": (0.3, 2.0),
    "num_rays": 16,
    "max_range": 5.0,
    "success_radius": 0.2,
    "collision_radius": 0.35,
    "integral_window": 10,
}

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def random_positions(n, xy_range, z_range, min_dist, max_retries=500):
    positions = []
    for _ in range(n):
        for _ in range(max_retries):
            pos = np.array([np.random.uniform(-xy_range, xy_range),
                            np.random.uniform(-xy_range, xy_range),
                            np.random.uniform(z_range[0], z_range[1])])
            if all(np.linalg.norm(pos - p_) > min_dist for p_ in positions):
                positions.append(pos); break
    return np.array(positions)

def sample_obstacles(start_pos, target_pos, max_obstacles=15, radius_range=(0.3,2.0),
                      drone_clearance=2.0, target_clearance=1.5):
    n_obs = np.random.randint(3, max_obstacles+1)
    n = len(start_pos)
    if n == 0: return np.zeros((0,3))
    sxy, txy = start_pos[:,:2], target_pos[:,:2]
    accepted = []
    for _ in range(n_obs):
        for _ in range(200):
            idx = np.random.randint(n)
            s, e = start_pos[idx], target_pos[idx]
            t = np.random.uniform(0.15, 0.85)
            mx = s[0]+t*(e[0]-s[0]); my = s[1]+t*(e[1]-s[1])
            td = np.array([e[0]-s[0], e[1]-s[1]]); tl = np.linalg.norm(td)
            if tl > 0.1:
                perp = np.array([-td[1],td[0]])/tl
                mx += np.random.uniform(-3,3)*perp[0]; my += np.random.uniform(-3,3)*perp[1]
            cr = np.random.uniform(*radius_range)
            c = np.array([mx,my])
            if np.any(np.linalg.norm(sxy-c,axis=1) < drone_clearance+cr): continue
            if np.any(np.linalg.norm(txy-c,axis=1) < target_clearance+cr): continue
            if accepted and np.any(np.linalg.norm(np.array(accepted)[:,:2]-c,axis=1)-np.array(accepted)[:,2]-cr < 1.0): continue
            accepted.append([mx,my,cr]); break
    return np.array(accepted, dtype=np.float64) if accepted else np.zeros((0,3))

def simple_obstacles(start_pos, max_obstacles=25, xy_bounds=(-15,15), radius_range=(0.3,2.0)):
    n_obs = np.random.randint(0, max_obstacles+1)
    if n_obs == 0: return np.zeros((0,3))
    pxy = start_pos[:,:2]; accepted = []
    for _ in range(n_obs):
        for _ in range(100):
            cx,cy = np.random.uniform(*xy_bounds), np.random.uniform(*xy_bounds)
            cr = np.random.uniform(*radius_range)
            if np.any(np.linalg.norm(pxy-[cx,cy],axis=1) < 3.0+cr): continue
            if accepted and np.any(np.linalg.norm(np.array(accepted)[:,:2]-[cx,cy],axis=1)-np.array(accepted)[:,2]-cr < 1.0): continue
            accepted.append([cx,cy,cr]); break
    return np.array(accepted, dtype=np.float64) if accepted else np.zeros((0,3))

def spawn_pybullet_obstacles(obstacles, client_id):
    ids = []
    for obs in obstacles:
        x,y,r = obs
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=10.0, rgbaColor=[0.8,0.2,0.1,0.6], physicsClientId=client_id)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=10.0, physicsClientId=client_id)
        ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, baseCollisionShapeIndex=col, basePosition=[x,y,5.0], physicsClientId=client_id))
    return ids

def spawn_target_spheres(targets, client_id):
    ids = []
    for t in targets:
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0.1,0.3,1.0,0.8], physicsClientId=client_id)
        ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[t[0],t[1],t[2]], physicsClientId=client_id))
    return ids

# ═══════════════════════════════════════════════════════════════════════════
# Formations (same as V2)
# ═══════════════════════════════════════════════════════════════════════════
def generate_random_cloud_setpoints(n, xy_limit, z_range, min_dist=1.5):
    return random_positions(n, xy_limit, z_range, min_dist)

def formation_a_offsets(n, spacing=2.0):
    off = np.zeros((n,3),dtype=np.float32)
    if n<=1: return off
    nc=(n//5) if n>5 else 0; nv=n-nc
    if nv%2==0: nc+=1; nv-=1
    off[0]=[0,0,0]
    for i in range(1,nv):
        lv=(i+1)//2; s=-1.0 if i%2==1 else 1.0; off[i,0]=s*lv*spacing; off[i,1]=lv*spacing
    if nc>0:
        ml=(nv+1)//2; mid=max(1,ml//2); lx=-mid*spacing; rx=mid*spacing; w=rx-lx
        for ci in range(nc): f=(ci+1)/(nc+1); off[nv+ci,0]=lx+f*w; off[nv+ci,1]=mid*spacing
    off[:,:2]-=np.mean(off[:,:2],axis=0); return off

def formation_rectangle_offsets(n, spacing=2.0):
    off=np.zeros((n,3),dtype=np.float32)
    if n==0: return off
    side=n*spacing/4.0
    corners=[[-side/2,-side/2],[side/2,-side/2],[side/2,side/2],[-side/2,side/2]]
    for i in range(min(n,4)): off[i,:2]=corners[i]
    if n>4:
        rem=n-4; dpe=[rem//4+(1 if e<rem%4 else 0) for e in range(4)]; ci=4
        for e in range(4):
            sc,ec=np.array(corners[e]),np.array(corners[(e+1)%4])
            for s in range(1,dpe[e]+1): off[ci,:2]=sc+s/(dpe[e]+1)*(ec-sc); ci+=1
    off[:,:2]-=np.mean(off[:,:2],axis=0); return off

def formation_triangle_offsets(n, spacing=2.0):
    off=np.zeros((n,3),dtype=np.float32)
    if n==0: return off
    side=n*spacing/3.0; h=side*np.sqrt(3)/2.0
    corners=[[-side/2,-h/3],[side/2,-h/3],[0,2*h/3]]
    for i in range(min(n,3)): off[i,:2]=corners[i]
    if n>3:
        rem=n-3; dpe=[rem//3+(1 if e<rem%3 else 0) for e in range(3)]; ci=3
        for e in range(3):
            sc,ec=np.array(corners[e]),np.array(corners[(e+1)%3])
            for s in range(1,dpe[e]+1): off[ci,:2]=sc+s/(dpe[e]+1)*(ec-sc); ci+=1
    off[:,:2]-=np.mean(off[:,:2],axis=0); return off

def formation_w_offsets(n, spacing=2.0):
    off=np.zeros((n,3),dtype=np.float32)
    if n==0: return off
    pts=np.array([[-2,2],[-1,0],[0,2],[1,0],[2,2]],dtype=np.float32)*spacing
    segs=pts[1:]-pts[:-1]; sl=np.linalg.norm(segs,axis=1); total=float(np.sum(sl))
    if total<=1e-6: return off
    dists=np.linspace(0,total,n,dtype=np.float32); acc=0.0; si=0
    for i,d in enumerate(dists):
        while si<len(sl)-1 and d>acc+sl[si]: acc+=sl[si]; si+=1
        t=0.0 if sl[si]<1e-6 else (d-acc)/sl[si]; off[i,:2]=pts[si]+t*segs[si]
    off[:,:2]-=np.mean(off[:,:2],axis=0); return off

def build_formation_positions(name, n, center_xy, alt, spacing=2.0, xy_limit=5.0):
    if name=="random_cloud": return generate_random_cloud_setpoints(n,xy_limit,(alt-1,alt+1),spacing)
    if name=="a": off=formation_a_offsets(n,spacing)
    elif name=="rectangle": off=formation_rectangle_offsets(n,spacing)
    elif name=="triangle": off=formation_triangle_offsets(n,spacing)
    elif name=="w": off=formation_w_offsets(n,spacing)
    else: off=formation_triangle_offsets(n,spacing)
    pos=np.zeros((n,3),dtype=np.float32); pos[:,:2]=center_xy+off[:,:2]; pos[:,2]=alt
    return pos

# ═══════════════════════════════════════════════════════════════════════════
# Feature Building (V3: correct velocity + integral_pos_err)
# ═══════════════════════════════════════════════════════════════════════════
def compute_lidar(pos, yaw, obstacles, num_rays=16, max_range=5.0):
    rays = np.full(num_rays, max_range)
    if len(obstacles)==0: return rays
    dp=pos[:2]; angles=np.linspace(0,2*np.pi,num_rays,endpoint=False)+yaw
    dirs=np.stack([np.cos(angles),np.sin(angles)],axis=1)
    oc,orr=obstacles[:,:2],obstacles[:,2]
    for i,ray in enumerate(dirs):
        W=oc-dp; t=np.dot(W,ray); hm=t>0
        if np.any(hm):
            Wh,th,rh=W[hm],t[hm],orr[hm]; dsq=np.sum(Wh**2,axis=1)-th**2; v=dsq<=rh**2
            if np.any(v):
                dd=th[v]-np.sqrt(rh[v]**2-dsq[v]); dd=dd[dd>0]
                if len(dd)>0: rays[i]=min(max_range,np.min(dd))
    return rays

def build_single_frame(drone, target, obstacles, integral_err, cfg):
    """Build 35-dim frame with CORRECT local velocity + integral_pos_err."""
    state = drone.state
    global_pos = np.array(state[3], copy=True)
    global_euler = np.array(state[1], copy=True)
    global_lin_vel_raw = np.array(state[2], copy=True)  # GLOBAL frame
    local_ang_vel = np.array(state[0], copy=True)

    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(global_euler))).reshape(3,3)
    local_lin_vel = R.T @ global_lin_vel_raw  # CORRECT local body velocity
    global_lin_vel = global_lin_vel_raw

    lidar = compute_lidar(global_pos, global_euler[2], obstacles, cfg["num_rays"], cfg["max_range"])
    tgt_pos = np.array(target[:3]); tgt_yaw = target[3]
    global_pos_err = tgt_pos - global_pos
    local_pos_err = R.T @ global_pos_err
    yaw_err = (tgt_yaw - global_euler[2] + np.pi) % (2*np.pi) - np.pi
    dist_floor = global_pos[2]; dist_ceil = cfg["ceiling"] - global_pos[2]

    # 35-dim: vel(3)+angvel(3)+lidar(16)+pos_err(3)+yaw(1)+floor/ceil(2)+integral(3)+formation(4)
    frame = np.concatenate([
        local_lin_vel, local_ang_vel, lidar, local_pos_err, [yaw_err],
        [dist_floor, dist_ceil], integral_err, [0,0,0,0]
    ])
    return frame, global_pos, global_euler, global_lin_vel, local_lin_vel, local_pos_err

def build_graph(drones, targets, obstacles, prev_frames, integral_bufs, cfg, device):
    N = len(drones)
    curr_frames,positions,eulers,global_vels,local_vels = [],[],[],[],[]
    local_pos_errs = []

    for i,drone in enumerate(drones):
        ie = np.mean(list(integral_bufs[i]),axis=0) if len(integral_bufs[i])>0 else np.zeros(3)
        frame,pos,euler,gvel,lvel,lpe = build_single_frame(drone,targets[i],obstacles,ie,cfg)
        curr_frames.append(frame); positions.append(pos); eulers.append(euler)
        global_vels.append(gvel); local_vels.append(lvel); local_pos_errs.append(lpe)

    curr_frames=np.array(curr_frames); positions=np.array(positions)
    eulers=np.array(eulers); global_vels=np.array(global_vels); local_vels=np.array(local_vels)

    if prev_frames is None: prev_frames=np.zeros_like(curr_frames)
    x_raw = np.concatenate([curr_frames, prev_frames], axis=1)

    edge_index,edge_attr = [],[]
    for i in range(N):
        Ri=np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))).reshape(3,3)
        for j in range(N):
            if i==j: continue
            rp=positions[j]-positions[i]; d=np.linalg.norm(rp)
            if d<=cfg["comm_radius"]:
                rv=global_vels[j]-global_vels[i]
                edge_index.append([i,j]); edge_attr.append(np.concatenate([Ri.T@rp,[d],Ri.T@rv]))

    x_t=torch.tensor(x_raw,dtype=torch.float32)
    if edge_index:
        ei_t=torch.tensor(edge_index,dtype=torch.long).T
        ea_t=torch.tensor(np.array(edge_attr),dtype=torch.float32)
    else:
        ei_t=torch.zeros((2,0),dtype=torch.long); ea_t=torch.zeros((0,cfg["edge_dim"]),dtype=torch.float32)

    graph=Data(x=x_t,edge_index=ei_t,edge_attr=ea_t).to(device)

    # Update integral buffers
    for i in range(N):
        integral_bufs[i].append(local_pos_errs[i].copy())

    return graph, curr_frames, positions, eulers, local_vels

def pred_to_global_setpoints(pred_scaled, positions, eulers, local_vels, targets, step, cfg):
    dt=cfg["dt"]; ramp=min(1.0,step/max(1,cfg["ramp_steps"]))
    gain_max=cfg["gain_max"]*ramp; yaw_gain=cfg["yaw_gain"]*ramp
    bpg=cfg["pred_gain"]; bgg=cfg["goal_gain"]; dh=cfg["d_half"]
    setpoints=np.zeros((len(positions),4))

    for i in range(len(positions)):
        dx,dy,dz,dyaw = pred_scaled[i]
        lv=local_vels[i]
        cx,cy,cz = dx-lv[0]*dt, dy-lv[1]*dt, dz-lv[2]*dt

        goal_err=targets[i,:3]-positions[i]; gd=np.linalg.norm(goal_err)
        dg=gain_max*np.tanh(gd/dh)
        carrot_local=np.array([cx*dg,cy*dg,cz*dg])
        R=np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(eulers[i]))).reshape(3,3)
        carrot_global=R@carrot_local

        ba=np.clip(gd/dh,0,1); pw=bpg*ba; gw=bgg+bpg*(1-ba)
        yaw=eulers[i][2]; gye=((targets[i,3]-yaw+np.pi)%(2*np.pi))-np.pi
        bxyz=carrot_global*pw+goal_err*gw
        byaw=dyaw*yaw_gain*pw+gye*gw

        bxy=bxyz[:2]; bn=np.linalg.norm(bxy)
        if bn>cfg["max_step_xy"]: bxy=bxy/bn*cfg["max_step_xy"]
        bz=np.clip(bxyz[2],-cfg["max_step_z"],cfg["max_step_z"])
        byaw=np.clip(byaw,-cfg["max_step_yaw"],cfg["max_step_yaw"])

        setpoints[i]=[positions[i][0]+bxy[0], positions[i][1]+bxy[1], yaw+byaw, max(0.5,positions[i][2]+bz)]
    return setpoints

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def run_episode(model, normalizer, cfg, device, formation_arg=None, drones_arg=None, obstacle_mode="comp"):
    num_drones = drones_arg or np.random.randint(cfg["min_drones"], cfg["max_drones"]+1)
    spawn_pos = random_positions(num_drones, cfg["arena_size"]*0.4, (1.0,3.0), cfg["min_spawn_dist"])
    spawn_orn = np.zeros((num_drones,3)); spawn_orn[:,2] = np.random.uniform(-np.pi,np.pi,num_drones)

    formation_choices = ["a","rectangle","triangle","w","random_cloud"]
    waypoints = []
    chosen = formation_arg or np.random.choice(formation_choices)
    cxy = np.array([np.random.uniform(-4,4), np.random.uniform(-4,4)])
    alt = np.random.uniform(2.0, 5.0)
    slots = build_formation_positions(chosen, num_drones, cxy, alt, spacing=2.0, xy_limit=cfg["arena_size"]*0.4)
    sa = np.random.permutation(num_drones)
    waypoints.append({"formation": chosen, "targets": slots[sa]})

    if obstacle_mode == "simp":
        obstacles = simple_obstacles(spawn_pos, cfg["max_obstacles"], (-cfg["arena_size"],cfg["arena_size"]), cfg["obs_radius_range"])
    else:
        obstacles = sample_obstacles(spawn_pos, waypoints[-1]["targets"], cfg["max_obstacles"], cfg["obs_radius_range"])

    print(f"\n{'='*60}")
    print(f"Episode: {num_drones} drones | Formation: {chosen} | {len(obstacles)} obstacles | goal_gain={cfg['goal_gain']}")
    print(f"{'='*60}")

    env = Aviary(start_pos=spawn_pos, start_orn=spawn_orn, drone_type="quadx", render=True,
                 drone_options={"control_hz":120}, physics_hz=240)
    env.set_mode(7)
    client = env._client

    targets = np.zeros((num_drones,4)); targets[:,:3]=waypoints[0]["targets"]; targets[:,3]=0.0
    target_ids = spawn_target_spheres(targets, client)
    obs_ids = spawn_pybullet_obstacles(obstacles, client)
    if obs_ids or target_ids: env.register_all_new_bodies()

    print("[*] Forcing Takeoff Phase...")
    for _ in range(cfg["warmup_steps"]):
        for i in range(num_drones):
            env.set_setpoint(i, np.array([spawn_pos[i][0],spawn_pos[i][1],spawn_orn[i][2],1.5]))
        env.step()

    prev_frames = None
    integral_bufs = [deque(maxlen=cfg["integral_window"]) for _ in range(num_drones)]
    current_setpoints = np.zeros((num_drones,4))
    for i in range(num_drones):
        st=env.drones[i].state; pos=np.array(st[3]); yaw=np.array(st[1])[2]
        current_setpoints[i]=[pos[0],pos[1],yaw,pos[2]]

    print("[*] Starting GNN Inference Loop...")
    for wp_idx, wp in enumerate(waypoints):
        targets[:,:3] = wp["targets"]
        for i,tid in enumerate(target_ids):
            p.resetBasePositionAndOrientation(tid, targets[i,:3], [0,0,0,1], physicsClientId=client)

        for step in range(cfg["max_steps"]):
            if step % cfg["ctrl_every"] == 0:
                drones = [env.drones[i] for i in range(num_drones)]
                for i in range(num_drones):
                    d = targets[i,:2] - np.array(env.drones[i].state[3][:2])
                    if np.linalg.norm(d) > 0.1: targets[i,3] = np.arctan2(d[1],d[0])

                graph, curr_frames, positions, eulers, local_vels = build_graph(
                    drones, targets, obstacles, prev_frames, integral_bufs, cfg, device)
                graph.x = (engineer_x(graph.x) - normalizer.x_mean) / normalizer.x_std
                if graph.edge_attr.numel() > 0:
                    graph.edge_attr = (graph.edge_attr - normalizer.e_mean) / normalizer.e_std

                with torch.no_grad():
                    pred_norm = model(graph.x, graph.edge_index, graph.edge_attr)
                pred_phys = (pred_norm * normalizer.y_scale).cpu().numpy()

                current_setpoints = pred_to_global_setpoints(
                    pred_phys, positions, eulers, local_vels, targets, step, cfg)
                prev_frames = curr_frames

                if step % 100 == 0:
                    d0=np.linalg.norm(positions[0]-targets[0,:3])
                    eg=cfg["gain_max"]*np.tanh(d0/cfg["d_half"])
                    print(f"[Step {step:4d}] Drone 0 → Dist: {d0:.2f}m | DistGain: {eg:.1f}")

            for i in range(num_drones): env.set_setpoint(i, current_setpoints[i])
            env.step(); time.sleep(1.0/cfg["physics_hz"])

            if all(np.linalg.norm(np.array(env.drones[i].state[3])-targets[i,:3])<=cfg["success_radius"] for i in range(num_drones)):
                if step > cfg["warmup_steps"]:
                    print(f"[*] Swarm converged at step {step}!"); break

    print("[+] Mission Complete.")
    try:
        while True: env.step(); time.sleep(0.01)
    except (KeyboardInterrupt, p.error): pass
    finally:
        try: env.disconnect()
        except: pass

def main():
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    parser = argparse.ArgumentParser(description="GNN Drone Swarm Inference V3")
    parser.add_argument("--model", default=os.path.join(ckpt, "best_gatv2.pth"))
    parser.add_argument("--stats", default=os.path.join(ckpt, "normalization_stats.pt"))
    parser.add_argument("--num_drones", type=int, default=None)
    parser.add_argument("--formation", type=str, default=None, choices=["a","rectangle","triangle","w","random_cloud"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simp", action="store_true", help="Simple random-arena obstacles")
    parser.add_argument("--comp", action="store_true", help="Trajectory-aware obstacles (default)")
    args = parser.parse_args()
    obstacle_mode = "simp" if args.simp else "comp"
    if args.seed: np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DEFAULT_CONFIG
    model = SetpointGATv2(in_ch=cfg["in_channels"], hid_ch=cfg["hidden_channels"],
        out_ch=cfg["out_channels"], edge_dim=cfg["edge_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()
    normalizer = DatasetNormalizer.load(args.stats, device=device)
    run_episode(model, normalizer, cfg, device, formation_arg=args.formation,
                drones_arg=args.num_drones, obstacle_mode=obstacle_mode)

if __name__ == "__main__":
    main()
