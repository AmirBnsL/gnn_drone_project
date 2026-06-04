/** GLB mesh target max dimension (m) — matches visible fallback size */
export const DRONE_GLB_SIZE_M = 0.9;
export const DRONE_FALLBACK_SIZE_M = 0.9;
/** Legacy training-scale reference (digit spacing context) */
export const DRONE_TARGET_SIZE_M = 0.1;
/** Visual-only obstacle sphere scale (physics radius unchanged in API) */
export const OBSTACLE_MESH_VIS_SCALE = 0.5;
export const FORMATION_SPACING = 2.0;
export const ARENA_HALF = 10;
export const MIN_DRONES = 5;
export const MAX_DRONES = 20;
export const DEFAULT_DRONES = 12;
/** Decentral comm radius (m) — keep in sync with backend viz_sim_config.COMM_RADIUS */
export const COMM_RADIUS_M = 10.0;
/** Propeller spin rate (rad/s at 1x playback); scaled by timeline speed */
export const PROPELLER_SPIN_RAD_PER_S = 110;
/** Arm length as fraction of drone size for procedural rotors */
export const PROPELLER_ARM_FRAC = 0.38;
