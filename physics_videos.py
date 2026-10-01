"""
Synthetic physics test videos for the V-JEPA 2 world-model demo.

Each scene has a physically correct "normal" version and one or more
"impossible" versions. All versions of a scene share the same first half
(the "past"); the violation only happens in the second half (the "future").

    import physics_videos as pv
    scenes = pv.all_scenes(seed=0)          # {scene: {variant: uint8 array (T, H, W, 3)}}
    pv.save_all("videos/generated")          # write them as .mp4 files (needs opencv)

Run as a script to write every video to videos/generated/:
    python physics_videos.py
"""
import os
import numpy as np

H = W = 256
N_FRAMES = 32            # 2 seconds at 16 fps
FPS = 16
HALF = N_FRAMES // 2
FLOOR = 214
YY, XX = np.mgrid[0:H, 0:W]

RED = (0.85, 0.15, 0.12)
BLUE = (0.15, 0.35, 0.85)
GREEN = (0.15, 0.65, 0.3)
WOOD = (0.55, 0.45, 0.35)


# ------------------------------------------------------------------ drawing
def background():
    img = np.zeros((H, W, 3), np.float32)
    g = np.linspace(0.93, 0.72, H)[:, None]
    img[:] = np.stack([g * 0.95, g * 0.97, g], -1)
    img[FLOOR:] = WOOD
    img[FLOOR:FLOOR + 3] = (0.35, 0.28, 0.22)
    return img


def ball(img, cx, cy, r=14, color=RED, shadow=True):
    if cx is None:
        return img
    if shadow and cy < FLOOR:
        sh = ((XX - cx) / (r * 1.2)) ** 2 + ((YY - (FLOOR + 4)) / 4) ** 2 < 1
        img[sh] *= 0.6
    m = (XX - cx) ** 2 + (YY - cy) ** 2 < r * r
    shade = np.clip(1.2 - np.sqrt((XX - cx + r / 3) ** 2 + (YY - cy + r / 3) ** 2) / (1.6 * r), 0.3, 1.2)
    img[m] = np.array(color)[None, :] * shade[m][:, None]
    return img


def rect(img, x0, y0, x1, y1, color):
    img[int(y0):int(y1), int(x0):int(x1)] = color
    return img


def segment(img, p0, p1, width, color):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    t = np.clip(((XX - p0[0]) * d[0] + (YY - p0[1]) * d[1]) / (d @ d), 0, 1)
    dist2 = (XX - (p0[0] + t * d[0])) ** 2 + (YY - (p0[1] + t * d[1])) ** 2
    img[dist2 < (width / 2) ** 2] = color
    return img


def to_uint8(frames):
    return (np.clip(np.stack(frames), 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ scenes
def pendulum(seed=0):
    """A swinging pendulum. Violations: freezes mid-air / swings ever higher (energy from nowhere)."""
    rng = np.random.default_rng(seed)
    pivot = np.array([128.0, 30.0]); L = rng.uniform(110, 120)
    th, om = rng.uniform(0.8, 1.0) * rng.choice([-1, 1]), 0.0
    w2, dt, sub = 15.0, 1.0 / FPS, 40
    thetas = []
    for _ in range(N_FRAMES):
        thetas.append(th)
        for _ in range(sub):
            om += -w2 * np.sin(th) * dt / sub - 0.05 * om * dt / sub
            th += om * dt / sub
    thetas = np.array(thetas)

    def render(ths):
        frames = []
        for t in ths:
            img = background()
            rect(img, 90, 22, 166, 30, (0.35, 0.35, 0.4))
            bob = pivot + L * np.array([np.sin(t), np.cos(t)])
            segment(img, pivot, bob, 2.5, (0.25, 0.25, 0.28))
            ball(img, bob[0], bob[1], r=15, shadow=False)
            frames.append(img)
        return to_uint8(frames)

    frozen = thetas.copy(); frozen[HALF + 1:] = thetas[HALF]
    growing = thetas.copy()
    growing[HALF + 1:] = np.clip(thetas[HALF + 1:] * np.linspace(1.15, 1.6, N_FRAMES - HALF - 1), -1.25, 1.25)
    return {"normal": render(thetas), "freezes mid-air": render(frozen), "swings higher and higher": render(growing)}


def collision(seed=0):
    """Two balls roll towards each other and bounce apart. Violation: they pass through each other."""
    rng = np.random.default_rng(seed)
    r = 14; y = FLOOR - r
    v = rng.uniform(70, 78)                  # px/s each (they meet just after the halfway point)
    x1, x2 = 30.0, 226.0
    t_hit = ((x2 - x1) - 2 * r) / (2 * v)    # seconds
    ts = np.arange(N_FRAMES) / FPS
    a_n, b_n, a_g, b_g = [], [], [], []
    for t in ts:
        xa, xb = x1 + v * t, x2 - v * t
        a_g.append(xa); b_g.append(xb)
        if t > t_hit:                        # elastic, equal masses: velocities swap
            xa = x1 + v * t_hit - v * (t - t_hit); xb = x2 - v * t_hit + v * (t - t_hit)
        a_n.append(xa); b_n.append(xb)

    def render(A, B):
        return to_uint8([ball(ball(background(), a, y, r, RED), b, y, r, BLUE) for a, b in zip(A, B)])

    return {"normal": render(a_n, b_n), "pass through each other": render(a_g, b_g)}


def occluder(seed=0):
    """A ball rolls behind a screen. Violations: never comes out / comes out as a different object."""
    rng = np.random.default_rng(seed)
    r = 14; y = FLOOR - r
    v = rng.uniform(82, 88)
    xs = 128 + v * (np.arange(N_FRAMES) / FPS - 1.0)     # exactly behind the screen at the halfway frame

    def render(show_after=True, color_after=RED):
        frames = []
        for i, x in enumerate(xs):
            img = background()
            if i <= HALF or show_after:
                ball(img, x, y, r, RED if i <= HALF else color_after)
            rect(img, 80, 110, 176, FLOOR, (0.3, 0.45, 0.55))           # the screen (drawn on top)
            frames.append(img)
        return to_uint8(frames)

    return {"normal": render(), "never comes out": render(show_after=False), "comes out a different colour": render(color_after=GREEN)}


def table_edge(seed=0):
    """A ball rolls off a table and falls. Violation: it keeps going straight in mid-air (cartoon physics)."""
    rng = np.random.default_rng(seed)
    r = 12; top = 130; edge = 125
    v = rng.uniform(88, 95); g = 900.0
    ts = np.arange(N_FRAMES) / FPS
    x_start = 20.0
    t_edge = (edge - x_start) / v
    xs = x_start + v * ts
    ys_n, ys_c = [], []
    for t in ts:
        if t <= t_edge:
            ys_n.append(top - r); ys_c.append(top - r)
        else:
            yy = top - r + 0.5 * g * (t - t_edge) ** 2
            ys_n.append(min(yy, FLOOR - r)); ys_c.append(top - r)

    def render(ys):
        frames = []
        for x, y in zip(xs, ys):
            img = background()
            rect(img, 0, top, edge + 2, top + 8, WOOD); rect(img, 15, top + 8, 25, FLOOR, WOOD); rect(img, edge - 12, top + 8, edge - 2, FLOOR, WOOD)
            ball(img, min(x, W - r - 2), y, r, RED, shadow=y > top)
            frames.append(img)
        return to_uint8(frames)

    return {"normal": render(ys_n), "floats off the edge": render(ys_c)}


def ramp(seed=0):
    """A ball rolls down a slope, speeding up. Violation: halfway down it slows and rolls back UP."""
    rng = np.random.default_rng(seed)
    p0, p1 = np.array([20.0, 80.0]), np.array([236.0, float(FLOOR)])
    d = p1 - p0; Lr = np.linalg.norm(d); u = d / Lr; nrm = np.array([u[1], -u[0]])
    a = 700.0 * u[1] * rng.uniform(0.3, 0.36)         # g sin(alpha), scaled
    ts = np.arange(N_FRAMES) / FPS
    s_n = np.minimum(0.5 * a * ts ** 2 + 5, Lr - 15)
    s_up = s_n.copy()
    t_mid = ts[HALF]; s_mid = s_n[HALF]; v_mid = a * t_mid
    for i in range(HALF + 1, N_FRAMES):
        dt = ts[i] - t_mid
        s_up[i] = max(s_mid + v_mid * dt - 2.5 * a * dt ** 2, 5)    # decelerates too fast, then climbs back
    r = 13

    def render(ss):
        frames = []
        for s in ss:
            img = background()
            poly = (YY > p0[1] + (XX - p0[0]) * d[1] / d[0]) & (XX > p0[0])
            img[poly] = (0.45, 0.42, 0.4)
            c = p0 + s * u + (r + 1) * nrm
            ball(img, c[0], c[1], r, RED, shadow=False)
            frames.append(img)
        return to_uint8(frames)

    return {"normal": render(s_n), "rolls back uphill": render(s_up)}


SCENES = {"pendulum": pendulum, "collision": collision, "hidden ball": occluder, "table edge": table_edge, "ramp": ramp}


def all_scenes(seed=0):
    return {name: f(seed) for name, f in SCENES.items()}


def to16(video):
    """Pick 16 evenly spaced frames (what the models expect)."""
    idx = np.linspace(0, len(video) - 1, 16).round().astype(int)
    return video[idx]


def save_mp4(video, path, fps=FPS, scale=2):
    import cv2
    h, w = video.shape[1] * scale, video.shape[2] * scale
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if vw.isOpened():
            break
    for f in video:
        vw.write(cv2.cvtColor(cv2.resize(f, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    vw.release()


def save_all(folder="videos/generated", seed=0):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for scene, variants in all_scenes(seed).items():
        for variant, vid in variants.items():
            p = os.path.join(folder, f"{scene}__{variant}.mp4".replace(" ", "_"))
            save_mp4(vid, p); paths.append(p)
    return paths


if __name__ == "__main__":
    for p in save_all():
        print("wrote", p)
