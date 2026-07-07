"""
Neural Cells — a few big amoebas learning to use their bodies.

A handful of large, named cells drift in a petri dish. Each has a tiny neural
brain and improves *within its own lifetime* by trial-and-error (a per-cell
evolution strategy): it keeps brain tweaks that help it reach food, reverts ones
that don't. So you can focus on ONE cell and watch it learn to move.

They engulf food (you see it drawn inward into a vacuole, then digested), heal,
build and excrete waste, and — only when two well-fed cells actually meet — make
a child whose brain blends both parents'. Calm, anatomical, outline-based look.

Run:
  python alife.py                 # live window
  python alife.py --fresh         # ignore saved cells, start over
  python alife.py --train 100000  # fast no-render training, saves to colony.pkl
  python alife.py --headless --steps 600 --gif out.gif
Controls (live): click a cell = focus it · SPACE pause · [ ] speed · S save · ESC quit
"""
import os, sys, math, argparse, pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------- config -----------------------------
DISH_W, DISH_H = 820, 680
PANEL_W = 380
W, H = DISH_W + PANEL_W, DISH_H

IN, H1, H2, OUT = 12, 10, 6, 3
LAYER_SIZES = [IN, H1, H2, OUT]

START_CELLS, MAX_CELLS, MIN_CELLS = 4, 8, 2
FOOD_CAP, FOOD_SPAWN_EVERY = 12, 60

BASE_R = 30.0
ACCEL, DRAG = 0.45, 0.14

E_MAX = 2.0
BASE_COST, MOVE_COST = 0.00050, 0.00055
FOOD_E, FOOD_WASTE = 0.55, 0.35
ENGULF_FRAMES = 22
DIGEST_RATE = 0.006
WASTE_CAP, WASTE_DMG = 1.0, 0.006
HEAL_E, HEAL_RATE = 1.2, 0.004
EX_MIN, EX_COST = 0.20, 0.03
REPRO_E, REPRO_CD = 1.35, 500

EVAL_STEPS, SIGMA, FIT_DECAY = 90, 0.08, 0.96
WALL_PEN = 0.015

AUTOSAVE_STEPS = 1500

# muted palette (no neon)
BG      = (17, 19, 25)
DISH_RING = (34, 40, 50)
MEM_LINE  = (206, 214, 224)
NUC_LINE  = (150, 160, 172)
VAC_LINE  = (120, 175, 150)
FOOD_FILL = (150, 170, 96)
FOOD_LINE = (188, 205, 130)
GRAN      = (176, 190, 180)      # cytoplasmic granules
MITO      = (156, 138, 170)      # mitochondria (muted violet)
NUCLEOLUS = (108, 118, 130)

CONS = "bcdfgklmnprstvwz"
VOWS = "aeiou"


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def make_name(rng):
    n = rng.integers(2, 4)
    s = "".join(rng.choice(list(CONS)) + rng.choice(list(VOWS)) for _ in range(n))
    return s.capitalize()


def rand_brain(rng):
    r = rng.standard_normal
    return {'W1': r((IN, H1)) * 0.6, 'b1': r(H1) * 0.3,
            'W2': r((H1, H2)) * 0.6, 'b2': r(H2) * 0.3,
            'W3': r((H2, OUT)) * 0.6, 'b3': r(OUT) * 0.3}


def copy_brain(b): return {k: v.copy() for k, v in b.items()}


def forward(b, x):
    a1 = np.tanh(x @ b['W1'] + b['b1'])
    a2 = np.tanh(a1 @ b['W2'] + b['b2'])
    o = a2 @ b['W3'] + b['b3']
    return a1, a2, o


# ----------------------------- cell -----------------------------
class Cell:
    def __init__(self, pos, brain, name, hue, energy=1.1):
        self.pos = np.array(pos, np.float32)
        self.vel = np.zeros(2, np.float32)
        self.energy = energy
        self.health = 1.0
        self.waste = 0.0
        self.age = 0
        self.name = name
        self.hue = hue
        self.brain = brain
        self.best_brain = copy_brain(brain)
        self.best_fit = -1e9
        self.cur_fit = 0.0
        self.eval_t = 0
        self.stale = 0         # eval windows since the last improvement
        self.eaten = 0
        self.repro_cd = 0
        self.phase = float(np.random.default_rng().uniform(0, 6.28))
        self.vacuoles = []     # {off:(dx,dy), digest:0..1}
        self.engulf = []       # {p0:(x,y), t:int}
        self.act = None        # [x,a1,a2,out] for the brain panel
        # stable anatomy layout derived from the name (survives save/load)
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % (2 ** 32)
        g = np.random.default_rng(seed)
        self.granules = [(g.uniform(0.12, 0.86), g.uniform(0, 6.283), g.uniform(0.6, 1.5)) for _ in range(26)]
        self.organelles = [(g.uniform(0.30, 0.68), g.uniform(0, 6.283), g.uniform(0, 6.283)) for _ in range(4)]
        self.nuc_off = (float(g.uniform(-0.10, 0.10)), float(g.uniform(-0.10, 0.10)))

    @property
    def radius(self):
        return BASE_R + 10.0 * min(1.0, self.energy / E_MAX)

    def heading(self):
        s = float(np.hypot(*self.vel))
        return (math.atan2(self.vel[1], self.vel[0]) if s > 1e-3 else 0.0), s


# ----------------------------- world -----------------------------
class World:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.births = 0
        self.champ = None      # best brain ever seen — seeds newcomers
        self.champ_fit = 0.0
        self.cells = [self._new_cell() for _ in range(START_CELLS)]
        self.food = [self._rand_pos() for _ in range(FOOD_CAP // 2)]
        self.waste_bits = []   # expelled poop {pos, t}
        self.focal = self.cells[0] if self.cells else None

    def _rand_pos(self):
        return self.rng.uniform([60, 60], [DISH_W - 60, DISH_H - 60]).astype(np.float32)

    def _new_cell(self, pos=None, brain=None, hue=None):
        if brain is None:
            if self.champ is not None:   # newcomers inherit the champion, mutated
                brain = {k: v + self.rng.standard_normal(v.shape) * 0.15
                         for k, v in self.champ.items()}
            else:
                brain = rand_brain(self.rng)
        return Cell(pos if pos is not None else self._rand_pos(),
                    brain,
                    make_name(self.rng),
                    hue if hue is not None else float(self.rng.uniform(0, 1)))

    def _nearest_food(self, cell):
        if not self.food:
            return None, None, None
        fa = np.asarray(self.food, np.float32)
        d = np.hypot(fa[:, 0] - cell.pos[0], fa[:, 1] - cell.pos[1])
        i = int(d.argmin())
        return i, float(d[i]), fa[i].copy()

    def select_nearest(self, x, y):
        if not self.cells: return
        p = np.array([x, y], np.float32)
        self.focal = min(self.cells, key=lambda c: np.hypot(*(c.pos - p)))

    # ---- one tick ----
    def step(self):
        self.steps += 1
        if self.steps % FOOD_SPAWN_EVERY == 0 and len(self.food) < FOOD_CAP:
            self.food.append(self._rand_pos())

        for c in self.cells:
            c.age += 1
            c.phase += 0.12
            if c.repro_cd > 0: c.repro_cd -= 1

            fi, dist_before, fpos = self._nearest_food(c)
            # senses
            x = np.zeros(IN, np.float32)
            if fi is not None:
                v = fpos - c.pos
                x[0] = np.clip(v[0] / 200, -1, 1); x[1] = np.clip(v[1] / 200, -1, 1)
                x[2] = 1.0 / (1.0 + dist_before / 60.0)
            x[3] = c.energy / E_MAX
            x[4] = c.health
            x[5] = c.waste
            x[6] = np.clip(np.hypot(*c.vel) / 5.0, 0, 1)
            x[7] = 1.0
            x[8] = c.pos[0] / DISH_W * 2 - 1      # where am I? (learn wall avoidance)
            x[9] = c.pos[1] / DISH_H * 2 - 1
            x[10] = np.clip(c.vel[0] / 5.0, -1, 1)  # which way am I drifting?
            x[11] = np.clip(c.vel[1] / 5.0, -1, 1)
            a1, a2, o = forward(c.brain, x)
            move = np.tanh(o[:2]); excrete = sigmoid(o[2])
            if c is self.focal:
                c.act = [x, a1, a2, np.concatenate([move, [excrete]])]

            # physics
            c.vel += move * ACCEL
            c.vel *= (1 - DRAG)
            c.pos += c.vel
            at_wall = False
            for ax, hi in ((0, DISH_W), (1, DISH_H)):
                if c.pos[ax] < 40: c.pos[ax] = 40; c.vel[ax] *= -0.4; at_wall = True
                if c.pos[ax] > hi - 40: c.pos[ax] = hi - 40; c.vel[ax] *= -0.4; at_wall = True

            # metabolism
            c.energy -= BASE_COST + MOVE_COST * float(np.abs(move).sum())

            # eat (engulf nearest food if inside membrane)
            ate = False
            if fi is not None and dist_before < c.radius * 0.8:
                c.engulf.append({'p0': self.food[fi].copy(), 't': 0})
                del self.food[fi]
                c.eaten += 1; ate = True

            # advance engulf animations -> vacuoles
            for e in c.engulf[:]:
                e['t'] += 1
                if e['t'] >= ENGULF_FRAMES:
                    ang = self.rng.uniform(0, 6.28); rr = self.rng.uniform(0.1, 0.5)
                    c.vacuoles.append({'off': (math.cos(ang) * rr, math.sin(ang) * rr), 'digest': 0.0})
                    c.engulf.remove(e)
            # digest vacuoles -> energy + waste
            for va in c.vacuoles[:]:
                va['digest'] += DIGEST_RATE
                c.energy = min(E_MAX, c.energy + FOOD_E * DIGEST_RATE)
                c.waste = min(1.5, c.waste + FOOD_WASTE * DIGEST_RATE)
                if va['digest'] >= 1.0: c.vacuoles.remove(va)

            # excrete (neural action, real tradeoff)
            if excrete > 0.5 and c.waste > EX_MIN:
                c.waste = 0.0; c.energy -= EX_COST
                back = -c.vel / (np.hypot(*c.vel) + 1e-3) * c.radius
                self.waste_bits.append({'pos': c.pos + back, 't': 0})

            # heal / waste damage
            if c.waste >= WASTE_CAP: c.health -= WASTE_DMG
            if c.energy > HEAL_E and c.health < 1.0: c.health = min(1.0, c.health + HEAL_RATE)

            # reward for this step -> lifetime learning signal
            r = 0.0
            if fpos is not None:
                dist_after = float(np.hypot(c.pos[0] - fpos[0], c.pos[1] - fpos[1]))
                r += (dist_before - dist_after) * 0.02
            if ate: r += 3.0
            if at_wall: r -= WALL_PEN               # hugging the wall never pays
            c.cur_fit += r

            # per-cell (1+1) evolution strategy: keep helpful tweaks, revert the rest
            c.eval_t += 1
            if c.eval_t >= EVAL_STEPS:
                if c.cur_fit > c.best_fit:
                    c.best_fit = c.cur_fit; c.best_brain = copy_brain(c.brain); c.stale = 0
                else:
                    c.stale += 1
                sig = SIGMA * min(3.0, 1.0 + 0.25 * c.stale)   # stuck? explore harder
                for k in c.brain:
                    c.brain[k] = c.best_brain[k] + self.rng.standard_normal(c.brain[k].shape) * sig
                c.best_fit *= FIT_DECAY
                c.cur_fit = 0.0; c.eval_t = 0

        # remember the best brain ever seen; it seeds replacement cells so
        # learning survives deaths (champ_fit decays so it stays contestable)
        for c in self.cells:
            if c.best_fit > self.champ_fit:
                self.champ_fit = c.best_fit; self.champ = copy_brain(c.best_brain)
        self.champ_fit *= 0.9999

        # deaths
        alive = [c for c in self.cells if c.energy > 0 and c.health > 0]
        if self.focal not in alive:
            self.focal = alive[0] if alive else None
        self.cells = alive

        # reproduction ONLY when two ready cells meet
        n = len(self.cells)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.cells[i], self.cells[j]
                if len(self.cells) >= MAX_CELLS: break
                if a.repro_cd or b.repro_cd: continue
                if a.energy < REPRO_E or b.energy < REPRO_E: continue
                if np.hypot(*(a.pos - b.pos)) < (a.radius + b.radius) * 0.7:
                    self._mate(a, b)

        # keep a minimum so there's always something to watch
        while len(self.cells) < MIN_CELLS:
            self.cells.append(self._new_cell())
        if self.focal is None and self.cells:
            self.focal = self.cells[0]

        # food drifts gently; waste bits age out
        for k in range(len(self.food)):
            self.food[k] += self.rng.uniform(-0.3, 0.3, 2).astype(np.float32)
            # keep food where cells can actually reach it (centers clamp at 40px)
            self.food[k][0] = np.clip(self.food[k][0], 55, DISH_W - 55)
            self.food[k][1] = np.clip(self.food[k][1], 55, DISH_H - 55)
        for wbit in self.waste_bits[:]:
            wbit['t'] += 1
            if wbit['t'] > 120: self.waste_bits.remove(wbit)

    def _mate(self, a, b):
        child = {}
        for k in a.brain:
            mask = self.rng.random(a.brain[k].shape) < 0.5
            child[k] = np.where(mask, a.brain[k], b.brain[k]) + self.rng.standard_normal(a.brain[k].shape) * 0.05
        pos = (a.pos + b.pos) / 2
        hue = (a.hue + b.hue) / 2 % 1.0
        kid = self._new_cell(pos=pos, brain=child, hue=hue)
        kid.energy = 0.8
        a.energy -= 0.5; b.energy -= 0.5
        a.repro_cd = b.repro_cd = kid.repro_cd = REPRO_CD
        self.cells.append(kid); self.births += 1

    # ---- persistence ----
    def save(self, path):
        blob = {'steps': self.steps, 'births': self.births,
                'champ': self.champ, 'champ_fit': self.champ_fit,
                'cells': [{'pos': c.pos, 'vel': c.vel, 'energy': c.energy, 'health': c.health,
                           'waste': c.waste, 'age': c.age, 'name': c.name, 'hue': c.hue,
                           'eaten': c.eaten, 'brain': c.brain, 'best_brain': c.best_brain,
                           'best_fit': c.best_fit} for c in self.cells]}
        with open(path, 'wb') as f: pickle.dump(blob, f)
        return len(self.cells)

    def load(self, path):
        if not os.path.exists(path): return 0
        with open(path, 'rb') as f: blob = pickle.load(f)

        def migrate(b):
            # older saves had fewer senses: pad the new input rows with zeros so
            # the brain behaves exactly as before until learning wires them up
            if b is None: return None
            if b['W1'].shape[0] < IN and b['W1'].shape[1] == H1:
                pad = np.zeros((IN - b['W1'].shape[0], H1))
                b = {k: v.copy() for k, v in b.items()}
                b['W1'] = np.vstack([b['W1'], pad])
            return b

        if blob['cells']:
            probe = migrate(blob['cells'][0]['brain'])
            if (probe['W1'].shape != (IN, H1) or probe['W2'].shape != (H1, H2)
                    or probe['W3'].shape != (H2, OUT)):
                print("saved brain shape differs — starting fresh"); return 0
        self.cells = []
        for d in blob['cells']:
            c = Cell(d['pos'], migrate(d['brain']), d['name'], d['hue'], d['energy'])
            c.vel = d['vel']; c.health = d['health']; c.waste = d['waste']; c.age = d['age']
            c.eaten = d['eaten']; c.best_brain = migrate(d['best_brain']); c.best_fit = d['best_fit']
            self.cells.append(c)
        self.steps = blob['steps']; self.births = blob['births']
        self.champ = migrate(blob.get('champ')); self.champ_fit = blob.get('champ_fit', 0.0)
        self.focal = self.cells[0] if self.cells else None
        return len(self.cells)


# ----------------------------- rendering -----------------------------
def hsv(h, s, v):
    i = int(h * 6) % 6; f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return (int(r * 255), int(g * 255), int(b * 255))


class Renderer:
    def __init__(self):
        try:
            self.f = ImageFont.truetype("arialbd.ttf", 15)
            self.s = ImageFont.truetype("arial.ttf", 12)
            self.big = ImageFont.truetype("arialbd.ttf", 20)
        except Exception:
            self.f = self.s = self.big = ImageFont.load_default()

    def _membrane(self, cx, cy, r, heading, speed, phase, n=44):
        pts = []
        pseudo = min(1.0, speed / 3.0) * r * 0.42
        for i in range(n):
            a = 2 * math.pi * i / n
            rr = r * (1 + 0.10 * math.sin(2 * a + 0.6 * phase)
                        + 0.075 * math.sin(3 * a - 0.4 * phase)
                        + 0.045 * math.sin(5 * a + phase))
            da = (a - heading + math.pi) % (2 * math.pi) - math.pi
            rr += pseudo * math.exp(-(da * da) / 0.22)          # pseudopod reaches toward motion
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts

    def draw_cell(self, dr, c, cx, cy, scale, label=False, diagram=False):
        r = c.radius * scale
        head, speed = c.heading()
        cyto = hsv(c.hue, 0.30, 0.70)
        pts = self._membrane(cx, cy, r, head, speed, c.phase)
        pin = self._membrane(cx, cy, r * 0.72, head, speed * 0.5, c.phase * 1.1)
        # cytoplasm: ectoplasm rim + slightly denser endoplasm
        dr.polygon(pts, fill=cyto + (26,))
        dr.polygon(pin, fill=cyto + (32,))
        # cytoplasmic granules, slowly streaming
        for rf, ang, sz in c.granules:
            aa = ang + c.phase * 0.15
            gx, gy = cx + math.cos(aa) * rf * r * 0.8, cy + math.sin(aa) * rf * r * 0.8
            gr = 1.4 * sz * scale
            dr.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=GRAN + (70,))
        # mitochondria (little capsules)
        for rf, ang, orient in c.organelles:
            ox, oy = cx + math.cos(ang) * rf * r * 0.72, cy + math.sin(ang) * rf * r * 0.72
            dx, dy = math.cos(orient) * 5 * scale, math.sin(orient) * 5 * scale
            dr.line((ox - dx, oy - dy, ox + dx, oy + dy), fill=MITO + (150,), width=max(2, int(3 * scale)))
        # nucleus: nuclear membrane (double) + nucleolus
        nx, ny = cx + c.nuc_off[0] * r, cy + c.nuc_off[1] * r
        nr = r * 0.24
        dr.ellipse((nx - nr, ny - nr, nx + nr, ny + nr), fill=hsv(c.hue, 0.2, 0.5) + (55,), outline=NUC_LINE + (230,), width=2)
        dr.ellipse((nx - nr * 0.82, ny - nr * 0.82, nx + nr * 0.82, ny + nr * 0.82), outline=NUC_LINE + (85,), width=1)
        nol = nr * 0.34
        dr.ellipse((nx - nol, ny - nol, nx + nol, ny + nol), fill=NUCLEOLUS + (215,))
        # food vacuoles (shrink as they digest, food speck inside)
        for va in c.vacuoles:
            vr = (1 - va['digest']) * r * 0.20 + 2
            vx, vy = cx + va['off'][0] * r, cy + va['off'][1] * r
            dr.ellipse((vx - vr, vy - vr, vx + vr, vy + vr), outline=VAC_LINE + (220,), width=2)
            sp = vr * 0.4
            dr.ellipse((vx - sp, vy - sp, vx + sp, vy + sp), fill=FOOD_FILL + (int(180 * (1 - va['digest'])),))
        # contractile vacuole (waste); radiates when full
        if c.waste > 0.05:
            wr = 3 + c.waste * r * 0.18
            wx, wy = cx + r * 0.5, cy - r * 0.5
            dr.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), outline=(192, 152, 122, 210), width=2)
            if c.waste > 0.6:
                for k in range(8):
                    a = k * math.pi / 4
                    dr.line((wx + math.cos(a) * wr, wy + math.sin(a) * wr,
                             wx + math.cos(a) * wr * 1.5, wy + math.sin(a) * wr * 1.5),
                            fill=(192, 152, 122, 120), width=1)
        # membrane outline on top (crisp outer + faint inner = bilayer)
        dr.line(pts + [pts[0]], fill=MEM_LINE + (235,), width=max(2, int(2 * scale)), joint="curve")
        dr.line(pin + [pin[0]], fill=MEM_LINE + (55,), width=1, joint="curve")
        # food being engulfed (drawn moving inward)
        for e in c.engulf:
            f = e['t'] / ENGULF_FRAMES
            if scale == 1:
                ex = c.pos[0] + (e['p0'][0] - c.pos[0]) * (1 - f)
                ey = c.pos[1] + (e['p0'][1] - c.pos[1]) * (1 - f)
            else:
                ex = cx + (e['p0'][0] - c.pos[0]) * (1 - f) * scale
                ey = cy + (e['p0'][1] - c.pos[1]) * (1 - f) * scale
            dr.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), fill=FOOD_FILL + (230,), outline=FOOD_LINE + (255,))
        if label:
            dr.text((cx - r, cy - r - 16), c.name, font=self.s, fill=(210, 218, 228, 255))
        if diagram:
            self._plabels(dr, c, cx, cy, r, head)

    def _plabels(self, dr, c, cx, cy, r, head):
        col = (150, 162, 175, 235); ln = (92, 102, 114, 200)
        def lab(ax, ay, tx, ty, text, right=False):
            dr.line((ax, ay, tx, ty), fill=ln, width=1)
            w = dr.textlength(text, font=self.s) if right else 0
            dr.text((tx - w, ty - 7), text, font=self.s, fill=col)
        px = DISH_W
        nx, ny = cx + c.nuc_off[0] * r, cy + c.nuc_off[1] * r
        lab(nx + r * 0.22, ny, W - 26, ny, "nucleus", right=True)
        lab(cx - r * 0.85, cy - r * 0.35, px + 16, cy - r * 0.55, "membrane")
        ppx, ppy = cx + math.cos(head) * r * 1.02, cy + math.sin(head) * r * 1.02
        lab(ppx, ppy, px + 16, cy + r * 0.75, "pseudopod")
        lab(cx + r * 0.5, cy - r * 0.5, W - 26, cy - r * 0.85, "contractile vacuole", right=True)
        if c.vacuoles:
            va = c.vacuoles[0]; vx, vy = cx + va['off'][0] * r, cy + va['off'][1] * r
            lab(vx, vy, W - 26, cy + r * 0.55, "food vacuole", right=True)

    def render(self, wd: World, focal_forced=None):
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img, "RGBA")
        # dish border
        dr.ellipse((10, 10, DISH_W - 10, DISH_H - 10), outline=DISH_RING, width=2)
        # panel bg
        dr.rectangle((DISH_W, 0, W, H), fill=(13, 15, 20))
        dr.line((DISH_W, 0, DISH_W, H), fill=(40, 46, 56), width=1)

        # food
        for f in wd.food:
            dr.ellipse((f[0] - 4, f[1] - 4, f[0] + 4, f[1] + 4), fill=FOOD_FILL + (220,), outline=FOOD_LINE + (255,))
        # waste bits
        for wb in wd.waste_bits:
            a = max(0, 1 - wb['t'] / 120)
            p = wb['pos']
            dr.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(150, 120, 90, int(160 * a)))
        # cells
        for c in wd.cells:
            self.draw_cell(dr, c, c.pos[0], c.pos[1], 1.0, label=True)
        # focal marker
        if wd.focal is not None:
            r = wd.focal.radius + 10
            fx, fy = wd.focal.pos
            dr.ellipse((fx - r, fy - r, fx + r, fy + r), outline=(120, 200, 235, 180), width=2)

        self._panel(dr, wd)
        self._hud(dr, wd)
        return img

    def _hud(self, dr, wd):
        spf = getattr(wd, "_spf", 1); fps = getattr(wd, "_fps", 0.0)
        dr.text((18, 16), f"step {wd.steps}   cells {len(wd.cells)}   births {wd.births}"
                          f"   speed {spf}x  (~{int(spf * fps)} steps/s)",
                font=self.s, fill=(150, 165, 178, 255))
        dr.text((18, DISH_H - 26),
                "click=focus · SPACE=pause · [ ]=speed · T=turbo · F=fullscreen · S=save · ESC=quit",
                font=self.s, fill=(95, 110, 122, 255))

    def _panel(self, dr, wd):
        px = DISH_W
        c = wd.focal
        if c is None:
            dr.text((px + 20, 20), "no cells", font=self.f, fill=(180, 190, 200, 255)); return
        # name
        dr.text((px + 20, 16), c.name, font=self.big, fill=hsv(c.hue, 0.4, 1.0) + (255,))
        dr.text((px + 20, 42), f"age {c.age}   eaten {c.eaten}", font=self.s, fill=(150, 160, 172, 255))
        # microscope portrait
        pcx, pcy = px + PANEL_W // 2, 150
        dr.ellipse((pcx - 92, pcy - 92, pcx + 92, pcy + 92), outline=(40, 46, 56), width=1)
        self.draw_cell(dr, c, pcx, pcy, 2.3, label=False, diagram=True)
        # stat bars
        y = 260
        for lab, val, col in (("energy", c.energy / E_MAX, (110, 190, 120)),
                              ("health", c.health, (120, 170, 210)),
                              ("waste", c.waste, (200, 160, 120)),
                              ("learned", max(0.0, min(1.0, c.best_fit / 8.0)), (180, 180, 140))):
            dr.text((px + 20, y), lab, font=self.s, fill=(150, 160, 172, 255))
            dr.rectangle((px + 96, y + 2, px + PANEL_W - 24, y + 13), outline=(60, 66, 76), width=1)
            wbar = int((PANEL_W - 24 - 96) * max(0, min(1, val)))
            dr.rectangle((px + 96, y + 2, px + 96 + wbar, y + 13), fill=col + (255,))
            y += 22
        # brain
        self._brain(dr, c, top=370)

    def _brain(self, dr, c, top):
        px = DISH_W
        dr.text((px + 20, top - 22), "BRAIN", font=self.f, fill=(160, 172, 184, 255))
        cols = len(LAYER_SIZES)
        x0, x1 = px + 40, W - 40
        colx = [x0 + (x1 - x0) * k / (cols - 1) for k in range(cols)]
        t, b = top, H - 40
        nodes = [[(colx[k], t + (b - t) * (j + 0.5) / n) for j in range(n)] for k, n in enumerate(LAYER_SIZES)]
        act = c.act
        if act is not None:
            mats = [c.brain['W1'], c.brain['W2'], c.brain['W3']]
            for li, Wm in enumerate(mats):
                mx = max(1e-3, float(np.abs(Wm).max()))
                for i, (ax, ay) in enumerate(nodes[li]):
                    for j, (bx, by) in enumerate(nodes[li + 1]):
                        mag = abs(Wm[i, j]) / mx
                        if mag < 0.12: continue
                        col = (110, 175, 130) if Wm[i, j] > 0 else (190, 120, 110)
                        dr.line((ax, ay, bx, by), fill=col + (int(25 + 120 * mag),), width=1)
        base = [(200, 165, 90)] + [(150, 175, 120)] * (cols - 2) + [(200, 120, 110)]
        for k, layer in enumerate(nodes):
            a = act[k] if act is not None else np.zeros(len(layer))
            for j, (nx, ny) in enumerate(layer):
                lit = min(1.0, abs(float(a[j]))) if j < len(a) else 0.0
                bc = base[k]
                fill = tuple(int(28 + (cc - 28) * (0.3 + 0.7 * lit)) for cc in bc)
                dr.ellipse((nx - 7, ny - 7, nx + 7, ny + 7), fill=fill + (255,), outline=(70, 78, 90, 220))


# ----------------------------- runners -----------------------------
def run_train(steps, save_path, fresh, seed):
    """Fast training: no rendering, just simulate and save. Thousands of steps/s."""
    import time
    wd = World(seed)
    if fresh:
        print("--fresh: new colony")
    else:
        n = wd.load(save_path)
        print(f"resumed {n} cells from {save_path} (step {wd.steps})" if n
              else f"no save at {save_path} — new colony")
    t0 = time.time(); start = wd.steps
    try:
        for t in range(steps):
            wd.step()
            if (t + 1) % 10000 == 0:
                rate = (wd.steps - start) / max(1e-9, time.time() - t0)
                fits = "  ".join(f"{c.name}:{c.best_fit:.1f}" for c in wd.cells)
                print(f"step {wd.steps}  ({rate:.0f}/s)  cells {len(wd.cells)}  "
                      f"births {wd.births}  best-fit {fits}")
            if (t + 1) % 500000 == 0:
                wd.save(save_path)      # checkpoint ~every 2 min
    except KeyboardInterrupt:
        print("interrupted — saving progress")
    wd.save(save_path)
    print(f"saved {len(wd.cells)} cells -> {save_path} (step {wd.steps})")


def run_headless(steps, gif_path, png_every, seed):
    wd = World(seed); rd = Renderer(); frames = []
    for t in range(steps):
        wd.step()
        if t % png_every == 0 or t == steps - 1:
            frames.append(np.asarray(rd.render(wd)))
    if gif_path:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, frames, duration=0.07)
        print(f"wrote {gif_path} ({len(frames)} frames)")
    out = os.path.dirname(gif_path) or "."
    for i, fr in enumerate([frames[0], frames[len(frames) // 2], frames[-1]]):
        Image.fromarray(fr).save(os.path.join(out, f"still_{i}.png"))
    tot = sum(c.eaten for c in wd.cells)
    print(f"final: cells={len(wd.cells)} births={wd.births} eaten(alive)={tot} steps={wd.steps}")


def run_live(seed, speed, save_path, fresh, fullscreen):
    import pygame
    pygame.init()

    def make_screen(fs):
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if fs else pygame.display.set_mode((W, H))

    screen = make_screen(fullscreen)
    pygame.display.set_caption("Neural Cells")
    clock = pygame.time.Clock()
    wd = World(seed); rd = Renderer()
    if fresh:
        print("--fresh: new colony")
    else:
        n = wd.load(save_path)
        print(f"resumed {n} cells from {save_path} (step {wd.steps})" if n
              else f"no save at {save_path} — new colony")

    def save(why=""):
        print(f"saved {wd.save(save_path)} cells -> {save_path} {why}")

    def present(img):
        SW, SH = screen.get_size()
        sc = min(SW / W, SH / H)                      # fit-to-screen, preserve aspect
        tw, th = int(W * sc), int(H * sc)
        ox, oy = (SW - tw) // 2, (SH - th) // 2
        surf = pygame.image.frombuffer(img.tobytes(), (W, H), "RGB")
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.smoothscale(surf, (tw, th)), (ox, oy))
        pygame.display.flip()
        return sc, ox, oy

    paused = False; spf = speed; last = wd.steps; sc, ox, oy = 1.0, 0, 0
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: save("(quit)"); pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: save("(quit)"); pygame.quit(); return
                if e.key == pygame.K_SPACE: paused = not paused
                if e.key == pygame.K_RIGHTBRACKET: spf = min(60, spf + 1)
                if e.key == pygame.K_LEFTBRACKET: spf = max(1, spf - 1)
                if e.key == pygame.K_t: spf = 150 if spf < 150 else speed      # turbo: race evolution
                if e.key == pygame.K_s: save("(manual)")
                if e.key == pygame.K_f:
                    fullscreen = not fullscreen; screen = make_screen(fullscreen)
            if e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = (e.pos[0] - ox) / sc, (e.pos[1] - oy) / sc      # screen -> sim coords
                if 0 <= mx < DISH_W and 0 <= my < H:
                    wd.select_nearest(mx, my)
        if not paused:
            for _ in range(spf): wd.step()
            if wd.steps - last >= AUTOSAVE_STEPS: save("(checkpoint)"); last = wd.steps
        wd._spf = spf; wd._fps = clock.get_fps()
        sc, ox, oy = present(rd.render(wd))
        clock.tick(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--train", type=int, default=0,
                    help="fast headless training: run N steps, save, exit")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--gif", default="out.gif")
    ap.add_argument("--png-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--save", default="colony.pkl")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    a = ap.parse_args()
    if a.train:
        run_train(a.train, a.save, a.fresh, a.seed)
    elif a.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        run_headless(a.steps, a.gif, a.png_every, a.seed)
    else:
        run_live(a.seed, a.speed, a.save, a.fresh, a.fullscreen)
