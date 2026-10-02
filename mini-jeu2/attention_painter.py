"""
attention_painter.py — crée des cartes d'attention réalistes "à la main".

On peint des zones sur l'image (clic gauche = attention, clic droit = non-attention),
puis un pipeline imite les particularités d'un vrai Grad-CAM :
  1. flou multi-échelle        -> plusieurs pics, pas une seule gaussienne
  2. grille basse résolution   -> l'aspect "pixelisé lissé" typique d'un CNN (7x7, 14x14...)
  3. raffinement par l'image   -> l'attention épouse les contours / textures de l'objet
  4. bruit basse fréquence     -> intensité irrégulière + petites taches parasites en fond
  5. normalisation + gamma     -> dynamique de couleurs proche d'un vrai Grad-CAM

Dépendances :
    pip install opencv-python numpy

Utilisation :
    python attention_painter.py chemin/vers/image.jpg

Contrôles :
    clic gauche  (+ glisser) : ajoute de l'attention
    clic droit   (+ glisser) : ajoute de la non-attention
    [ / ]   : taille du pinceau
    , / .   : largeur du dégradé (flou)
    g       : résolution de grille (off / 28 / 14 / 7)
    e       : raffinement par les contours de l'image (on/off)
    n       : niveau de bruit (off / faible / moyen / fort)
    N       : retirer au sort un nouveau bruit
    c       : style d'overlay (classique Grad-CAM / transparent sur zones froides)
    m       : affichage (overlay / carte seule / image seule)
    r       : tout effacer
    s       : sauvegarder (carte + overlay + .npy)
    q ou Échap : quitter
"""

import sys
import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------- #
# Paramètres modifiables
BRUSH = 40                   # rayon initial du pinceau (pixels)
BLUR_SIGMA = 60.0            # largeur du dégradé (pixels, à pleine résolution)
COLORMAP = cv2.COLORMAP_JET  # essaie aussi COLORMAP_TURBO, COLORMAP_INFERNO
OVERLAY_MAX_ALPHA = 0.75     # style "transparent" : opacité max sur les zones chaudes
CLASSIC_ALPHA = 0.4          # style "classique" : 0.4*carte + 0.6*image (comme pytorch-grad-cam)
WORK_SIZE = 384              # le calcul se fait à cette taille puis est agrandi (rapide)
GRID_CHOICES = [0, 28, 14, 7]            # 0 = pas de grille
NOISE_LEVELS = [0.0, 0.15, 0.30, 0.50]
EDGE_EPS = 0.01              # plus petit = l'attention colle davantage aux contours
GAMMA = 0.85                 # <1 : étale un peu plus les zones moyennes (cyan/vert/jaune)
# ----------------------------------------------------------------------------- #


def guided_filter(guide, src, r, eps):
    """Filtre guidé (He et al.) : lisse `src` en respectant les contours de `guide`."""
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.blur(x, k)
    m_i, m_p = box(guide), box(src)
    cov = box(guide * src) - m_i * m_p
    var = box(guide * guide) - m_i * m_i
    a = cov / (var + eps)
    b = m_p - a * m_i
    return box(a) * guide + box(b)


def smooth_noise(rng, shape, cells):
    """Bruit basse fréquence dans [0, 1] : une petite grille aléatoire agrandie en bicubique."""
    h, w = shape
    ch = max(2, int(round(cells * h / max(h, w))))
    cw = max(2, int(round(cells * w / max(h, w))))
    n = rng.random((ch, cw)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n -= n.min()
    return n / (n.max() + 1e-6)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Impossible de charger l'image : {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    scale = min(1.0, 1400 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # --- Pré-calculs à résolution de travail -------------------------------- #
    ws = min(1.0, WORK_SIZE / max(h, w))
    ww, wh = max(8, int(w * ws)), max(8, int(h * ws))
    small = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    structure = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (0, 0), sigmaX=max(ww, wh) / 40)
    structure /= structure.max() + 1e-6          # 0..1 : zones texturées / contrastées

    rng = np.random.default_rng()

    state = {
        "pos": np.zeros((h, w), np.float32),
        "neg": np.zeros((h, w), np.float32),
        "brush": BRUSH,
        "sigma": BLUR_SIGMA,
        "drawing": None,
        "last": None,
        "view": 0,        # 0 = overlay, 1 = carte seule, 2 = image seule
        "grid": 2,        # index dans GRID_CHOICES (14x14 par défaut)
        "edge": True,
        "noise": 2,       # index dans NOISE_LEVELS
        "classic": True,
        "dirty": True,
        "heat": np.zeros((h, w), np.float32),
        "frame": img.copy(),
    }

    def new_noise():
        state["n1"] = smooth_noise(rng, (wh, ww), 10)      # modulation de l'intensité
        n2 = smooth_noise(rng, (wh, ww), 22)               # taches parasites éparses
        state["n2"] = np.clip((n2 - 0.65) / 0.35, 0, 1)

    new_noise()

    # --- Pinceau ------------------------------------------------------------ #
    def stamp(layer, x, y):
        r = state["brush"]
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = ((xx - x) ** 2 + (yy - y) ** 2) / float(r * r)
        # Noyau doux + "pression" aléatoire -> pics irréguliers, plus organiques
        patch = np.exp(-2.5 * d2) * (d2 <= 1.0) * rng.uniform(0.7, 1.0)
        roi = state[layer][y0:y1, x0:x1]
        np.maximum(roi, patch.astype(np.float32), out=roi)

    def stamp_line(layer, x, y):
        last = state["last"]
        if last is None:
            pts = [(x, y)]
        else:
            lx, ly = last
            dist = max(abs(x - lx), abs(y - ly))
            step = max(1, state["brush"] // 3)
            n = max(1, dist // step)
            pts = [(int(lx + (x - lx) * t / n), int(ly + (y - ly) * t / n)) for t in range(1, n + 1)]
        for px, py in pts:
            stamp(layer, px, py)
        state["last"] = (x, y)
        state["dirty"] = True

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"], state["last"] = "pos", None
            stamp_line("pos", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing"], state["last"] = "neg", None
            stamp_line("neg", x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            stamp_line(state["drawing"], x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["drawing"], state["last"] = None, None

    win = "attention_painter"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    # --- Pipeline "réaliste" ------------------------------------------------ #
    def normalize(a):
        a = np.clip(a, 0, None)
        m = float(a.max())
        return a / m if m > 1e-6 else a

    def compute_heatmap():
        pos = cv2.resize(state["pos"], (ww, wh), interpolation=cv2.INTER_AREA)
        neg = cv2.resize(state["neg"], (ww, wh), interpolation=cv2.INTER_AREA)
        signed = pos - neg
        sigma = max(1.0, state["sigma"] * ws)

        # 1. Flou multi-échelle : un cœur net + des halos plus larges
        blur = lambda a, k: cv2.GaussianBlur(a, (0, 0), sigmaX=max(0.5, sigma * k))
        heat = 0.30 * blur(signed, 0.4) + 0.45 * blur(signed, 1.0) + 0.25 * blur(signed, 2.2)
        heat = normalize(heat)

        # 2. Grille basse résolution (signature d'un CNN : 7x7, 14x14...)
        g = GRID_CHOICES[state["grid"]]
        if g:
            gh = max(2, int(round(g * wh / max(ww, wh))))
            gw = max(2, int(round(g * ww / max(ww, wh))))
            low = cv2.resize(heat, (gw, gh), interpolation=cv2.INTER_AREA)
            heat = cv2.resize(low, (ww, wh), interpolation=cv2.INTER_LINEAR)

        # 3. Raffinement par l'image : l'attention suit les contours et les zones texturées
        if state["edge"] and heat.max() > 1e-6:
            r = max(2, int(0.05 * max(ww, wh)))
            refined = guided_filter(gray, heat, r, EDGE_EPS)
            heat = 0.35 * heat + 0.65 * np.clip(refined, 0, None)
            heat *= 0.55 + 0.45 * structure
            heat = normalize(heat)

        # 4. Bruit : modulation irrégulière + taches parasites faibles (comme en vrai)
        amt = NOISE_LEVELS[state["noise"]]
        if amt > 0:
            heat = heat * (1.0 + amt * (2.0 * state["n1"] - 1.0))
            heat = heat + 0.25 * amt * state["n2"]
            heat = normalize(heat)

        # 5. Gamma + agrandissement lisse à la taille de l'image
        heat = np.power(heat, GAMMA)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(heat, 0.0, 1.0).astype(np.float32)

    def make_overlay(heat):
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), COLORMAP)
        if state["classic"]:
            # Style Grad-CAM habituel : la carte couvre toute l'image (fond bleu)
            out = img.astype(np.float32) * (1 - CLASSIC_ALPHA) + colored.astype(np.float32) * CLASSIC_ALPHA
        else:
            alpha = (heat * OVERLAY_MAX_ALPHA)[..., None]
            out = img.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def render(heat):
        if state["view"] == 2:
            return img.copy()
        if state["view"] == 1:
            return cv2.applyColorMap((heat * 255).astype(np.uint8), COLORMAP)
        return make_overlay(heat)

    def save():
        heat = state["heat"]
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.dirname(os.path.abspath(path))
        p_map = os.path.join(out_dir, f"{base}_attention.png")
        p_over = os.path.join(out_dir, f"{base}_overlay.png")
        p_npy = os.path.join(out_dir, f"{base}_attention.npy")
        cv2.imwrite(p_map, (heat * 255).astype(np.uint8))
        cv2.imwrite(p_over, make_overlay(heat))
        np.save(p_npy, heat)
        print(f"Sauvegardé :\n  {p_map}\n  {p_over}\n  {p_npy}")

    print(__doc__)
    while True:
        if state["dirty"]:
            state["heat"] = compute_heatmap()
            state["frame"] = render(state["heat"])
            state["dirty"] = False

        g = GRID_CHOICES[state["grid"]]
        cv2.setWindowTitle(
            win,
            f"pinceau={state['brush']}  flou={state['sigma']:.0f}  "
            f"grille={g or 'off'}  contours={'on' if state['edge'] else 'off'}  "
            f"bruit={NOISE_LEVELS[state['noise']]:.2f}  "
            f"vue={['overlay', 'carte', 'image'][state['view']]}",
        )
        cv2.imshow(win, state["frame"])

        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        elif k == ord("["):
            state["brush"] = max(4, state["brush"] - 4)
        elif k == ord("]"):
            state["brush"] = min(400, state["brush"] + 4)
        elif k == ord(","):
            state["sigma"] = max(2.0, state["sigma"] - 5)
        elif k == ord("."):
            state["sigma"] = min(300.0, state["sigma"] + 5)
        elif k == ord("g"):
            state["grid"] = (state["grid"] + 1) % len(GRID_CHOICES)
        elif k == ord("e"):
            state["edge"] = not state["edge"]
        elif k == ord("n"):
            state["noise"] = (state["noise"] + 1) % len(NOISE_LEVELS)
        elif k == ord("N"):
            new_noise()
        elif k == ord("c"):
            state["classic"] = not state["classic"]
        elif k == ord("m"):
            state["view"] = (state["view"] + 1) % 3
        elif k == ord("r"):
            state["pos"][:] = 0
            state["neg"][:] = 0
        elif k == ord("s"):
            save()
            continue
        state["dirty"] = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()