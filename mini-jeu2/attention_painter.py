"""
attention_painter.py — crée des cartes d'attention "à la main".

Idée : pas besoin d'entraîner un classifieur. On peint des zones sur l'image
(clic gauche = attention, clic droit = non-attention), puis un flou gaussien
transforme ces taches en un joli dégradé continu, comme la sortie d'un Grad-CAM.

Dépendances :
    pip install opencv-python numpy

Utilisation :
    python attention_painter.py chemin/vers/image.jpg

Contrôles :
    clic gauche  (+ glisser) : ajoute de l'attention (zone "importante")
    clic droit   (+ glisser) : ajoute de la non-attention (zone "ignorée")
    [ / ]                    : diminue / augmente la taille du pinceau
    , / .                    : diminue / augmente le flou (netteté du dégradé)
    m                       : change l'affichage (overlay / carte seule / image seule)
    r                       : tout effacer
    s                       : sauvegarder (carte + overlay + .npy)
    q ou Échap              : quitter
"""

import sys
import os
import cv2
import numpy as np

# ----------------------------------------------------------------------------- #
# Paramètres modifiables
BRUSH = 40           # rayon initial du pinceau, en pixels
BLUR_SIGMA = 80.0    # sigma du flou gaussien -> largeur du dégradé
OVERLAY_MAX_ALPHA = 0.75   # opacité max de la couleur sur les zones chaudes
COLORMAP = cv2.COLORMAP_JET  # essaie aussi COLORMAP_TURBO, COLORMAP_INFERNO
# ----------------------------------------------------------------------------- #


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Impossible de charger l'image : {path}")
        sys.exit(1)

    # On réduit les très grandes images pour que l'édition reste fluide.
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    state = {
        "pos": np.zeros((h, w), np.float32),   # coups de pinceau "attention"
        "neg": np.zeros((h, w), np.float32),   # coups de pinceau "non-attention"
        "brush": BRUSH,
        "sigma": BLUR_SIGMA,
        "drawing": None,   # 'pos', 'neg' ou None
        "view": 0,         # 0 = overlay, 1 = carte seule, 2 = image seule
    }

    def stamp(layer, x, y):
        # Pinceau à bords doux : un disque plein puis un léger flou local.
        cv2.circle(state[layer], (x, y), state["brush"], 1.0, -1, lineType=cv2.LINE_AA)

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = "pos"
            stamp("pos", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["drawing"] = "neg"
            stamp("neg", x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            stamp(state["drawing"], x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            state["drawing"] = None

    win = "attention_painter"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def compute_heatmap():
        # Différence des deux couches, puis flou -> dégradé continu.
        signed = state["pos"] - state["neg"]
        heat = cv2.GaussianBlur(signed, (0, 0), sigmaX=state["sigma"], sigmaY=state["sigma"])
        heat = np.clip(heat, 0, None)          # on ne garde que l'attention positive
        peak = float(heat.max())
        if peak > 1e-6:
            heat = heat / peak                 # normalisation 0..1
        return heat

    def make_overlay(heat):
        # Overlay : alpha proportionnel à l'attention -> les zones froides
        # laissent voir l'image d'origine.
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), COLORMAP)
        alpha = (heat * OVERLAY_MAX_ALPHA)[..., None]
        out = img.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
        return out.astype(np.uint8)

    def render(heat):
        if state["view"] == 2:
            return img.copy()
        if state["view"] == 1:
            return cv2.applyColorMap((heat * 255).astype(np.uint8), COLORMAP)
        return make_overlay(heat)

    def save():
        heat = compute_heatmap()
        base = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.dirname(os.path.abspath(path))
        p_map = os.path.join(out_dir, f"{base}_attention.png")
        p_over = os.path.join(out_dir, f"{base}_overlay.png")
        p_npy = os.path.join(out_dir, f"{base}_attention.npy")
        # On sauvegarde des images propres, sans le texte d'interface (pinceau/flou).
        cv2.imwrite(p_map, (heat * 255).astype(np.uint8))
        cv2.imwrite(p_over, make_overlay(heat))
        np.save(p_npy, heat)
        print(f"Sauvegardé :\n  {p_map}\n  {p_over}\n  {p_npy}")

    print(__doc__)
    while True:
        heat = compute_heatmap()
        frame = render(heat)

        # Les infos vont dans la barre de titre de la fenêtre, jamais sur l'image.
        cv2.setWindowTitle(
            win,
            f"attention_painter  |  pinceau={state['brush']}  "
            f"flou={state['sigma']:.0f}  vue={['overlay', 'carte', 'image'][state['view']]}",
        )
        cv2.imshow(win, frame)

        k = cv2.waitKey(16) & 0xFF
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
        elif k == ord("m"):
            state["view"] = (state["view"] + 1) % 3
        elif k == ord("r"):
            state["pos"][:] = 0
            state["neg"][:] = 0
        elif k == ord("s"):
            save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
