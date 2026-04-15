from ultralytics import YOLO
# ================================================================
# MEDICAL WASTE DETECTION SYSTEM — PROFESSIONAL EDITION
# YOLOv8 Segmentation | Monte Carlo UQ | Heatmap | Tracking
# ================================================================
#
# ✅ NEW FEATURES OVER ORIGINAL:
#   [UI]        Full dark-themed canvas: top status bar + side panel
#   [UI]        Rounded bounding boxes, hazard-colored overlays
#   [UI]        Per-hazard color system (RED=HIGH, AMBER=MED, GREEN=LOW)
#   [GRAPHS]    Real-time EMA-smoothed risk + uncertainty graphs
#   [HEATMAP]   Spatial detection heatmap (toggle H, reset R)
#   [TRACKING]  Lightweight IoU-based centroid tracker (persistent IDs)
#   [ALERTS]    Pulsing red border + banner on HIGH risk or uncertainty
#   [PERF]      MC-Dropout every N frames only (not every frame)
#   [PERF]      FP16 half-precision inference on GPU
#   [PERF]      IMG_SIZE reduced 640→480 for better FPS
#   [PERF]      torch.backends.cudnn.benchmark enabled
#   [PERF]      EMA smoothing replaces noisy per-frame values
#   [SUMMARY]   Matplotlib session summary: class freq + risk dist pie
#
# pip install ultralytics tqdm torch torchvision opencv-python numpy matplotlib
# ================================================================

import os, sys, time, warnings
import cv2
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm


warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ CONFIGURATION  — paths kept exactly as original
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE_PATH   = r"C:\Users\gahan\Desktop\dbhdsnet_project"
RUNS_DIR    = os.path.join(BASE_PATH, "runs")
RUN_NAME    = "medwaste_seg_v16"
BEST_MODEL  = os.path.join(RUNS_DIR, "segment", RUN_NAME, "weights", "best.pt")

CONF_THRESH  = 0.40
IOU_THRESH   = 0.45
IMG_SIZE     = 480          # ⬇ 640→480: saves ~30% VRAM, boosts FPS

MODE         = "webcam"     # "webcam" | "video" | "image"
INPUT_PATH   = r"C:\path\to\your\image_or_video.jpg"
WEBCAM_INDEX = 0

SAVE_OUTPUT  = True
OUTPUT_DIR   = os.path.join(BASE_PATH, "realtime_output")

# ── Performance knobs ────────────────────────────────────────
MC_PASSES     = 3     # forward passes for MC-Dropout uncertainty
MC_INTERVAL   = 20    # run MC only every N frames  (KEY PERF WIN)
RISK_EMA_K    = 0.18  # EMA smoothing coefficient for risk score
UQ_EMA_K      = 0.20  # EMA smoothing coefficient for uncertainty
GRAPH_LEN     = 120   # rolling history length for mini-graphs

# ── Heatmap ──────────────────────────────────────────────────
HM_SCALE      = 4     # internal downsample factor (speed)
HM_DECAY      = 0.97  # per-frame decay (<1 fades old detections)
HM_ALPHA      = 0.45  # overlay blend strength

# ── Layout (px) ──────────────────────────────────────────────
SIDE_W        = 232   # right panel width
TOP_H         = 70    # top status bar height
GRAPH_H       = 98    # height of each mini-graph
TRACKER_LOST  = 15    # frames before track is dropped
TRACKER_IOU   = 0.30  # IoU threshold for track association

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ CLASS & HAZARD DEFINITIONS  (38 classes — unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MASTER_CLASSES = [
    "bloody_objects","mask","n95","oxygen_cylinder","radioactive_objects",
    "bandage","blade","capsule","cotton_swab","covid_buffer",
    "covid_buffer_box","covid_test_case","gauze","glass_bottle",
    "harris_uni_core","harris_uni_core_cap","iodine_swab","mercury_thermometer",
    "paperbox","pill","plastic_medical_bag","plastic_medical_bottle",
    "medical_gloves","reagent_tube","reagent_tube_cap","scalpel",
    "single_channel_pipette","syringe","transferpettor_glass",
    "transferpettor_plastic","tweezer_metal","tweezer_plastic","unguent",
    "electronic_thermometer","cap_plastic","drug_packaging",
    "medical_infusion_bag","needle",
]

HAZARD_LEVELS = {
    # ● HIGH RISK
    "needle":3,"syringe":3,"scalpel":3,"blade":3,
    "bloody_objects":3,"radioactive_objects":3,"mercury_thermometer":3,
    # ● MEDIUM RISK
    "bandage":2,"gauze":2,"cotton_swab":2,"medical_gloves":2,
    "mask":2,"n95":2,"covid_buffer":2,"covid_buffer_box":2,
    "covid_test_case":2,"iodine_swab":2,"reagent_tube":2,
    "reagent_tube_cap":2,"single_channel_pipette":2,
    "transferpettor_glass":2,"transferpettor_plastic":2,
    "tweezer_metal":2,"tweezer_plastic":2,"medical_infusion_bag":2,
    "electronic_thermometer":2,"pill":2,"capsule":2,"unguent":2,
    # ● LOW RISK
    "drug_packaging":1,"paperbox":1,"plastic_medical_bag":1,
    "plastic_medical_bottle":1,"glass_bottle":1,"cap_plastic":1,
    "oxygen_cylinder":1,"harris_uni_core":1,"harris_uni_core_cap":1,
}

# Per-class unique colours (BGR) — kept exactly as original
CLASS_COLORS_BGR = [
    ( 56, 56,255),( 51,157,255),( 56,225,255),( 56,220, 56),
    (255,190, 56),(255, 56, 56),(255, 56,220),(160, 56,255),
    ( 56,255,180),(180,255, 56),(120,120,255),(120,255,120),
    (255,120,120),( 80,200,255),(200, 80,255),(255,200, 80),
    (200,255, 80),( 80,255,200),(255, 80,200),( 80,160,255),
    (255,160, 80),(160,255, 80),(200, 56,200),( 56,200,200),
    (200,200, 56),( 80, 80,255),(255, 80, 80),( 80,255, 80),
    (100,100,230),(230,100,100),(100,230,100),( 50,180,230),
    (230, 50,180),(180,230, 50),( 30,140,255),(255,140, 30),
    (140, 30,255),(150,100,200),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ THEME PALETTE  (BGR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C_BG      = ( 18,  18,  24)   # main dark canvas
C_PANEL   = ( 26,  26,  34)   # panel background
C_PANEL2  = ( 36,  36,  48)   # alternating row / lighter strip
C_ACCENT  = ( 50, 185, 255)   # cyan accent (title, FPS)
C_TEXT    = (228, 228, 240)   # primary label text
C_MUTED   = (120, 120, 148)   # secondary / dim text
C_BORDER  = ( 52,  52,  72)   # separator lines
C_HIGH    = (  0,  48, 255)   # HIGH  — red
C_MED     = (  0, 175, 255)   # MEDIUM — amber/orange
C_LOW     = (  0, 205,  75)   # LOW   — green

RISK_COL  = {"HIGH": C_HIGH, "MEDIUM": C_MED, "LOW": C_LOW}
UQ_COL    = {"HIGH": C_HIGH, "MEDIUM": C_MED, "LOW": C_LOW}
HZ_COL    = {3: C_HIGH, 2: C_MED, 1: C_LOW}   # hazard → bbox color

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ CENTROID / IoU TRACKER  (lightweight, single-file)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CentroidTracker:
    """Assigns persistent integer IDs to objects across frames via IoU matching."""

    def __init__(self, max_lost=TRACKER_LOST, iou_thresh=TRACKER_IOU):
        self.nxt      = 0
        self.tracks   = {}        # id → {box, cls, lost, age}
        self.max_lost = max_lost
        self.iou_thr  = iou_thresh

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
        ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0: return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(ua, 1e-6)

    def update(self, dets):
        """
        dets : list of (x1,y1,x2,y2, cls_id)
        returns: list of (x1,y1,x2,y2, cls_id, track_id)
        """
        if not dets:
            for tid in list(self.tracks):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
            return []

        det_boxes = [d[:4] for d in dets]
        det_cls   = [d[4]  for d in dets]
        assigned  = {}   # det_idx → track_id
        used      = set()

        for tid, tk in list(self.tracks.items()):
            best_i, best_v = -1, self.iou_thr
            for i, db in enumerate(det_boxes):
                if i in used: continue
                v = self._iou(tk["box"], db)
                if v > best_v: best_v, best_i = v, i
            if best_i >= 0:
                self.tracks[tid].update(
                    {"box": det_boxes[best_i], "cls": det_cls[best_i], "lost": 0})
                self.tracks[tid]["age"] += 1
                assigned[best_i] = tid
                used.add(best_i)

        for i in range(len(dets)):
            if i not in assigned:
                self.tracks[self.nxt] = {
                    "box": det_boxes[i], "cls": det_cls[i], "lost": 0, "age": 0}
                assigned[i] = self.nxt
                self.nxt   += 1

        for tid in list(self.tracks):
            if tid not in assigned.values():
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return [(*det_boxes[i], det_cls[i], assigned[i]) for i in range(len(dets))]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ HEATMAP ACCUMULATOR  (downscaled for speed, Gaussian-blurred)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HeatmapAccumulator:
    """Accumulates detection positions into a persistent spatial heatmap."""

    def __init__(self, H, W, scale=HM_SCALE, decay=HM_DECAY):
        self.scale = scale
        self.decay = decay
        self.sh    = max(H // scale, 1)
        self.sw    = max(W // scale, 1)
        self.buf   = np.zeros((self.sh, self.sw), dtype=np.float32)

    def update(self, boxes_xyxy):
        """Add detection blobs and apply temporal decay."""
        self.buf *= self.decay
        for b in boxes_xyxy:
            x1, y1, x2, y2 = b[:4]
            cx = int(np.clip((x1 + x2) / 2 / self.scale, 0, self.sw - 1))
            cy = int(np.clip((y1 + y2) / 2 / self.scale, 0, self.sh - 1))
            r  = max(4, int(((x2 - x1) + (y2 - y1)) / 4 / self.scale))
            cv2.circle(self.buf, (cx, cy), r, 1.0, -1)
        # Gaussian blur creates smooth Gaussian-like blobs
        self.buf = cv2.GaussianBlur(self.buf, (0, 0), 3)
        self.buf = np.clip(self.buf, 0, 10.0)  # prevent unbounded growth

    def render_overlay(self, frame, alpha=HM_ALPHA):
        """Return frame with heatmap blended in."""
        norm  = self.buf / (self.buf.max() + 1e-6)
        up    = cv2.resize(norm, (frame.shape[1], frame.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        u8    = (up * 255).astype(np.uint8)
        jet   = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        mask  = (up > 0.04).astype(np.float32)[..., np.newaxis]
        out   = (frame.astype(np.float32) * (1 - mask * alpha) +
                 jet.astype(np.float32)   * mask * alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def reset(self):
        self.buf[:] = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ DRAWING UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_rounded_rect(img, pt1, pt2, color, r=10, thick=2, filled=False):
    """Draw a rounded rectangle (outline or filled) on img."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if x2 <= x1 or y2 <= y1: return
    r = max(1, min(r, (x2-x1)//2 - 1, (y2-y1)//2 - 1))
    t = -1 if filled else thick
    if filled:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, t)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, t)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, t)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, t)
    if not filled:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thick)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thick)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thick)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thick)


def put_txt(img, text, pos, scale=0.48, color=C_TEXT, thick=1, shadow=True):
    """Put anti-aliased text with optional drop-shadow for legibility."""
    x, y = int(pos[0]), int(pos[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (x+1, y+1), font, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def fill_alpha(canvas, x1, y1, x2, y2, color, alpha):
    """Blend a solid color rectangle into canvas at given transparency."""
    x1,y1,x2,y2 = (int(v) for v in (x1,y1,x2,y2))
    x1=max(0,x1); y1=max(0,y1)
    x2=min(canvas.shape[1],x2); y2=min(canvas.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = np.clip(
        roi * (1-alpha) + np.array(color, np.float32) * alpha, 0, 255
    ).astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ MINI GRAPH RENDERER  (risk + uncertainty over time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_graph(history, title, line_color, W=None, H=GRAPH_H,
                 y_max=None, ticks=None):
    """
    Render a smoothed line graph as a numpy BGR image.
    history : deque/list of float
    ticks   : list of (value, bgr_color) for threshold lines
    """
    W = W or (SIDE_W - 10)
    img  = np.full((H, W, 3), C_PANEL, dtype=np.uint8)
    cv2.rectangle(img, (0,0), (W-1,H-1), C_BORDER, 1)

    arr = np.array(history, dtype=np.float32)
    n   = len(arr)
    if n < 2:
        put_txt(img, title,            (5, 14), 0.33, C_MUTED, 1, False)
        put_txt(img, "collecting...",  (5, H//2+6), 0.32, C_BORDER, 1, False)
        return img

    vmax = float(np.max(arr)) if y_max is None else float(y_max)
    vmax = max(vmax, 1e-6)
    px, py = 5, 20    # x / y padding from edges
    gw = W - 2*px
    gh = H - py - 8

    def v2y(v):
        return int(py + gh * (1.0 - min(max(v, 0), vmax) / vmax))

    def i2x(i):
        return px + int(i * gw / max(n-1, 1))

    # Horizontal threshold lines
    if ticks:
        for tv, tc in ticks:
            ty = v2y(tv)
            if py <= ty <= H-8:
                cv2.line(img, (px, ty), (W-px, ty), tc, 1)

    # Build point list
    pts = np.array([[i2x(i), max(py, min(H-8, v2y(v)))]
                    for i, v in enumerate(arr)], dtype=np.int32)

    # Shaded fill under the curve
    fill_pts = [(px, H-8)] + pts.tolist() + [(pts[-1][0], H-8)]
    fp = np.array(fill_pts, np.int32)
    tinted = np.full_like(img, line_color)
    mask   = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [fp], 255)
    img[mask > 0] = np.clip(
        img[mask > 0].astype(np.float32) * 0.82 +
        tinted[mask > 0].astype(np.float32) * 0.18,
        0, 255
    ).astype(np.uint8)

    # Line (gradient: dim → bright toward current frame)
    for i in range(len(pts)-1):
        t = i / max(len(pts)-2, 1)
        c = tuple(int(c0 * 0.25 + c1 * 0.75 * t + c1 * 0.25)
                  for c0, c1 in zip(C_PANEL, line_color))
        c = tuple(min(255, max(0, v)) for v in c)
        cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), c, 1, cv2.LINE_AA)

    # Current value dot
    cv2.circle(img, tuple(pts[-1]), 3, line_color, -1, cv2.LINE_AA)

    # Labels
    put_txt(img, title,            (7, 14),  0.33, C_MUTED,     1, False)
    put_txt(img, f"{arr[-1]:.2f}", (W-48,14),0.36, line_color,  1, False)

    return img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ CONTAMINATION RISK SCORER  (vectorized, base-hazard included)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_contamination_risk(result):
    """
    Improved risk logic:
    - Pairwise proximity hazard (same as original)
    - Additional base contribution from individual hazard levels
    Returns (label: str, score: float)
    """
    if result.boxes is None or len(result.boxes) == 0:
        return "LOW", 0.0

    boxes   = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    n       = len(boxes)

    h_vals = np.array(
        [HAZARD_LEVELS.get(MASTER_CLASSES[min(c, len(MASTER_CLASSES)-1)], 1)
         for c in cls_ids], dtype=np.float32)

    # Base contribution per object
    risk_score = float(np.sum(h_vals) * 0.4)

    # Pairwise proximity contribution
    if n >= 2:
        cx = (boxes[:,0] + boxes[:,2]) * 0.5
        cy = (boxes[:,1] + boxes[:,3]) * 0.5
        for i in range(n):
            for j in range(i+1, n):
                dist = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if dist < 200:
                    risk_score += (h_vals[i] * h_vals[j]) / max(dist, 1.0) * 100

    if risk_score < 2:    return "LOW",    risk_score
    elif risk_score < 8:  return "MEDIUM", risk_score
    else:                 return "HIGH",   risk_score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ MONTE CARLO UNCERTAINTY  (throttled to every MC_INTERVAL frames)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_uncertainty(model, frame, passes=MC_PASSES, half_flag=False):
    """
    Run `passes` stochastic forward passes; return (mean_dets, variance).
    Variance is used as the uncertainty proxy.
    """
    counts = []
    for _ in range(passes):
        with torch.no_grad():
            res = model.predict(frame, conf=0.25, iou=0.45,
                                imgsz=IMG_SIZE, half=half_flag, verbose=False)
        counts.append(len(res[0].boxes) if res[0].boxes is not None else 0)
    return float(np.mean(counts)), float(np.var(counts))


def uq_classify(var):
    """Map variance to (label, color)."""
    if var < 1.0: return "LOW",    C_LOW
    if var < 3.0: return "MEDIUM", C_MED
    return         "HIGH",         C_HIGH


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ DRAW DETECTIONS  (masks + rounded boxes + hazard badges + IDs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def draw_detections(frame, result, tracker):
    """
    Draw segmentation masks, rounded hazard-colored bounding boxes,
    and label badges with track IDs.

    Returns:
        annotated_frame  : np.ndarray
        detected         : list of (cls_name, conf, hazard_level)
    """
    H, W    = frame.shape[:2]
    detected = []

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        tracker.update([])
        return frame, detected

    masks   = result.masks.data.cpu().numpy()           # (N, h, w)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()
    xyxy    = result.boxes.xyxy.cpu().numpy().astype(int)

    overlay = frame.copy()
    dets_for_tracker = []

    # ── 1. Segmentation mask overlay ───────────────────────
    for i in range(len(cls_ids)):
        cid = min(cls_ids[i], len(MASTER_CLASSES)-1)
        cls_name = MASTER_CLASSES[cid]
        hazard   = HAZARD_LEVELS.get(cls_name, 1)
        base_col = CLASS_COLORS_BGR[cid % len(CLASS_COLORS_BGR)]
        conf     = float(confs[i])
        box      = xyxy[i]

        detected.append((cls_name, conf, hazard))
        dets_for_tracker.append((*box.tolist(), int(cid)))

        # Resize mask to frame dimensions
        mask_r   = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_LINEAR)
        m        = mask_r > 0.5
        overlay[m] = np.clip(
            overlay[m].astype(np.float32) * 0.38 +
            np.array(base_col, np.float32) * 0.62,
            0, 255
        ).astype(np.uint8)

        # Mask contour (thin outline)
        m_u8  = (mask_r * 255).astype(np.uint8)
        ctrs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, ctrs, -1, base_col, 1)

    # Blend mask layer onto frame
    frame = cv2.addWeighted(overlay, 0.87, frame, 0.13, 0)

    # ── 2. Update tracker ──────────────────────────────────
    tracked = tracker.update(dets_for_tracker)

    # Build lookup: (x1,y1) → track_id for fast matching
    pos_to_tid = {}
    for t in tracked:
        pos_to_tid[(t[0], t[1])] = t[5]

    # ── 3. Bounding boxes + badges ─────────────────────────
    for i in range(len(cls_ids)):
        cid      = min(cls_ids[i], len(MASTER_CLASSES)-1)
        cls_name = MASTER_CLASSES[cid]
        hazard   = HAZARD_LEVELS.get(cls_name, 1)
        hz_col   = HZ_COL[hazard]
        conf     = float(confs[i])
        x1,y1,x2,y2 = xyxy[i]

        # Get track ID if available
        tid = pos_to_tid.get((x1, y1), None)
        tid_str  = f" #{tid}" if tid is not None else ""

        # Rounded bounding box (hazard-colored)
        draw_rounded_rect(frame, (x1,y1), (x2,y2), hz_col, r=8, thick=2)

        # Label badge
        label    = f"{cls_name}{tid_str}  {conf:.2f}"
        lscale   = 0.40
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
        bx1 = x1
        bx2 = x1 + lw + 10
        by2 = y1
        by1 = max(0, y1 - lh - 9)

        if by2 > by1 and bx2 > bx1:
            draw_rounded_rect(frame, (bx1,by1), (bx2,by2), hz_col, r=4, filled=True)
            put_txt(frame, label, (bx1+5, by2-3), lscale, (255,255,255), 1, shadow=False)

    return frame, detected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ CANVAS BUILDER  (assembles top bar + video + side panel)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_canvas(frame, fps, risk_label, risk_score, uq_lbl, uq_var,
                 detected, risk_hist, uq_hist,
                 frame_count, show_heatmap, paused, alert_active):
    """
    Compose the final display canvas:
    ┌─────────────────────────────────┬──────────┐
    │  TOP STATUS BAR (full width)              │
    ├─────────────────────────────────┬──────────┤
    │  ANNOTATED VIDEO FRAME          │  SIDE    │
    │                                 │  PANEL   │
    │                                 │  (graphs │
    │                                 │   + list)│
    └─────────────────────────────────┴──────────┘
    """
    H, W     = frame.shape[:2]
    tot_H    = H + TOP_H
    tot_W    = W + SIDE_W

    canvas = np.full((tot_H, tot_W, 3), C_BG, dtype=np.uint8)

    # ── Place video frame ───────────────────────────────────
    canvas[TOP_H:TOP_H+H, 0:W] = frame

    # ━━ TOP STATUS BAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    canvas[0:TOP_H, :] = C_PANEL
    cv2.line(canvas, (0, TOP_H-1), (tot_W, TOP_H-1), C_BORDER, 1)

    # Logo block
    put_txt(canvas, "MED",    (12, 28), 0.72, C_ACCENT, 2, shadow=True)
    put_txt(canvas, "WASTE",  (12, 50), 0.56, C_TEXT,   1)
    put_txt(canvas, "DETECT", (12, 66), 0.36, C_MUTED,  1)
    cv2.line(canvas, (82, 6), (82, TOP_H-6), C_BORDER, 1)

    # Status pills — helper
    def pill(x, y, label, value, vc):
        put_txt(canvas, label, (x, y+1),  0.31, C_MUTED, 1, shadow=False)
        put_txt(canvas, value, (x, y+20), 0.62, vc,      2, shadow=True)

    rc = RISK_COL.get(risk_label, C_LOW)
    uc = UQ_COL.get(uq_lbl,    C_LOW)

    pill( 92,  14, "FPS",         f"{fps:.1f}",        C_ACCENT)
    pill(155,  14, "RISK",         risk_label,          rc)
    pill(238,  14, "UNCERTAINTY",  uq_lbl,              uc)
    pill(368,  14, "OBJECTS",     str(len(detected)),   C_TEXT)
    pill(448,  14, "SCORE",       f"{risk_score:.2f}",  rc)
    pill(522,  14, "VAR",         f"{uq_var:.2f}",      uc)

    # Keyboard hints (top-right corner of bar)
    hints = "[H]Heatmap  [R]Reset  [S]Save  [P]Pause  [Q]Quit"
    put_txt(canvas, hints, (tot_W - 410, TOP_H-10), 0.30, C_MUTED, 1, shadow=False)

    # Frame counter
    put_txt(canvas, f"Frame {frame_count}", (W - 100, TOP_H-10), 0.31, C_MUTED, 1, shadow=False)

    # Heatmap indicator
    if show_heatmap:
        put_txt(canvas, "HEATMAP ON", (W - 100, TOP_H-26), 0.34, C_ACCENT, 1)

    # Paused overlay on bar
    if paused:
        fill_alpha(canvas, 0, 0, tot_W, TOP_H, (0,0,0), 0.45)
        put_txt(canvas, "PAUSED", (tot_W//2 - 50, TOP_H//2+8), 0.9, C_MED, 2)

    # ━━ SIDE PANEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    sx = W
    canvas[TOP_H:, sx:sx+SIDE_W] = C_PANEL
    cv2.line(canvas, (sx, TOP_H), (sx, tot_H), C_BORDER, 1)

    # Panel section: Detected Objects header
    canvas[TOP_H:TOP_H+28, sx:sx+SIDE_W] = C_PANEL2
    cv2.line(canvas, (sx, TOP_H+28), (sx+SIDE_W, TOP_H+28), C_BORDER, 1)
    put_txt(canvas, "DETECTED OBJECTS", (sx+8, TOP_H+19), 0.38, C_ACCENT, 1, shadow=False)

    # Object list — dynamic max rows based on available height
    graph_block = GRAPH_H * 2 + 20   # space reserved for graphs
    list_avail  = H - 28 - graph_block - 10
    max_rows    = max(1, list_avail // 22)
    show_count  = min(len(detected), max_rows)

    for j in range(show_count):
        cls_name, conf, hazard = detected[j]
        ey  = TOP_H + 28 + j * 22
        bg  = C_PANEL2 if j % 2 == 0 else C_PANEL
        canvas[ey:ey+22, sx:sx+SIDE_W] = bg

        dot_col = HZ_COL[hazard]
        cv2.circle(canvas, (sx+11, ey+11), 5, dot_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx+11, ey+11), 5, (0,0,0),  1, cv2.LINE_AA)

        put_txt(canvas, cls_name[:19], (sx+22, ey+15), 0.37, C_TEXT,  1, shadow=False)
        put_txt(canvas, f"{conf:.2f}", (sx+SIDE_W-40, ey+15), 0.35, C_MUTED, 1, shadow=False)

    if len(detected) > show_count:
        extra_y = TOP_H + 28 + show_count * 22
        put_txt(canvas, f"+ {len(detected)-show_count} more ...",
                (sx+10, extra_y+14), 0.34, C_MUTED, 1, shadow=False)

    # ── Mini Graphs ─────────────────────────────────────────
    gy1 = tot_H - GRAPH_H*2 - 14   # graph 1 top y
    gy2 = tot_H - GRAPH_H    - 6   # graph 2 top y

    # Graph region separator line
    cv2.line(canvas, (sx, gy1-4), (sx+SIDE_W, gy1-4), C_BORDER, 1)

    if len(risk_hist) > 1:
        g1 = render_graph(risk_hist, "RISK SCORE",    rc,
                          W=SIDE_W-8, H=GRAPH_H,
                          ticks=[(2.0, C_LOW), (8.0, C_MED)])
        canvas[gy1:gy1+GRAPH_H, sx+4:sx+4+SIDE_W-8] = g1

    if len(uq_hist) > 1:
        g2 = render_graph(uq_hist, "UNCERTAINTY", uc,
                          W=SIDE_W-8, H=GRAPH_H,
                          ticks=[(1.0, C_LOW), (3.0, C_MED)])
        y2s = min(gy2, tot_H - GRAPH_H - 2)
        canvas[y2s:y2s+GRAPH_H, sx+4:sx+4+SIDE_W-8] = g2

    # ━━ ALERT OVERLAY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if alert_active:
        # Pulsing border (sin-wave brightness)
        pulse = 0.45 + 0.35 * abs(np.sin(frame_count * 0.18))
        col   = tuple(int(c * pulse) for c in C_HIGH)
        for thick in [12, 6, 3]:
            cv2.rectangle(canvas, (0, TOP_H), (W-1, tot_H-1), col, thick)
        # Warning banner
        banner = "  ⚠  HIGH RISK DETECTED  ⚠  "
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
        bx = (W - bw) // 2
        fill_alpha(canvas, bx-10, TOP_H+8, bx+bw+10, TOP_H+bh+22, (0,0,0), 0.6)
        put_txt(canvas, banner, (bx, TOP_H+bh+10), 0.70, C_HIGH, 2)

    return canvas


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ SESSION SUMMARY  (matplotlib charts saved + displayed on exit)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def show_session_summary(session_counts, risk_hist, uq_hist, frame_count, out_dir):
    """Produce and display a 3-chart matplotlib session summary."""
    det_cls = {k:v for k,v in session_counts.items() if v > 0}
    if not det_cls:
        print("\n   No detections recorded — summary skipped.")
        return

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.patch.set_facecolor("#0E0E16")
    fig.suptitle(
        f"Medical Waste Detection — Session Summary  ({frame_count} frames)",
        color="white", fontsize=15, fontweight="bold", y=1.02)

    # ── Chart 1: Class frequency ───────────────────────────
    ax1 = axes[0]
    top  = sorted(det_cls.items(), key=lambda x:x[1], reverse=True)[:16]
    names, counts = zip(*top)
    bar_colors = [
        "tomato"    if HAZARD_LEVELS.get(n,1)==3 else
        "orange"    if HAZARD_LEVELS.get(n,1)==2 else
        "limegreen" for n in names
    ]
    bars = ax1.barh(list(names)[::-1], list(counts)[::-1],
                    color=list(bar_colors)[::-1], edgecolor="#222", linewidth=0.6)
    ax1.set_title("Class Detection Frequency", color="white", pad=8, fontsize=12)
    ax1.set_facecolor("#14141E")
    ax1.tick_params(colors="white", labelsize=8)
    for sp in ax1.spines.values(): sp.set_color("#333")
    for bar, cnt in zip(bars, list(counts)[::-1]):
        ax1.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                 str(cnt), va="center", color="white", fontsize=8)
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="tomato",    label="HIGH"),
                         Patch(color="orange",    label="MEDIUM"),
                         Patch(color="limegreen", label="LOW")],
               fontsize=8, facecolor="#14141E", edgecolor="#333", loc="lower right")

    # ── Chart 2: Risk score over time ─────────────────────
    ax2 = axes[1]
    if len(risk_hist) > 1:
        x = np.arange(len(risk_hist))
        y = np.array(list(risk_hist), dtype=np.float32)
        ax2.fill_between(x, y, alpha=0.25, color="tomato")
        ax2.plot(x, y, color="tomato", linewidth=1.4, label="Risk Score (EMA)")
        ax2.axhline(2.0, color="limegreen", ls="--", lw=1, label="LOW → MED  (2)")
        ax2.axhline(8.0, color="orange",    ls="--", lw=1, label="MED → HIGH (8)")
        ax2.set_xlabel("Frame index", color="#888", fontsize=9)
        ax2.set_ylabel("Risk Score",  color="#888", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#14141E", edgecolor="#333")
    ax2.set_title("Risk Score Over Time", color="white", pad=8, fontsize=12)
    ax2.set_facecolor("#14141E")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_color("#333")

    # ── Chart 3: Hazard level pie ─────────────────────────
    ax3 = axes[2]
    h_counts = {1:0, 2:0, 3:0}
    for cname, cnt in det_cls.items():
        h_counts[HAZARD_LEVELS.get(cname,1)] += cnt
    sizes   = [h_counts[1], h_counts[2], h_counts[3]]
    labels  = [f"LOW (1)\n{h_counts[1]}", f"MED (2)\n{h_counts[2]}",
               f"HIGH (3)\n{h_counts[3]}"]
    colors  = ["limegreen", "orange", "tomato"]
    explode = (0.0, 0.05, 0.10)
    wedges, texts, autos = ax3.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=120,
        textprops={"color":"white","fontsize":9},
        wedgeprops={"edgecolor":"#0E0E16","linewidth":2},
    )
    for at in autos: at.set_color("white"); at.set_fontsize(9)
    ax3.set_title("Hazard Level Distribution", color="white", pad=8, fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"session_summary_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0E0E16")
    plt.close()
    print(f"\n   Session summary → {out_path}")

    # Display with OpenCV
    img = cv2.imread(out_path)
    if img is not None:
        cv2.imshow("Session Summary — any key to close", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Session Summary — any key to close")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("  MEDICAL WASTE DETECTION — PROFESSIONAL EDITION")
print("=" * 65)

device    = "0" if torch.cuda.is_available() else "cpu"
half_flag = (device != "cpu")   # FP16 only on GPU

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory // (1024**2)
    print(f"\n  GPU   : {props.name}  ({vram} MB VRAM)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True   # ← boosts conv speed
else:
    print(f"\n  CPU mode (GPU not detected)")

if not os.path.exists(BEST_MODEL):
    print(f"\n  ERROR — Model not found:\n  {BEST_MODEL}\n  Run training first.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"  Model  : {BEST_MODEL}")
print(f"  Mode   : {MODE}")
print(f"  Device : {'GPU (FP16)' if half_flag else 'CPU'}")

# ── Load model ──────────────────────────────────────────────
print("\n  Loading model ...", end="", flush=True)
model = YOLO(BEST_MODEL)
print("  OK")
print(f"  Conf={CONF_THRESH}  IoU={IOU_THRESH}  ImgSize={IMG_SIZE}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ IMAGE MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if MODE == "image":
    print("\n" + "="*60)
    print("IMAGE MODE")
    print("="*60)

    if not os.path.exists(INPUT_PATH):
        print(f"  ERROR: Image not found: {INPUT_PATH}"); sys.exit(1)

    frame = cv2.imread(INPUT_PATH)
    if frame is None:
        print(f"  ERROR: Could not read image: {INPUT_PATH}"); sys.exit(1)

    results = model.predict(INPUT_PATH, conf=CONF_THRESH, iou=IOU_THRESH,
                            imgsz=IMG_SIZE, device=device,
                            half=half_flag, verbose=False)
    result  = results[0]

    tracker  = CentroidTracker()
    frame, detected = draw_detections(frame, result, tracker)
    rl, rs   = compute_contamination_risk(result)
    _, uv    = compute_uncertainty(model, frame, passes=MC_PASSES, half_flag=half_flag)
    ul, _    = uq_classify(uv)

    canvas = build_canvas(
        frame, fps=0, risk_label=rl, risk_score=rs, uq_lbl=ul, uq_var=uv,
        detected=detected, risk_hist=deque([rs]), uq_hist=deque([uv]),
        frame_count=1, show_heatmap=False, paused=False,
        alert_active=(rl=="HIGH" or ul=="HIGH")
    )

    out_path = os.path.join(OUTPUT_DIR, "annotated_" + os.path.basename(INPUT_PATH))
    cv2.imwrite(out_path, canvas)
    print(f"\n  Detected {len(detected)} objects")
    for cls_name, conf, hz in detected:
        print(f"    [{['','LOW','MED','HIGH'][hz]}] {cls_name}  conf={conf:.2f}")
    print(f"\n  Risk: {rl}  Score: {rs:.2f}")
    print(f"  Uncertainty: {ul}  Var: {uv:.2f}")
    print(f"\n  Saved → {out_path}")
    print("  Press any key to close.\n")

    cv2.imshow("Medical Waste Detection", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ VIDEO MODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif MODE == "video":
    print("\n" + "="*60)
    print("VIDEO MODE")
    print("="*60)

    if not os.path.exists(INPUT_PATH):
        print(f"  ERROR: Video not found: {INPUT_PATH}"); sys.exit(1)

    cap          = cv2.VideoCapture(INPUT_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps     = cap.get(cv2.CAP_PROP_FPS)
    VW           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VH           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path     = os.path.join(OUTPUT_DIR, "annotated_" + os.path.basename(INPUT_PATH))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(out_path, fourcc, orig_fps,
                                   (VW + SIDE_W, VH + TOP_H))

    tracker        = CentroidTracker()
    heatmap        = HeatmapAccumulator(VH, VW)
    risk_hist      = deque(maxlen=GRAPH_LEN)
    uq_hist        = deque(maxlen=GRAPH_LEN)
    session_counts = {c:0 for c in MASTER_CLASSES}
    frame_count    = 0
    fps            = orig_fps
    fps_buf        = deque(maxlen=12)
    uq_var_cache   = 0.0
    rs_ema         = 0.0
    uv_ema         = 0.0
    show_heatmap   = False

    print(f"\n  Frames  : {total_frames}")
    print(f"  FPS     : {orig_fps:.1f}")
    print(f"  Output  : {out_path}")
    print(f"  Press Q to quit early\n")

    pbar = tqdm(total=total_frames, desc="  Processing", unit="fr", ncols=72)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        t0 = time.time()
        results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                imgsz=IMG_SIZE, device=device,
                                half=half_flag, verbose=False)
        result  = results[0]

        frame, detected = draw_detections(frame, result, tracker)

        if result.boxes is not None and len(result.boxes):
            heatmap.update(result.boxes.xyxy.cpu().numpy())
        if show_heatmap:
            frame = heatmap.render_overlay(frame)

        rl, rs  = compute_contamination_risk(result)
        rs_ema  = rs_ema * (1-RISK_EMA_K) + rs * RISK_EMA_K

        if frame_count % MC_INTERVAL == 0:
            _, uq_var_cache = compute_uncertainty(model, frame, MC_PASSES, half_flag)
        uv_ema = uv_ema * (1-UQ_EMA_K) + uq_var_cache * UQ_EMA_K

        ul, _ = uq_classify(uv_ema)
        risk_hist.append(rs_ema)
        uq_hist.append(uv_ema)

        for cls_name, conf, hz in detected:
            if cls_name in session_counts: session_counts[cls_name] += 1

        fps_buf.append(1.0 / max(time.time()-t0, 1e-6))
        fps = float(np.mean(fps_buf))
        frame_count += 1

        canvas = build_canvas(frame, fps, rl, rs_ema, ul, uv_ema,
                              detected, risk_hist, uq_hist,
                              frame_count, show_heatmap, False,
                              (rl=="HIGH" or ul=="HIGH"))
        writer.write(canvas)
        cv2.imshow("Medical Waste Detection — Q to quit", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27): break
        elif key == ord("h"):     show_heatmap = not show_heatmap
        elif key == ord("r"):     heatmap.reset()

        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n  Saved → {out_path}")
    show_session_summary(session_counts, risk_hist, uq_hist, frame_count, OUTPUT_DIR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ■ WEBCAM MODE  (main real-time loop with all features)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif MODE == "webcam":
    print("\n" + "="*60)
    print("WEBCAM MODE — Real-time Detection")
    print("="*60)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"\n  ERROR: Webcam #{WEBCAM_INDEX} unavailable. Try WEBCAM_INDEX=1.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Camera     : {W}×{H}")
    print(f"  Controls   : Q/ESC=Quit  S=Snapshot  P=Pause  H=Heatmap  R=Reset heatmap")

    # Optional recording
    writer = None
    if SAVE_OUTPUT:
        ts       = time.strftime("%Y%m%d_%H%M%S")
        rec_path = os.path.join(OUTPUT_DIR, f"webcam_{ts}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(rec_path, fourcc, 20, (W+SIDE_W, H+TOP_H))
        print(f"  Recording  : {rec_path}")

    # State
    tracker        = CentroidTracker()
    heatmap        = HeatmapAccumulator(H, W)
    risk_hist      = deque(maxlen=GRAPH_LEN)
    uq_hist        = deque(maxlen=GRAPH_LEN)
    session_counts = {c:0 for c in MASTER_CLASSES}
    risk_dist      = {"LOW":0, "MEDIUM":0, "HIGH":0}

    frame_count  = 0
    fps_buf      = deque(maxlen=15)
    fps_display  = 0.0
    paused       = False
    show_heatmap = False
    uq_var_cache = 0.0
    rs_ema       = 0.0
    uv_ema       = 0.0
    last_canvas  = None

    print("\n  Starting...\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  WARNING: Frame read failed."); break

            t0 = time.time()

            # ── Inference ─────────────────────────────────────
            results = model.predict(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                                    imgsz=IMG_SIZE, device=device,
                                    half=half_flag, verbose=False)
            result = results[0]

            # ── Draw detections ────────────────────────────────
            frame, detected = draw_detections(frame, result, tracker)

            # ── Heatmap ───────────────────────────────────────
            if result.boxes is not None and len(result.boxes):
                heatmap.update(result.boxes.xyxy.cpu().numpy())
            if show_heatmap:
                frame = heatmap.render_overlay(frame)

            # ── Risk (EMA-smoothed) ────────────────────────────
            rl, rs = compute_contamination_risk(result)
            rs_ema  = rs_ema * (1-RISK_EMA_K) + rs * RISK_EMA_K
            risk_dist[rl] += 1

            # ── Uncertainty (every MC_INTERVAL frames) ─────────
            if frame_count % MC_INTERVAL == 0:
                _, uq_var_cache = compute_uncertainty(
                    model, frame, MC_PASSES, half_flag)
            uv_ema = uv_ema * (1-UQ_EMA_K) + uq_var_cache * UQ_EMA_K

            ul, _ = uq_classify(uv_ema)
            risk_hist.append(rs_ema)
            uq_hist.append(uv_ema)

            # ── Session counters ───────────────────────────────
            for cls_name, conf, hz in detected:
                if cls_name in session_counts:
                    session_counts[cls_name] += 1

            # ── FPS (rolling) ──────────────────────────────────
            fps_buf.append(1.0 / max(time.time()-t0, 1e-6))
            fps_display = float(np.mean(fps_buf))
            frame_count += 1

            alert = (rl == "HIGH") or (ul == "HIGH")

            canvas = build_canvas(
                frame, fps_display, rl, rs_ema, ul, uv_ema,
                detected, risk_hist, uq_hist,
                frame_count, show_heatmap, False, alert
            )
            last_canvas = canvas

            if writer is not None:
                writer.write(canvas)

        else:
            # Paused — show frozen frame + overlay
            if last_canvas is not None:
                canvas = last_canvas.copy()
                fill_alpha(canvas, 0, 0, W, H+TOP_H, (0,0,0), 0.40)
                put_txt(canvas, "PAUSED",
                        (W//2 - 65, (H+TOP_H)//2 + 12), 1.5, C_MED, 3)
            else:
                cv2.waitKey(30)
                continue

        cv2.imshow("Medical Waste Detection System", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):         # Quit
            break
        elif key == ord("s"):             # Snapshot
            ts_s  = time.strftime("%Y%m%d_%H%M%S")
            sp    = os.path.join(OUTPUT_DIR, f"snapshot_{ts_s}.jpg")
            cv2.imwrite(sp, canvas)
            print(f"  Snapshot → {sp}")
        elif key == ord("p"):             # Pause toggle
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'}")
        elif key == ord("h"):             # Heatmap toggle
            show_heatmap = not show_heatmap
            print(f"  Heatmap: {'ON' if show_heatmap else 'OFF'}")
        elif key == ord("r"):             # Reset heatmap
            heatmap.reset()
            print("  Heatmap reset.")

    # ── Cleanup ─────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n  Recording saved → {rec_path}")
    cv2.destroyAllWindows()

    # ── Terminal session stats ───────────────────────────────
    print("\n" + "="*65)
    print("  SESSION STATISTICS")
    print("="*65)

    det_cls = {k:v for k,v in session_counts.items() if v > 0}
    if det_cls:
        sorted_c = sorted(det_cls.items(), key=lambda x:x[1], reverse=True)
        mx = max(v for _,v in sorted_c)
        print(f"\n  {'Class':<28} {'Count':>6}  {'Bar':40}  Level")
        for cname, cnt in sorted_c:
            bar = "█" * int(30 * cnt / max(mx, 1))
            hl  = HAZARD_LEVELS.get(cname, 1)
            tag = ["","LOW ","MED ","HIGH"][hl]
            print(f"  {cname:<28} {cnt:>6}  {bar:<30}  {tag}")
    else:
        print("  No objects detected.")

    print(f"\n  Total frames : {frame_count}")
    print(f"  Risk dist    : LOW={risk_dist['LOW']}  "
          f"MED={risk_dist['MEDIUM']}  HIGH={risk_dist['HIGH']}")

    # ── Matplotlib session summary ───────────────────────────
    show_session_summary(session_counts, risk_hist, uq_hist, frame_count, OUTPUT_DIR)


else:
    print(f"\n  ERROR: Unknown mode '{MODE}'  — set to 'webcam', 'video', or 'image'.")
    sys.exit(1)


print("\n" + "="*65)
print("  Done.")
print("="*65)
