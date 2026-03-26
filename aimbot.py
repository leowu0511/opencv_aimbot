# run_cv.py — AI Aim Lab v5 | 純 OpenCV Cyan 偵測版 + PD Controller
# 核心改動：換成 PD 控制器解決震盪問題
#   - P 項：拉近目標
#   - D 項：煞車（接近目標自動減速，不衝過頭）
#   - D_CLAMP_PX：限制 D 項單幀最大貢獻，防止切換目標時暴衝

import ctypes
import math
import time
from ctypes import wintypes

import cv2
import dxcam
import keyboard
import numpy as np


# ===================== 基本參數 =====================
TARGET_FPS      = 144
DEADZONE_PX     = 4          # 靜止球用大一點死區，進去就不動
FOV_RADIUS_PX   = 520
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 80000

# ---------- Cyan HSV 範圍 ----------
HSV_LOWER = np.array([85,  150, 150])
HSV_UPPER = np.array([100, 255, 255])

# ---------- 目標選擇 ----------
NEAR_RADIUS_PX        = 220
LOCK_RADIUS_PX        = 260
LOCK_STICKINESS       = 0.35
CENTER_PRIORITY_PX    = 170
SWITCH_ADVANTAGE_PX   = 30
LOST_RESET_FRAMES     = 8
SWITCH_CONFIRM_FRAMES = 3    # 多確認一幀再切換，靜止球不急（舊值 2）

# ---------- PD 控制器參數（1600 DPI 中靈敏，靜止球）----------
KP           = 0.22   # 比例項：靜止球不需要追很急（舊值 0.30）
KD           = 0.30   # 微分項：煞車更強，最後幾像素不震盪（舊值 0.20）
D_CLAMP_PX   = 12     # D 項單幀最大貢獻，防切換暴衝
MAX_STEP     = 28     # 每幀最大位移像素（舊值 40）

# ---------- 自動開火 ----------
AUTO_FIRE_RADIUS_PX      = 10   # 靜止球精準度高，縮小開槍距離（舊值 14）
AUTO_FIRE_HIT_RADIUS_PX  = 20
AUTO_FIRE_COOLDOWN       = 0
AUTO_FIRE_CONFIRM_FRAMES = 4    # 更嚴格，完全穩定才開槍（舊值 3）

# ---------- Debug 視窗 ----------
DEBUG_OVERLAY        = False
OVERLAY_SCALE        = 0.5
PRINT_EVERY_N_FRAMES = 5


# ===================== SendInput =====================
ULONG_PTR = ctypes.POINTER(wintypes.ULONG)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("mi",   MOUSEINPUT),
    ]

MOUSEEVENTF_MOVE     = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004
INPUT_MOUSE          = 0

def move_rel(dx: int, dy: int) -> None:
    inp = INPUT()
    inp.type         = INPUT_MOUSE
    inp.mi.dx        = int(dx)
    inp.mi.dy        = int(dy)
    inp.mi.mouseData = 0
    inp.mi.dwFlags   = MOUSEEVENTF_MOVE
    inp.mi.time      = 0
    inp.mi.dwExtraInfo = None
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def mouse_left_click() -> None:
    down, up = INPUT(), INPUT()
    down.type = up.type = INPUT_MOUSE
    down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    up.mi.dwFlags   = MOUSEEVENTF_LEFTUP
    arr = (INPUT * 2)(down, up)
    ctypes.windll.user32.SendInput(2, ctypes.byref(arr), ctypes.sizeof(INPUT))


# ===================== 初始化 =====================
print("=== AI Aim Lab v5 | OpenCV Cyan Tracker + PD Controller ===")

SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
CX = SCREEN_W / 2.0
CY = SCREEN_H / 2.0
print(f"桌面解析度: {SCREEN_W}x{SCREEN_H}  準心: ({CX}, {CY})")

REGION_L = max(0, int(CX - FOV_RADIUS_PX))
REGION_T = max(0, int(CY - FOV_RADIUS_PX))
REGION_R = min(SCREEN_W, int(CX + FOV_RADIUS_PX))
REGION_B = min(SCREEN_H, int(CY + FOV_RADIUS_PX))
REGION   = (REGION_L, REGION_T, REGION_R, REGION_B)

RCX = CX - REGION_L
RCY = CY - REGION_T

print(f"擷取區域: {REGION}  區域準心: ({RCX:.1f}, {RCY:.1f})")
print("按 K 開關瞄準 | 按 N 切換自動開火 | 按 D 切換 Debug 視窗 | 按 ESC 結束")
print(f"PD 參數：KP={KP}  KD={KD}  D_CLAMP={D_CLAMP_PX}px  MAX_STEP={MAX_STEP}px")

camera = dxcam.create(device_idx=0, output_color="BGR")
camera.start(target_fps=TARGET_FPS, region=REGION)

_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ===================== CV 偵測核心 =====================
def find_targets_cv(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        rx = M["m10"] / M["m00"]
        ry = M["m01"] / M["m00"]
        tx = rx + REGION_L
        ty = ry + REGION_T

        dx_c  = tx - CX
        dy_c  = ty - CY
        dist2 = dx_c * dx_c + dy_c * dy_c

        if dist2 > FOV_RADIUS_PX ** 2:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        candidates.append({
            "conf":  1.0,
            "tx": tx, "ty": ty,
            "rx": rx, "ry": ry,
            "dx": dx_c, "dy": dy_c,
            "dist2": dist2,
            "area":  area,
            "bbox":  (bx, by, bw, bh),
            "cnt":   cnt,
        })

    return candidates, mask


# ===================== 目標選擇 =====================
def pick_best(candidates, last_target):
    best_center = min(candidates, key=lambda c: c["dist2"])
    best = best_center

    if last_target is not None:
        lx, ly = last_target
        best_score = float("inf")
        for c in candidates:
            tx, ty = c["tx"], c["ty"]
            center_dist = math.sqrt(c["dist2"])
            lock_dist   = math.hypot(tx - lx, ty - ly)
            score = (center_dist + lock_dist * LOCK_STICKINESS
                     if lock_dist <= LOCK_RADIUS_PX else center_dist)
            if score < best_score:
                best_score = score
                best = c

    if best_center["dist2"] <= CENTER_PRIORITY_PX ** 2:
        best = best_center
    elif math.sqrt(best_center["dist2"]) + SWITCH_ADVANTAGE_PX < math.sqrt(best["dist2"]):
        best = best_center

    return best


# ===================== PD 控制器 =====================
class PDController:
    """
    移動量 = KP * error + KD * (error - prev_error)
                 ↑ 拉近         ↑ 煞車項

    D 項原理：接近目標時 error 縮小，(error - prev_error) 為負，
    自動抵消 P 項 → 煞車，不衝過頭。
    D_CLAMP 防止切換目標瞬間 D 項爆炸。
    """
    def __init__(self):
        self.prev_ex = 0.0
        self.prev_ey = 0.0

    def reset(self):
        self.prev_ex = 0.0
        self.prev_ey = 0.0

    def compute(self, ex: float, ey: float):
        # D 項
        d_raw_x = ex - self.prev_ex
        d_raw_y = ey - self.prev_ey

        # Clamp：切換目標瞬間誤差跳很大，限制 D 項影響
        d_x = max(-D_CLAMP_PX, min(D_CLAMP_PX, d_raw_x))
        d_y = max(-D_CLAMP_PX, min(D_CLAMP_PX, d_raw_y))

        out_x = KP * ex + KD * d_x
        out_y = KP * ey + KD * d_y

        # 整體限幅
        out_x = max(-MAX_STEP, min(MAX_STEP, out_x))
        out_y = max(-MAX_STEP, min(MAX_STEP, out_y))

        self.prev_ex = ex
        self.prev_ey = ey

        return int(out_x), int(out_y)


# ===================== Debug overlay =====================
def draw_overlay(frame, candidates, best, mask, move_x=0, move_y=0):
    vis = frame.copy()

    cv2.circle(vis, (int(RCX), int(RCY)), FOV_RADIUS_PX,
               (255, 255, 0), 1, cv2.LINE_AA)

    cs = 12
    cv2.line(vis, (int(RCX)-cs, int(RCY)), (int(RCX)+cs, int(RCY)),
             (0, 255, 0), 1)
    cv2.line(vis, (int(RCX), int(RCY)-cs), (int(RCX), int(RCY)+cs),
             (0, 255, 0), 1)

    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        is_best = (best is not None and c is best)
        color     = (0, 0, 255) if is_best else (255, 220, 0)
        thickness = 2 if is_best else 1

        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, thickness)
        cv2.circle(vis, (int(c["rx"]), int(c["ry"])), 3, color, -1)

        dist = math.sqrt(c["dist2"])
        cv2.putText(vis, f"{dist:.0f}px", (bx, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if is_best:
            cv2.line(vis, (int(RCX), int(RCY)),
                     (int(c["rx"]), int(c["ry"])),
                     (0, 0, 255), 1, cv2.LINE_AA)
            # 移動向量箭頭（青色）
            if move_x != 0 or move_y != 0:
                arrow_scale = 4
                ax = int(RCX + move_x * arrow_scale)
                ay = int(RCY + move_y * arrow_scale)
                cv2.arrowedLine(vis, (int(RCX), int(RCY)), (ax, ay),
                                (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined   = np.hstack([vis, mask_color])
    ow = int(combined.shape[1] * OVERLAY_SCALE)
    oh = int(combined.shape[0] * OVERLAY_SCALE)
    combined = cv2.resize(combined, (ow, oh), interpolation=cv2.INTER_LINEAR)
    cv2.putText(combined, "FRAME | MASK", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("AimLab CV Debug", combined)
    cv2.waitKey(1)


# ===================== 主迴圈 =====================
aimbot_enabled    = False
auto_fire_enabled = False
last_toggle_time  = 0.0
last_fire_time    = 0.0
frame_index       = 0
fire_confirm_frames = 0

last_target    = None
lost_frames    = 0
pending_target = None
pending_confirm = 0

move_x = move_y = 0

pd = PDController()

try:
    while True:
        if keyboard.is_pressed("esc"):
            break

        now = time.time()

        # ---- 按鍵 ----
        if keyboard.is_pressed("k"):
            if now - last_toggle_time > 0.20:
                aimbot_enabled = not aimbot_enabled
                last_toggle_time = now
                last_target = None
                lost_frames = 0
                pending_target = None
                pending_confirm = 0
                fire_confirm_frames = 0
                pd.reset()
                print(f"Aimbot: {'ON' if aimbot_enabled else 'OFF'}")

        if keyboard.is_pressed("n"):
            if now - last_toggle_time > 0.20:
                auto_fire_enabled = not auto_fire_enabled
                last_toggle_time = now
                fire_confirm_frames = 0
                print(f"Auto Fire: {'ON' if auto_fire_enabled else 'OFF'}")

        if keyboard.is_pressed("d"):
            if now - last_toggle_time > 0.20:
                DEBUG_OVERLAY = not DEBUG_OVERLAY
                last_toggle_time = now
                if not DEBUG_OVERLAY:
                    cv2.destroyAllWindows()
                print(f"Debug Overlay: {'ON' if DEBUG_OVERLAY else 'OFF'}")

        if not aimbot_enabled:
            if DEBUG_OVERLAY:
                frame = camera.grab()
                if frame is not None:
                    _, mask = find_targets_cv(frame)
                    draw_overlay(frame, [], None, mask)
            else:
                time.sleep(0.002)
            continue

        frame = camera.grab()
        if frame is None:
            continue

        # ---- CV 偵測 ----
        candidates, mask = find_targets_cv(frame)

        if not candidates:
            lost_frames += 1
            fire_confirm_frames = 0
            pending_confirm = 0
            if lost_frames >= LOST_RESET_FRAMES:
                last_target = None
                pending_target = None
                pd.reset()
            if DEBUG_OVERLAY:
                draw_overlay(frame, [], None, mask)
            continue

        # ---- 目標選擇 ----
        raw_best = pick_best(candidates, last_target)

        same_as_last = (
            last_target is not None
            and math.hypot(raw_best["tx"] - last_target[0],
                           raw_best["ty"] - last_target[1]) < 60
        )

        if same_as_last or last_target is None:
            best = raw_best
            pending_target = None
            pending_confirm = 0
        else:
            raw_pos = (raw_best["tx"], raw_best["ty"])
            if (pending_target is not None
                    and math.hypot(raw_pos[0] - pending_target[0],
                                   raw_pos[1] - pending_target[1]) < 60):
                pending_confirm += 1
            else:
                pending_target = raw_pos
                pending_confirm = 1

            if pending_confirm >= SWITCH_CONFIRM_FRAMES:
                best = raw_best
                pending_target = None
                pending_confirm = 0
                pd.reset()   # 切換目標時重置 PD，避免舊 prev_error 影響
            else:
                best_by_last = min(
                    candidates,
                    key=lambda c: math.hypot(c["tx"] - last_target[0],
                                             c["ty"] - last_target[1])
                ) if last_target else raw_best
                best = best_by_last

        lost_frames  = 0
        last_target  = (best["tx"], best["ty"])

        dx   = best["dx"]
        dy   = best["dy"]
        dist = math.sqrt(best["dist2"])

        # ---- PD 移動 ----
        if dist < DEADZONE_PX:
            move_x = move_y = 0
            pd.prev_ex = 0.0
            pd.prev_ey = 0.0
        else:
            move_x, move_y = pd.compute(dx, dy)
            if move_x != 0 or move_y != 0:
                move_rel(move_x, move_y)

        # ---- 自動開火 ----
        can_fire = (dist <= AUTO_FIRE_RADIUS_PX) or (dist <= AUTO_FIRE_HIT_RADIUS_PX)
        if auto_fire_enabled and can_fire:
            fire_confirm_frames += 1
        else:
            fire_confirm_frames = 0

        now = time.time()
        if (auto_fire_enabled
                and fire_confirm_frames >= AUTO_FIRE_CONFIRM_FRAMES
                and (now - last_fire_time) >= AUTO_FIRE_COOLDOWN):
            mouse_left_click()
            last_fire_time = now
            fire_confirm_frames = 0

        # ---- Debug Overlay ----
        if DEBUG_OVERLAY:
            draw_overlay(frame, candidates, best, mask, move_x, move_y)

        # ---- 終端 print ----
        frame_index += 1
        if frame_index % PRINT_EVERY_N_FRAMES == 0:
            print(
                f"Target:({best['tx']:.1f},{best['ty']:.1f}) | "
                f"Dist:{dist:.1f} | Move:({move_x},{move_y}) | "
                f"KP={KP} KD={KD} | "
                f"balls:{len(candidates)} | pending:{pending_confirm} | "
                f"AutoFire:{'ON' if auto_fire_enabled else 'OFF'}"
            )

except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("程式結束")
