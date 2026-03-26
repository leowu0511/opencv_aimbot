# run_cv.py — AI Aim Lab v5 | 純 OpenCV Cyan 偵測版
# 特色：
#   - 完全移除 YOLO，改用 HSV 顏色篩選 + contour，單幀 < 1ms
#   - 保留 v4 的 pick_best / switch_confirm / auto_fire 邏輯
#   - 新增 DEBUG_OVERLAY：用獨立視窗顯示偵測框、FOV 圓、準心
#   - 按 D 切換 debug 視窗

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
DEADZONE_PX     = 4
FOV_RADIUS_PX   = 520
MIN_CONTOUR_AREA = 200      # 太小的色塊濾掉（雜訊）
MAX_CONTOUR_AREA = 80000    # 太大的也濾掉（UI 元素）

# ---------- Cyan HSV 範圍（針對 Aim Lab 青色球）----------
# 可按 D 開 debug 視窗，根據 mask 畫面微調
HSV_LOWER = np.array([85,  150, 150])
HSV_UPPER = np.array([100, 255, 255])

# ---------- 目標選擇（沿用 v4 v1-logic）----------
NEAR_RADIUS_PX      = 220
LOCK_RADIUS_PX      = 260
LOCK_STICKINESS     = 0.35
CENTER_PRIORITY_PX  = 170
SWITCH_ADVANTAGE_PX = 30
LOST_RESET_FRAMES   = 8
SWITCH_CONFIRM_FRAMES = 2

# ---------- Gain（兩段線性，1600 DPI）----------
MOVE_GAIN_NEAR  = 0.42
MOVE_GAIN_FAR   = 0.58
MAX_STEP_NEAR   = 18
MAX_STEP_FAR    = 220

# ---------- 自動開火 ----------
AUTO_FIRE_RADIUS_PX      = 12
AUTO_FIRE_HIT_RADIUS_PX  = 22
AUTO_FIRE_COOLDOWN       = 0
AUTO_FIRE_CONFIRM_FRAMES = 2

# ---------- Debug 視窗 ----------
DEBUG_OVERLAY        = False     # 按 D 切換
OVERLAY_SCALE        = 0.5       # 縮小顯示（避免視窗太大）
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
print("=== AI Aim Lab v5 | OpenCV Cyan Tracker ===")

SCREEN_W = ctypes.windll.user32.GetSystemMetrics(0)
SCREEN_H = ctypes.windll.user32.GetSystemMetrics(1)
CX = SCREEN_W / 2.0
CY = SCREEN_H / 2.0
print(f"桌面解析度: {SCREEN_W}x{SCREEN_H}  準心: ({CX}, {CY})")

# dxcam 只擷取 FOV 範圍，大幅減少處理像素
REGION_L = max(0, int(CX - FOV_RADIUS_PX))
REGION_T = max(0, int(CY - FOV_RADIUS_PX))
REGION_R = min(SCREEN_W, int(CX + FOV_RADIUS_PX))
REGION_B = min(SCREEN_H, int(CY + FOV_RADIUS_PX))
REGION   = (REGION_L, REGION_T, REGION_R, REGION_B)

# region 內的準心相對座標
RCX = CX - REGION_L
RCY = CY - REGION_T

print(f"擷取區域: {REGION}  區域準心: ({RCX:.1f}, {RCY:.1f})")
print("按 K 開關瞄準 | 按 N 切換自動開火 | 按 D 切換 Debug 視窗 | 按 ESC 結束")

camera = dxcam.create(device_idx=0, output_color="BGR")
camera.start(target_fps=TARGET_FPS, region=REGION)

# morphology kernel（去雜訊用）
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ===================== CV 偵測核心 =====================
def find_targets_cv(frame):
    """
    輸入：BGR frame（已裁切到 REGION）
    輸出：candidates list，格式與 v4 相同
          同時回傳 mask（供 debug 顯示）
    """
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

        # 相對 region 的中心
        rx = M["m10"] / M["m00"]
        ry = M["m01"] / M["m00"]

        # 轉換回絕對螢幕座標
        tx = rx + REGION_L
        ty = ry + REGION_T

        dx_c  = tx - CX
        dy_c  = ty - CY
        dist2 = dx_c * dx_c + dy_c * dy_c

        if dist2 > FOV_RADIUS_PX ** 2:
            continue

        # bounding rect（供 debug 框框）
        bx, by, bw, bh = cv2.boundingRect(cnt)

        candidates.append({
            "conf":  1.0,
            "tx": tx, "ty": ty,
            "rx": rx, "ry": ry,       # region 內座標，給 overlay 用
            "dx": dx_c, "dy": dy_c,
            "dist2": dist2,
            "area":  area,
            "bbox":  (bx, by, bw, bh),   # region 內 bounding box
            "cnt":   cnt,
        })

    return candidates, mask


# ===================== 目標選擇（v4 v1-logic）=====================
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


# ===================== Debug overlay 繪製 =====================
def draw_overlay(frame, candidates, best, mask):
    """
    在 frame 副本上畫：
      - 青色框框（所有偵測到的目標）
      - 紅色大框（當前鎖定目標）
      - 白色 FOV 圓
      - 綠色準心十字
      - 右半部顯示 mask
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # FOV 圓
    cv2.circle(vis, (int(RCX), int(RCY)), FOV_RADIUS_PX,
               (255, 255, 0), 1, cv2.LINE_AA)

    # 準心
    cs = 12
    cv2.line(vis, (int(RCX)-cs, int(RCY)), (int(RCX)+cs, int(RCY)),
             (0, 255, 0), 1)
    cv2.line(vis, (int(RCX), int(RCY)-cs), (int(RCX), int(RCY)+cs),
             (0, 255, 0), 1)

    for c in candidates:
        bx, by, bw, bh = c["bbox"]
        is_best = (best is not None and c is best)

        # 框框顏色：鎖定目標=紅，其他=青
        color     = (0, 0, 255) if is_best else (255, 220, 0)
        thickness = 2 if is_best else 1

        cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), color, thickness)

        # 圓心點
        cv2.circle(vis, (int(c["rx"]), int(c["ry"])), 3, color, -1)

        # 標籤
        dist = math.sqrt(c["dist2"])
        label = f"{dist:.0f}px"
        cv2.putText(vis, label, (bx, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 鎖定目標：畫準心到目標的連線
        if is_best:
            cv2.line(vis, (int(RCX), int(RCY)),
                     (int(c["rx"]), int(c["ry"])),
                     (0, 0, 255), 1, cv2.LINE_AA)

    # 右半部拼接 mask（偽彩色，方便看）
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # mask 大小與 vis 相同（都是 region frame）
    combined = np.hstack([vis, mask_color])

    # 縮小後顯示
    ow = int(combined.shape[1] * OVERLAY_SCALE)
    oh = int(combined.shape[0] * OVERLAY_SCALE)
    combined = cv2.resize(combined, (ow, oh), interpolation=cv2.INTER_LINEAR)

    # HUD 文字
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
            # 即使 aimbot 關閉，debug 視窗還是要更新
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

        # ---- 移動 ----
        if dist < DEADZONE_PX:
            move_x = move_y = 0
        else:
            ratio    = min(1.0, dist / float(NEAR_RADIUS_PX))
            gain     = MOVE_GAIN_NEAR + (MOVE_GAIN_FAR - MOVE_GAIN_NEAR) * ratio
            max_step = int(MAX_STEP_NEAR + (MAX_STEP_FAR - MAX_STEP_NEAR) * ratio)
            move_x   = int(max(-max_step, min(max_step, dx * gain)))
            move_y   = int(max(-max_step, min(max_step, dy * gain)))
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
            draw_overlay(frame, candidates, best, mask)

        # ---- 終端 print ----
        frame_index += 1
        if frame_index % PRINT_EVERY_N_FRAMES == 0:
            ratio_d = min(1.0, dist / float(NEAR_RADIUS_PX))
            gain_d  = MOVE_GAIN_NEAR + (MOVE_GAIN_FAR - MOVE_GAIN_NEAR) * ratio_d
            step_d  = int(MAX_STEP_NEAR + (MAX_STEP_FAR - MAX_STEP_NEAR) * ratio_d)
            print(
                f"Target:({best['tx']:.1f},{best['ty']:.1f}) | "
                f"Dist:{dist:.1f} | Move:({move_x},{move_y}) | "
                f"gain:{gain_d:.3f} step:{step_d} | "
                f"balls:{len(candidates)} | pending:{pending_confirm} | "
                f"AutoFire:{'ON' if auto_fire_enabled else 'OFF'}"
            )

except KeyboardInterrupt:
    pass
finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("程式結束")