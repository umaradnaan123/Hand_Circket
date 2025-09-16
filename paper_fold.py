#!/usr/bin/env python3
"""
Advanced Hand Cricket (Gesture) Game
- Save as gesture_hand_cricket_advanced.py
- Install requirements: pip install -r requirements.txt
- Run: python gesture_hand_cricket_advanced.py

Controls:
- 1 / 2 / 3 : choose overs at initial screen
- O / E : choose Odd/Even for toss (during toss phase)
- B / L : if user wins toss, choose B(at) or L(bowl)
- Q or Esc or window close : quit

Notes:
- Show 1..6 fingers as your 'ball' or 'run'.
- Keep the hand steady for ~0.8s for the gesture to register.
- Program uses your default webcam (index 0). If different, change cv2.VideoCapture(0).
"""
import cv2
import mediapipe as mp
import pygame
import random
import time
from collections import deque

# ---------------- Config ----------------
GESTURE_STABLE_TIME = 0.8     # seconds the finger count must be steady before accepting
GESTURE_COOLDOWN = 0.9        # seconds between accepted gestures
TOSS_COUNTDOWN = 3            # countdown seconds for toss
POST_MATCH_DELAY = 4.0        # seconds to show result before restart
MAX_OVERS_OPTIONS = [1, 2, 3] # allowed overs
CAM_INDEX = 0                 # webcam index (change if needed)

# Colors
BG = (15, 24, 40)
TEXT = (240, 240, 240)
HIGHLIGHT = (0, 220, 140)
WARN = (255, 180, 80)
COMMENT = (255, 215, 0)

# ---------------- Setup modules ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# create hands detector; reuse for entire program
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.6,
                                min_tracking_confidence=0.6)

pygame.init()
pygame.font.init()
# Some systems may not have "Arial"; fallback to default font when necessary
try:
    font_big = pygame.font.SysFont("Arial", 34)
    font_med = pygame.font.SysFont("Arial", 26)
    font_small = pygame.font.SysFont("Arial", 20)
except Exception:
    font_big = pygame.font.Font(None, 34)
    font_med = pygame.font.Font(None, 26)
    font_small = pygame.font.Font(None, 20)

SCREEN_W, SCREEN_H = 1000, 700
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Advanced Gesture Hand Cricket")

cap = cv2.VideoCapture(CAM_INDEX)

# ---------------- Helpers ----------------
def count_fingers(hand_landmarks, handedness_label=None):
    """
    Simple finger count (0-6). Thumb heuristic included.
    Works best when palm faces camera. Returns integer 0..6.
    handedness_label: 'Left' or 'Right' if available (helps thumb heuristic).
    """
    tips = [8, 12, 16, 20]  # index, middle, ring, pinky tip indices
    mcp  = [5, 9, 13, 17]
    count = 0
    try:
        # count four fingers (index, middle, ring, pinky)
        for t, b in zip(tips, mcp):
            if hand_landmarks.landmark[t].y < hand_landmarks.landmark[b].y:
                count += 1
        # thumb: compare tip(4) x with ip(3) x;
        # orientation depends on whether it's a right or left hand
        thumb_tip_x = hand_landmarks.landmark[4].x
        thumb_ip_x  = hand_landmarks.landmark[3].x
        if handedness_label:
            # if it's a right hand, thumb will be on left side of image (smaller x) when palm faces camera.
            if handedness_label.lower().startswith('r'):
                if thumb_tip_x < thumb_ip_x:
                    count += 1
            else:
                if thumb_tip_x > thumb_ip_x:
                    count += 1
        else:
            # fallback heuristic (original behavior)
            if thumb_tip_x > thumb_ip_x:
                count += 1
    except Exception:
        return 0
    return max(0, min(6, count))

def draw_text(surface, txt, pos, font_obj=font_med, color=TEXT):
    surface.blit(font_obj.render(txt, True, color), pos)

# ---------------- Game State ----------------
def new_match_state(overs=1):
    return {
        "overs": overs,
        "balls_per_over": 6,
        "user_score": 0,
        "comp_score": 0,
        "phase": "choose_overs",    # choose_overs -> show_countdown -> toss -> choose_bat_bowl -> user_batting -> comp_batting -> match_over
        "toss_choice": None,       # 'odd' / 'even'
        "toss_countdown_start": None,
        "toss_winner": None,
        "bat_first": None,
        "user_last": None,
        "comp_last": None,
        "gesture_last_time": 0.0,
        "gesture_count_buf": deque(maxlen=24),  # buffer of last counts (timestamp,count)
        "last_stable_count": None,
        "last_stable_time": 0.0,
        "last_accept_time": 0.0,
        "balls_done": 0,           # balls completed in current innings
        "commentary": "",
        "match_result_time": None,
        "match_result": None,       # 'user'/'comp'/'tie'
        "series_updated": False     # to avoid double-updating series score
    }

# persistent scoreboard across continuous matches
series_score = {"user": 0, "comp": 0, "tie": 0}
state = new_match_state(overs=1)

# -------------- Gameplay helpers --------------
def stable_gesture_check(state, detected_count, now):
    """
    Maintain buffer of counts; determine if count has been stable for GESTURE_STABLE_TIME.
    Return accepted_count or None.
    """
    state['gesture_count_buf'].append((now, detected_count))
    # Remove old items older than a window slightly larger than stable time
    cutoff = now - (GESTURE_STABLE_TIME * 1.5)
    while state['gesture_count_buf'] and state['gesture_count_buf'][0][0] < cutoff:
        state['gesture_count_buf'].popleft()

    if len(state['gesture_count_buf']) >= 3:
        counts = [c for (_, c) in state['gesture_count_buf']]
        most = max(set(counts), key=counts.count)
        same_frac = counts.count(most) / len(counts)
        age = now - state['gesture_count_buf'][0][0]
        if same_frac > 0.85 and age >= GESTURE_STABLE_TIME:
            return most
    return None

def accept_gesture(state, count, now):
    """Check cooldown and accept gesture if allowed. Update last_accept_time."""
    if count < 1 or count > 6:
        return False
    if now - state['last_accept_time'] < GESTURE_COOLDOWN:
        return False
    state['last_accept_time'] = now
    return True

def comp_pick():
    return random.randint(1, 6)

def update_commentary(state, phase, user_play, comp_play):
    if phase == "user_batting":
        if user_play == comp_play:
            state['commentary'] = f"OUT! You matched {comp_play}."
        else:
            state['commentary'] = f"You scored {user_play}."
    elif phase == "comp_batting":
        if user_play == comp_play:
            state['commentary'] = f"OUT! You bowled the computer with {comp_play}."
        else:
            state['commentary'] = f"Computer scored {comp_play}."

# -------------- Main Loop --------------
running = True
clock = pygame.time.Clock()

while running:
    ret, frame = cap.read()
    if not ret:
        print("No camera. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe returns multi_hand_landmarks and multi_handedness
    results = hands_detector.process(rgb)

    # Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            # choosing overs at very start
            if state['phase'] == 'choose_overs':
                if event.key == pygame.K_1:
                    state = new_match_state(overs=1)
                    state['phase'] = 'show_countdown'
                    state['toss_countdown_start'] = time.time()
                if event.key == pygame.K_2:
                    state = new_match_state(overs=2)
                    state['phase'] = 'show_countdown'
                    state['toss_countdown_start'] = time.time()
                if event.key == pygame.K_3:
                    state = new_match_state(overs=3)
                    state['phase'] = 'show_countdown'
                    state['toss_countdown_start'] = time.time()
            # toss choices (only during toss)
            if state['phase'] == 'toss':
                if event.key == pygame.K_o:
                    state['toss_choice'] = 'odd'
                if event.key == pygame.K_e:
                    state['toss_choice'] = 'even'
            # bat/bowl choice after winning toss (user)
            if state['phase'] == 'choose_bat_bowl' and state['toss_winner'] == 'user':
                if event.key == pygame.K_b:
                    state['bat_first'] = 'user'
                    state['phase'] = 'user_batting'
                    state['balls_done'] = 0
                if event.key == pygame.K_l:
                    state['bat_first'] = 'comp'
                    state['phase'] = 'comp_batting'
                    state['balls_done'] = 0

    # detect finger count
    detected_count = None
    handedness_label = None
    if results.multi_hand_landmarks:
        # optionally get handedness label if available
        try:
            if results.multi_handedness and len(results.multi_handedness) > 0:
                handedness_label = results.multi_handedness[0].classification[0].label
        except Exception:
            handedness_label = None

        for hl in results.multi_hand_landmarks:
            detected_count = count_fingers(hl, handedness_label)
            # draw on frame for user's feedback
            try:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            except Exception:
                pass
            break  # only first hand

    now = time.time()
    stable_count = None
    if detected_count is not None:
        stable_count = stable_gesture_check(state, detected_count, now)
    else:
        # small decay of buffer when hand not visible
        if state['gesture_count_buf'] and now - state['gesture_count_buf'][-1][0] > 0.6:
            state['gesture_count_buf'].clear()

    # ---------- Game phases ----------
    if state['phase'] == 'show_countdown':
        # show countdown then go to toss
        if state['toss_countdown_start'] is None:
            state['toss_countdown_start'] = now
        elapsed = now - state['toss_countdown_start']
        if elapsed >= TOSS_COUNTDOWN:
            state['phase'] = 'toss'
            state['toss_choice'] = None

    elif state['phase'] == 'toss':
        # accept stable_count as toss gesture if user already pressed O/E
        if state['toss_choice'] in ('odd', 'even') and stable_count is not None and accept_gesture(state, stable_count, now):
            user_pick = stable_count
            comp = comp_pick()
            state['user_last'] = user_pick
            state['comp_last'] = comp
            total = user_pick + comp
            if (total % 2 == 0 and state['toss_choice'] == 'even') or (total % 2 == 1 and state['toss_choice'] == 'odd'):
                state['toss_winner'] = 'user'
            else:
                state['toss_winner'] = 'comp'
            # move to choose bat/bowl; if comp wins we auto-decide and advance to batting
            if state['toss_winner'] == 'user':
                state['phase'] = 'choose_bat_bowl'
            else:
                # comp decides randomly and we immediately go to batting
                state['bat_first'] = random.choice(['user', 'comp'])
                state['phase'] = 'user_batting' if state['bat_first'] == 'user' else 'comp_batting'
                state['balls_done'] = 0

    elif state['phase'] == 'choose_bat_bowl':
        # waiting for user to press B/L if user won toss
        pass

    elif state['phase'] == 'user_batting':
        # stable_count indicates user's run for this ball
        if stable_count is not None and accept_gesture(state, stable_count, now):
            run = stable_count
            comp = comp_pick()
            state['user_last'] = run
            state['comp_last'] = comp
            # wicket?
            if run == comp:
                state['commentary'] = f"OUT! You matched {comp}."
                # end user's innings -> comp batting begins
                state['phase'] = 'comp_batting'
                state['balls_done'] = 0
            else:
                state['user_score'] += run
                state['balls_done'] += 1
                state['commentary'] = f"You scored {run}."
                # check overs end
                if state['balls_done'] >= state['overs'] * state['balls_per_over']:
                    # user's innings complete
                    state['phase'] = 'comp_batting'
                    state['balls_done'] = 0

    elif state['phase'] == 'comp_batting':
        # comp picks its runs every ball, but user bowls via stable_count
        if stable_count is not None and accept_gesture(state, stable_count, now):
            user_bowl = stable_count
            comp_run = comp_pick()
            state['user_last'] = user_bowl
            state['comp_last'] = comp_run
            # wicket?
            if user_bowl == comp_run:
                state['commentary'] = f"OUT! You bowled computer with {comp_run}."
                # decide match result
                if state['comp_score'] > state['user_score']:
                    state['match_result'] = 'comp'
                elif state['comp_score'] == state['user_score']:
                    state['match_result'] = 'tie'
                else:
                    state['match_result'] = 'user'
                state['match_result_time'] = now
                state['phase'] = 'match_over'
                state['series_updated'] = False
            else:
                state['comp_score'] += comp_run
                state['balls_done'] += 1
                state['commentary'] = f"Computer scored {comp_run}."
                # check if comp already chased target
                if state['comp_score'] > state['user_score']:
                    state['match_result'] = 'comp'
                    state['match_result_time'] = now
                    state['phase'] = 'match_over'
                    state['series_updated'] = False
                elif state['balls_done'] >= state['overs'] * state['balls_per_over']:
                    # overs exhausted -> decide winner
                    if state['comp_score'] > state['user_score']:
                        state['match_result'] = 'comp'
                    elif state['comp_score'] == state['user_score']:
                        state['match_result'] = 'tie'
                    else:
                        state['match_result'] = 'user'
                    state['match_result_time'] = now
                    state['phase'] = 'match_over'
                    state['series_updated'] = False

    elif state['phase'] == 'match_over':
        # update series scoreboard once
        if not state.get('series_updated', False) and state['match_result'] is not None:
            if state['match_result'] == 'user':
                series_score['user'] += 1
            elif state['match_result'] == 'comp':
                series_score['comp'] += 1
            else:
                series_score['tie'] += 1
            state['series_updated'] = True

        # schedule next match reset after POST_MATCH_DELAY
        if state['match_result_time'] and (time.time() - state['match_result_time'] > POST_MATCH_DELAY):
            old_overs = state['overs']
            state = new_match_state(overs=old_overs)
            state['phase'] = 'choose_overs'
            state['gesture_count_buf'].clear()

    # ---------- Draw UI ----------
    screen.fill(BG)

    # Series scoreboard
    screen.blit(font_big.render("Hand Cricket — Series Scoreboard", True, HIGHLIGHT), (30, 18))
    screen.blit(font_med.render(f"You: {series_score['user']}", True, TEXT), (40, 70))
    screen.blit(font_med.render(f"Comp: {series_score['comp']}", True, TEXT), (180, 70))
    screen.blit(font_med.render(f"Ties: {series_score['tie']}", True, TEXT), (340, 70))

    # Phase & instructions box
    box_x, box_y = 30, 110
    pygame.draw.rect(screen, (22, 34, 60), (box_x, box_y, 440, 220), border_radius=8)
    screen.blit(font_med.render(f"Phase: {state['phase']}", True, HIGHLIGHT), (box_x + 10, box_y + 10))

    # Show helpful instructions depending on phase
    if state['phase'] == 'choose_overs':
        screen.blit(font_med.render("Choose overs (1 / 2 / 3) to start match.", True, TEXT), (box_x + 10, box_y + 60))
    elif state['phase'] == 'show_countdown':
        if state['toss_countdown_start'] is None:
            state['toss_countdown_start'] = now
        elapsed = time.time() - state['toss_countdown_start']
        left = max(0, TOSS_COUNTDOWN - int(elapsed))
        screen.blit(font_med.render(f"Toss Countdown: {left}", True, WARN), (box_x + 10, box_y + 60))
    elif state['phase'] == 'toss':
        screen.blit(font_med.render("Toss: press O (Odd) or E (Even), then show 1-6 fingers steadily.", True, TEXT), (box_x + 10, box_y + 60))
        if state['toss_choice']:
            screen.blit(font_med.render(f"You chose: {state['toss_choice'].upper()}", True, HIGHLIGHT), (box_x + 10, box_y + 100))
            if state['user_last'] is not None:
                screen.blit(font_med.render(f"Last toss picks — You: {state['user_last']}  Comp: {state['comp_last']}", True, TEXT), (box_x + 10, box_y + 140))
    elif state['phase'] == 'choose_bat_bowl':
        if state['toss_winner'] == 'user':
            screen.blit(font_med.render("You won toss! Press B to Bat or L to Bowl.", True, HIGHLIGHT), (box_x + 10, box_y + 60))
        else:
            if state['bat_first'] == 'user':
                screen.blit(font_med.render("Computer won toss and chose: YOU to Bat first.", True, HIGHLIGHT), (box_x + 10, box_y + 60))
            else:
                screen.blit(font_med.render("Computer won toss and chose: COMPUTER to Bat first.", True, HIGHLIGHT), (box_x + 10, box_y + 60))
    elif state['phase'] in ('user_batting', 'comp_batting'):
        total_balls = state['overs'] * state['balls_per_over']
        left = total_balls - state['balls_done']
        screen.blit(font_med.render(f"Overs: {state['overs']}    Balls left in innings: {left}", True, TEXT), (box_x + 10, box_y + 60))
        if state['phase'] == 'comp_batting':
            target = state['user_score'] + 1
            screen.blit(font_med.render(f"Target for Comp: {target}", True, HIGHLIGHT), (box_x + 10, box_y + 100))
    elif state['phase'] == 'match_over':
        screen.blit(font_med.render("Match Over", True, HIGHLIGHT), (box_x + 10, box_y + 60))
        if state['match_result']:
            txt = "You Win!" if state['match_result'] == 'user' else ("Computer Wins!" if state['match_result'] == 'comp' else "It's a Tie!")
            screen.blit(font_med.render(txt, True, COMMENT), (box_x + 10, box_y + 100))

    # Commentary box
    pygame.draw.rect(screen, (18, 26, 46), (30, 350, 440, 120), border_radius=8)
    screen.blit(font_med.render("Commentary", True, HIGHLIGHT), (40, 360))
    screen.blit(font_small.render(state['commentary'] or "-", True, COMMENT), (40, 400))

    # Right panel: camera preview and last plays
    cam_x, cam_y = 500, 20
    cam_w, cam_h = 460, 345
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (cam_w, cam_h))
        surf = pygame.image.frombuffer(frame_rgb.tobytes(), (cam_w, cam_h), "RGB")
        screen.blit(surf, (cam_x, cam_y))
    except Exception:
        # If frame conversion fails, don't crash UI
        pygame.draw.rect(screen, (40, 40, 50), (cam_x, cam_y, cam_w, cam_h))
    pygame.draw.rect(screen, (50, 50, 70), (cam_x-2, cam_y-2, cam_w+4, cam_h+4), 2, border_radius=6)
    screen.blit(font_small.render("Camera (show 1-6 fingers)", True, TEXT), (cam_x, cam_y + cam_h + 6))

    # Last plays & scores
    info_x = cam_x
    info_y = cam_y + cam_h + 40
    screen.blit(font_med.render(f"User Score: {state['user_score']}", True, TEXT), (info_x, info_y))
    screen.blit(font_med.render(f"Comp Score: {state['comp_score']}", True, TEXT), (info_x, info_y + 30))
    screen.blit(font_med.render(f"Last — You: {state['user_last'] if state['user_last'] is not None else '-'}  Comp: {state['comp_last'] if state['comp_last'] is not None else '-'}", True, TEXT), (info_x, info_y + 70))
    # quick tips
    screen.blit(font_small.render("Controls: 1/2/3 overs | O/E toss | B/L bat/bowl | Q/Esc quit", True, (200,200,200)), (30, 500))

    pygame.display.flip()

    # show cv2 small window as backup (optional)
    try:
        cv2.imshow("Hand Tracking (ESC/Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
    except Exception:
        # if cv window fails (headless env), ignore
        pass

    clock.tick(30)

# Cleanup
try:
    hands_detector.close()
except Exception:
    pass
cap.release()
cv2.destroyAllWindows()
pygame.quit()
