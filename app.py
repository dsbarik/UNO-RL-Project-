import os
import time

import numpy as np
import streamlit as st
import torch

from environment import Card, CardColor, CardValue, Deck
from model import DQN

# --- Helper Functions ---


def get_state_vector(hand, discard_counts, current_color, opponent_hand_size):
    hand_vector = np.zeros(54, dtype=np.float32)
    for card in hand:
        hand_vector[card.to_index()] += 1

    color_offsets = {
        CardColor.RED: 0,
        CardColor.BLUE: 1,
        CardColor.GREEN: 2,
        CardColor.YELLOW: 3,
    }

    color_one_hot = np.zeros(4, dtype=np.float32)
    if current_color in color_offsets:
        color_one_hot[color_offsets[current_color]] = 1.0

    return np.concatenate([
        hand_vector,
        discard_counts,
        color_one_hot,
        np.array([opponent_hand_size], dtype=np.float32),
    ])


def get_legal_actions(hand, top_card, current_color):
    legal = []
    for idx, card in enumerate(hand):
        if card.color == CardColor.WILD:
            legal.append(idx)
        elif card.color == current_color:
            legal.append(idx)
        elif top_card.color != CardColor.WILD and card.value == top_card.value:
            legal.append(idx)
    return legal


def reshuffle_discard():
    # Remove top card from counts
    st.session_state.discard_counts[st.session_state.top_card.to_index()] -= 1

    # Rebuild cards
    new_cards = []
    for idx, count in enumerate(st.session_state.discard_counts):
        for _ in range(int(count)):
            new_cards.append(Card.from_index(idx))

    st.session_state.deck.cards = new_cards
    st.session_state.deck.shuffle()

    # Reset counts
    st.session_state.discard_counts = np.zeros(54, dtype=np.float32)
    st.session_state.discard_counts[st.session_state.top_card.to_index()] = 1.0
    st.session_state.logs.insert(
        0, "🔄 The discard pile was reshuffled into a new draw deck!"
    )


def safe_draw():
    if len(st.session_state.deck.cards) == 0:
        reshuffle_discard()
    if len(st.session_state.deck.cards) > 0:
        return st.session_state.deck.draw()
    return None


def deal_penalty(hand, count):
    """Safely draws multiple cards for +2 and +4 penalties."""
    for _ in range(count):
        card = safe_draw()
        if card:
            hand.append(card)


# --- Game Initialization ---


def init_game():
    st.session_state.deck = Deck()
    st.session_state.discard_counts = np.zeros(54, dtype=np.float32)
    st.session_state.human_hand = [st.session_state.deck.draw() for _ in range(7)]
    st.session_state.ai_hand = [st.session_state.deck.draw() for _ in range(7)]

    top_card = st.session_state.deck.draw()
    st.session_state.discard_counts[top_card.to_index()] += 1
    st.session_state.top_card = top_card

    if top_card.color == CardColor.WILD:
        # NumPy string truncation fix: use randint to select from a list
        base_colors = [CardColor.RED, CardColor.BLUE, CardColor.GREEN, CardColor.YELLOW]
        st.session_state.current_color = base_colors[np.random.randint(0, 4)]
    else:
        st.session_state.current_color = top_card.color

    st.session_state.turn = "Human"
    st.session_state.logs = ["Game started. Good luck!"]
    st.session_state.game_over = False
    st.session_state.awaiting_wild_color = False
    st.session_state.pending_wild_card = None
    st.session_state.game_started = True


# Load Model once and cache it
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DQN().to(device)
    if os.path.exists("uno_agent.pth"):
        model.load_state_dict(torch.load("uno_agent.pth", map_location=device))
    model.eval()
    return model, device


# --- Streamlit UI Config ---
st.set_page_config(page_title="Deep UNO Arena", page_icon="🃏", layout="wide")

ai_model, device = load_model()

# ── Vivid, highly distinct color palette ──
COLOR_RED = "#DC2626"
COLOR_BLUE = "#2563EB"
COLOR_GREEN = "#16A34A"
COLOR_YELLOW = "#EAB308"
COLOR_WILD = "#7C3AED"

color_map = {
    CardColor.RED: COLOR_RED,
    CardColor.BLUE: COLOR_BLUE,
    CardColor.GREEN: COLOR_GREEN,
    CardColor.YELLOW: COLOR_YELLOW,
    CardColor.WILD: COLOR_WILD,
}

# Text color that contrasts well on each card background
text_on_card = {
    CardColor.RED: "#fff",
    CardColor.BLUE: "#fff",
    CardColor.GREEN: "#fff",
    CardColor.YELLOW: "#1a1a1a",
    CardColor.WILD: "#fff",
}

card_value_display = {
    CardValue.ZERO: "0",
    CardValue.ONE: "1",
    CardValue.TWO: "2",
    CardValue.THREE: "3",
    CardValue.FOUR: "4",
    CardValue.FIVE: "5",
    CardValue.SIX: "6",
    CardValue.SEVEN: "7",
    CardValue.EIGHT: "8",
    CardValue.NINE: "9",
    CardValue.SKIP: "⊘",
    CardValue.REVERSE: "⟲",
    CardValue.DRAW_TWO: "+2",
    CardValue.STANDARD: "W",
    CardValue.WILD_FOUR: "+4",
}

color_name_display = {
    CardColor.RED: "RED",
    CardColor.BLUE: "BLUE",
    CardColor.GREEN: "GREEN",
    CardColor.YELLOW: "YELLOW",
    CardColor.WILD: "WILD",
}


def render_card_html(card, size="small", playable=False, dim=False):
    """Renders a single card as styled HTML. All styles on one line to avoid Streamlit parsing issues."""
    bg = color_map.get(card.color, "#555")
    txt = text_on_card.get(card.color, "#fff")
    val = card_value_display.get(card.value, "?")
    label = color_name_display.get(card.color, "")

    if size == "large":
        w, h, fs_val, fs_label = "130px", "190px", "44px", "11px"
    else:
        w, h, fs_val, fs_label = "72px", "100px", "22px", "8px"

    opacity = "0.35" if dim else "1"
    border = "3px solid #fff" if playable else "2px solid rgba(255,255,255,0.2)"
    shadow = "0 4px 16px rgba(0,0,0,0.35)" if playable else "0 2px 8px rgba(0,0,0,0.2)"

    style = (
        f"width:{w};height:{h};background:{bg};border-radius:12px;"
        f"display:flex;flex-direction:column;align-items:center;justify-content:center;"
        f"color:{txt};font-family:Inter,sans-serif;font-weight:800;font-size:{fs_val};"
        f"border:{border};box-shadow:{shadow};opacity:{opacity};"
        f"text-shadow:0 1px 3px rgba(0,0,0,0.3);flex-shrink:0;"
    )
    lbl_style = f"font-size:{fs_label};font-weight:700;letter-spacing:1px;margin-top:4px;opacity:0.85;"

    return f'<div style="{style}">{val}<span style="{lbl_style}">{label}</span></div>'



# ─────────────────────────── Global CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Base */
.stApp { font-family: 'Inter', -apple-system, sans-serif !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1100px !important;
}

/* Section header */
.section-hdr {
    font-family: 'Inter', sans-serif;
    font-size: 12px; font-weight: 700;
    color: #888; text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #e0e0e0;
}
[data-testid="stAppViewContainer"][data-theme="dark"] .section-hdr {
    color: #888; border-bottom-color: #333;
}

/* Stat card */
.scard {
    background: #f8f8f8;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
[data-testid="stAppViewContainer"][data-theme="dark"] .scard {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.1);
}
.scard-val {
    font-size: 32px; font-weight: 800;
    font-family: 'Inter', sans-serif;
    line-height: 1;
}
.scard-lbl {
    font-size: 10px; font-weight: 700;
    color: #999; text-transform: uppercase;
    letter-spacing: 1.5px; margin-top: 4px;
}

/* Log */
.log-item {
    font-family: 'Inter', sans-serif;
    font-size: 13px; color: #777;
    padding: 5px 0;
    border-bottom: 1px solid rgba(0,0,0,0.04);
    line-height: 1.5;
}
.log-item:first-child { color: #333; font-weight: 600; }
[data-testid="stAppViewContainer"][data-theme="dark"] .log-item { color: #999; border-bottom-color: rgba(255,255,255,0.04); }
[data-testid="stAppViewContainer"][data-theme="dark"] .log-item:first-child { color: #eee; }

/* Turn badge */
.tbadge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.tbadge-you { background: rgba(22,163,74,0.12); color: #16A34A; border: 1px solid rgba(22,163,74,0.3); }
.tbadge-ai { background: rgba(220,38,38,0.12); color: #DC2626; border: 1px solid rgba(220,38,38,0.3); }

/* Active color pill */
.color-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 18px; border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 700; color: #fff;
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-top: 10px;
}
.color-dot {
    width: 9px; height: 9px; border-radius: 50%; background: #fff;
}

/* UNO alert */
.uno-alert {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 14px; border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 12px; font-weight: 800;
    color: #DC2626; letter-spacing: 1px;
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.3);
}

/* Hand card grid */
.hand-grid {
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 8px 0;
}

/* Landing */
.landing-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 68vh; text-align: center;
    padding: 32px 20px;
}
.landing-title {
    font-family: 'Inter', sans-serif;
    font-size: 64px; font-weight: 900;
    background: linear-gradient(135deg, #DC2626, #EAB308, #16A34A, #2563EB, #7C3AED);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gshift 5s ease infinite;
    line-height: 1.1; margin-bottom: 8px;
}
@keyframes gshift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.landing-sub {
    font-family: 'Inter', sans-serif;
    font-size: 17px; font-weight: 500;
    color: #888; max-width: 480px;
    line-height: 1.6; margin-bottom: 36px;
}
.landing-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 600;
    color: #7C3AED; letter-spacing: 0.5px;
    margin-bottom: 20px;
}
.landing-cards {
    display: flex; gap: 12px; margin-bottom: 40px;
    flex-wrap: wrap; justify-content: center;
}

/* Game over */
.gameover-box {
    text-align: center; padding: 36px 20px;
    border-radius: 16px; margin: 20px 0;
    border: 1px solid rgba(0,0,0,0.08);
}
.gameover-box h1 {
    font-family: 'Inter', sans-serif;
    font-size: 42px; font-weight: 900;
    margin: 0; line-height: 1.2;
}
.gameover-box p {
    font-family: 'Inter', sans-serif;
    font-size: 15px; color: #888; margin-top: 6px;
}

/* Custom divider */
.cdiv { height: 1px; background: #e5e5e5; margin: 16px 0; }
[data-testid="stAppViewContainer"][data-theme="dark"] .cdiv { background: #333; }

/* Hide chrome - modified to show deploy button */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────── LANDING PAGE / START SCREEN ───────────────
if not st.session_state.get("game_started", False):
    # Build landing card HTML
    landing_cards = ""
    demo = [
        (COLOR_RED, "7"), (COLOR_BLUE, "⊘"), (COLOR_GREEN, "⟲"),
        (COLOR_YELLOW, "+2"), (COLOR_WILD, "W"),
    ]
    for bg, val in demo:
        txt_c = "#1a1a1a" if bg == COLOR_YELLOW else "#fff"
        landing_cards += f'<div style="width:60px;height:86px;border-radius:12px;background:{bg};display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:800;font-family:Inter,sans-serif;color:{txt_c};box-shadow:0 3px 12px rgba(0,0,0,0.2);border:1.5px solid rgba(255,255,255,0.2);text-shadow:0 1px 4px rgba(0,0,0,0.3);">{val}</div>'

    st.markdown(f"""
    <div class="landing-wrap">
        <div class="landing-badge">🧠 Powered by Deep Q-Network</div>
        <div class="landing-title">DEEP UNO</div>
        <div class="landing-sub">
            Challenge a neural network trained through reinforcement learning.<br>
            Can you outsmart the machine?
        </div>
        <div class="landing-cards">{landing_cards}</div>
    </div>
    """, unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 1, 1])
    with col_center:
        if st.button("🎮  Start Game", use_container_width=True, type="primary"):
            init_game()
            st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:40px; opacity:0.35;">
        <p style="font-family:'Inter',sans-serif; font-size:11px; letter-spacing:1px;">
            PYTORCH &nbsp;•&nbsp; STREAMLIT &nbsp;•&nbsp; DQN AGENT
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ─────────────────── GAME LOGIC FUNCTIONS ───────────────────


def ai_play_turn():
    if st.session_state.game_over:
        return

    # --- 1. Generate 61-dim mask for AI ---
    mask = np.zeros(61, dtype=np.float32)
    has_legal = False
    for card in st.session_state.ai_hand:
        idx = card.to_index()
        if card.color == CardColor.WILD:
            if card.value == CardValue.STANDARD:
                mask[52:56] = 1.0
            else:
                mask[56:60] = 1.0
            has_legal = True
        elif card.color == st.session_state.current_color or (
            st.session_state.top_card.color != CardColor.WILD
            and card.value == st.session_state.top_card.value
        ):
            mask[idx] = 1.0
            has_legal = True

    if not has_legal:
        mask[60] = 1.0

    # --- 2. Get AI Prediction ---
    state_vector = get_state_vector(
        st.session_state.ai_hand,
        st.session_state.discard_counts,
        st.session_state.current_color,
        len(st.session_state.human_hand),
    )
    state_t = torch.FloatTensor(state_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = ai_model(state_t).squeeze(0).cpu().numpy()

    q_values = q_values + (1.0 - mask) * -1e9
    best_action_index = np.argmax(q_values)

    # --- 3. Execute AI Action ---
    if best_action_index == 60:
        st.session_state.logs.insert(0, "🤖 AI had no legal moves. Drew a card.")
        drawn_card = safe_draw()
        if drawn_card:
            st.session_state.ai_hand.append(drawn_card)
        st.session_state.turn = "Human"
        return

    # Map the action back to the physical card
    base_colors = [CardColor.RED, CardColor.BLUE, CardColor.GREEN, CardColor.YELLOW]
    chosen_color = None

    if 52 <= best_action_index <= 55:
        target_idx = 52
        chosen_color = base_colors[best_action_index - 52]
    elif 56 <= best_action_index <= 59:
        target_idx = 53
        chosen_color = base_colors[best_action_index - 56]
    else:
        target_idx = best_action_index

    played_card = next(
        c for c in st.session_state.ai_hand if c.to_index() == target_idx
    )
    st.session_state.ai_hand.remove(played_card)

    st.session_state.top_card = played_card
    st.session_state.discard_counts[target_idx] += 1

    # --- 4. Apply Effects ---
    if played_card.color == CardColor.WILD:
        st.session_state.current_color = chosen_color
        st.session_state.logs.insert(
            0, f"🤖 AI played {played_card.value} and chose {chosen_color.value}!"
        )

        if played_card.value == CardValue.WILD_FOUR:
            deal_penalty(st.session_state.human_hand, 4)
            st.session_state.logs.insert(
                0, "🚨 You were forced to draw 4 cards! Turn skipped."
            )
        else:
            st.session_state.turn = "Human"
    else:
        st.session_state.current_color = played_card.color
        st.session_state.logs.insert(
            0, f"🤖 AI played {played_card.color} {played_card.value}."
        )

        if played_card.value == CardValue.DRAW_TWO:
            deal_penalty(st.session_state.human_hand, 2)
            st.session_state.logs.insert(
                0, "🚨 You were forced to draw 2 cards! Turn skipped."
            )
        elif played_card.value in [CardValue.SKIP, CardValue.REVERSE]:
            st.session_state.logs.insert(0, "⏭️ AI skipped your turn!")
        else:
            st.session_state.turn = "Human"

    check_win()


def play_card(idx):
    card = st.session_state.human_hand.pop(idx)
    st.session_state.top_card = card
    st.session_state.discard_counts[card.to_index()] += 1

    if card.color == CardColor.WILD:
        st.session_state.awaiting_wild_color = True
        st.session_state.pending_wild_card = card
        st.session_state.logs.insert(0, "✨ You played a Wild Card! Choose a color.")
    else:
        st.session_state.current_color = card.color
        st.session_state.logs.insert(0, f"👤 You played {card.color} {card.value}.")

        if card.value == CardValue.DRAW_TWO:
            deal_penalty(st.session_state.ai_hand, 2)
            st.session_state.logs.insert(
                0, "🚨 AI was forced to draw 2 cards! You go again."
            )
        elif card.value in [CardValue.SKIP, CardValue.REVERSE]:
            st.session_state.logs.insert(
                0, "⏭️ You skipped the AI's turn! You go again."
            )
        else:
            st.session_state.turn = "AI"

        check_win()


def set_wild_color(color):
    st.session_state.current_color = color
    st.session_state.awaiting_wild_color = False
    st.session_state.logs.insert(0, f"🎨 Color changed to {color.value}.")

    if st.session_state.pending_wild_card.value == CardValue.WILD_FOUR:
        deal_penalty(st.session_state.ai_hand, 4)
        st.session_state.logs.insert(
            0, "🚨 AI was forced to draw 4 cards! You go again."
        )
    else:
        st.session_state.turn = "AI"

    check_win()


def check_win():
    if len(st.session_state.human_hand) == 0:
        st.session_state.game_over = True
        st.session_state.logs.insert(0, "🏆 YOU WIN!")
    elif len(st.session_state.ai_hand) == 0:
        st.session_state.game_over = True
        st.session_state.logs.insert(0, "💀 AI WINS!")


def draw_card():
    drawn = safe_draw()
    if drawn:
        st.session_state.human_hand.append(drawn)
        st.session_state.logs.insert(0, "👤 You drew a card.")
    st.session_state.turn = "AI"


# ────────────────────── UI RENDERING ──────────────────────

# AI auto-play
if st.session_state.turn == "AI" and not st.session_state.game_over:
    with st.spinner("🤖 AI is thinking…"):
        time.sleep(0.4)
        ai_play_turn()
        st.rerun()

# ── Game Over State ──
if st.session_state.game_over:
    human_won = len(st.session_state.human_hand) == 0
    bg = "rgba(22,163,74,0.06)" if human_won else "rgba(220,38,38,0.06)"
    emoji = "🏆" if human_won else "💀"
    title = "Victory!" if human_won else "Defeated"
    subtitle = "You outsmarted the DQN agent!" if human_won else "The neural network prevails."
    color = "#16A34A" if human_won else "#DC2626"

    st.markdown(f"""
    <div class="gameover-box" style="background: {bg};">
        <div style="font-size: 56px; margin-bottom: 10px;">{emoji}</div>
        <h1 style="color: {color};">{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄  Play Again", use_container_width=True, type="primary"):
            init_game()
            st.rerun()
    st.stop()


# ── Header ──
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-family:'Inter',sans-serif; font-size:24px; font-weight:900;">🃏 DEEP UNO</span>
        <span class="landing-badge" style="margin-bottom:0;">ARENA</span>
    </div>
    """, unsafe_allow_html=True)
with h_right:
    if st.session_state.turn == "Human":
        st.markdown('<div style="text-align:right;"><span class="tbadge tbadge-you">● Your Turn</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right;"><span class="tbadge tbadge-ai">● AI Turn</span></div>', unsafe_allow_html=True)

st.markdown('<div class="cdiv"></div>', unsafe_allow_html=True)


# ── Main layout ──
game_col, sidebar_col = st.columns([3, 1], gap="large")

with sidebar_col:
    # Stats
    st.markdown('<div class="section-hdr">Game Stats</div>', unsafe_allow_html=True)

    ai_count = len(st.session_state.ai_hand)
    ai_val_color = COLOR_RED if ai_count <= 2 else "inherit"
    st.markdown(f"""
    <div class="scard" style="margin-bottom:10px;">
        <div class="scard-val" style="color:{ai_val_color};">🤖 {ai_count}</div>
        <div class="scard-lbl">AI Cards</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="scard" style="margin-bottom:10px;">
        <div class="scard-val">🂠 {len(st.session_state.deck.cards)}</div>
        <div class="scard-lbl">Draw Pile</div>
    </div>
    """, unsafe_allow_html=True)

    if ai_count == 1:
        st.markdown('<div style="text-align:center; margin:8px 0;"><span class="uno-alert">🚨 AI says UNO!</span></div>', unsafe_allow_html=True)

    # Match log
    st.markdown('<div class="section-hdr" style="margin-top:18px;">Match Log</div>', unsafe_allow_html=True)
    for log in st.session_state.logs[:8]:
        st.markdown(f'<div class="log-item">{log}</div>', unsafe_allow_html=True)

    st.markdown('<div class="cdiv"></div>', unsafe_allow_html=True)
    if st.button("🔄  Restart Game", use_container_width=True):
        init_game()
        st.rerun()


with game_col:
    # ── Top Card ──
    st.markdown('<div class="section-hdr">Discard Pile</div>', unsafe_allow_html=True)

    top = st.session_state.top_card
    top_card_html = render_card_html(top, size="large")

    # Active color
    c_color = color_map[st.session_state.current_color]
    display_color = (
        st.session_state.current_color.value
        if hasattr(st.session_state.current_color, "value")
        else st.session_state.current_color
    )

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:center; padding: 10px 0 6px 0;">
        {top_card_html}
        <div class="color-pill" style="background:{c_color};">
            <span class="color-dot"></span>
            Active: {display_color}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cdiv"></div>', unsafe_allow_html=True)

    # ── Your Hand ──
    human_count = len(st.session_state.human_hand)
    uno_badge_html = '&nbsp;&nbsp;<span class="uno-alert">UNO!</span>' if human_count == 1 else ""
    hand_title = f"Your Hand — {human_count} card{'s' if human_count != 1 else ''}{uno_badge_html}"
    st.markdown(f'<div class="section-hdr">{hand_title}</div>', unsafe_allow_html=True)

    legal_moves = get_legal_actions(
        st.session_state.human_hand,
        st.session_state.top_card,
        st.session_state.current_color,
    )

    # Wild color chooser
    if st.session_state.awaiting_wild_color:
        st.markdown("""
        <div style="padding:12px 16px; border-radius:12px; background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.25); margin-bottom:12px;">
            <span style="font-family:'Inter',sans-serif; font-size:14px; font-weight:700; color:#7C3AED;">✨ Choose the new color:</span>
        </div>
        """, unsafe_allow_html=True)

        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            if st.button("🔴 Red", key="wc_red", use_container_width=True):
                set_wild_color(CardColor.RED)
                st.rerun()
        with wc2:
            if st.button("🔵 Blue", key="wc_blue", use_container_width=True):
                set_wild_color(CardColor.BLUE)
                st.rerun()
        with wc3:
            if st.button("🟢 Green", key="wc_green", use_container_width=True):
                set_wild_color(CardColor.GREEN)
                st.rerun()
        with wc4:
            if st.button("🟡 Yellow", key="wc_yellow", use_container_width=True):
                set_wild_color(CardColor.YELLOW)
                st.rerun()

        st.markdown('<div class="cdiv"></div>', unsafe_allow_html=True)

    # ── Render hand as clickable card-styled buttons ──
    cards_per_row = 7
    for row_start in range(0, len(st.session_state.human_hand), cards_per_row):
        row_cards = list(enumerate(st.session_state.human_hand))[row_start:row_start + cards_per_row]
        cols = st.columns(cards_per_row)

        for col_idx, (idx, card) in enumerate(row_cards):
            is_legal = (idx in legal_moves) and not st.session_state.awaiting_wild_color
            is_playable = is_legal and st.session_state.turn == "Human"

            bg = color_map.get(card.color, "#555")
            txt = text_on_card.get(card.color, "#fff")
            val = card_value_display.get(card.value, "?")
            clabel = color_name_display.get(card.color, "")
            btn_key = f"c_{idx}_{card.to_index()}_{row_start}"
            opacity = "1" if is_playable else "0.4"
            border_col = "#fff" if is_playable else "rgba(255,255,255,0.15)"


            with cols[col_idx]:
                # Per-button card styling via a wrapping div with a unique class
                card_class = f"card-btn-{idx}-{row_start}"
                st.markdown(f"""<style>
                .{card_class} button {{
                    background: {bg} !important;
                    color: {txt} !important;
                    border: 2px solid {border_col} !important;
                    border-radius: 12px !important;
                    min-height: 100px !important;
                    font-size: 20px !important;
                    font-weight: 800 !important;
                    font-family: 'Inter', sans-serif !important;
                    opacity: {opacity} !important;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
                    padding: 8px 4px !important;
                    line-height: 1.2 !important;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
                }}
                .{card_class} button:hover:not(:disabled) {{
                    opacity: 1 !important;
                    border-color: #fff !important;
                    box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
                }}
                .{card_class} button:disabled {{
                    opacity: 0.35 !important;
                }}
                </style>""", unsafe_allow_html=True)

                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                if st.button(
                    f"{val}\n{clabel}",
                    key=btn_key,
                    disabled=not is_playable,
                    use_container_width=True,
                ):
                    play_card(idx)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # No legal moves → draw
    if (
        not legal_moves
        and st.session_state.turn == "Human"
        and not st.session_state.game_over
        and not st.session_state.awaiting_wild_color
    ):
        st.markdown("""
        <div style="padding:12px 16px; border-radius:12px; background:rgba(220,38,38,0.06); border:1px solid rgba(220,38,38,0.15); margin-top:10px; margin-bottom:6px;">
            <span style="font-family:'Inter',sans-serif; font-size:14px; font-weight:600; color:#DC2626;">No playable cards — you must draw.</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📥  Draw a Card", use_container_width=True, type="primary"):
            draw_card()
            st.rerun()

