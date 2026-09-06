"""
Progress You Can See — Stage 2.2

This module exists because the sales pitch names three differentiators —
Smart Review Scheduling, Listen & Correct, Progress You Can See — and only
the first two existed in code. This is the third: a color-coded map of the
user's entire memorized Quran, driven by real grading history, not a
decorative mock-up.

Design intent, matching the pitch's own language ("a visual map of your
entire memorized Quran, color-coded by strength, so you watch your hard
work turn into something real"):

  - One tile per surah the user has ever attempted.
  - Tile color = mastery band, derived from the *most recent* graded
    attempt on each ayah in that surah (not an all-time average, so a
    surah that used to be weak but was recently re-mastered shows green
    immediately — consistent with how the SM-2 review queue already
    treats "most recent evidence" as the source of truth).
  - Un-attempted surahs render as empty outline tiles, not colored —
    "not started" must never look like "weak", or the map lies to the
    user about where the real risk is.

No new tables required: this reads session_words (join sessions for
recency) which already stores per-word status for every graded attempt,
plus review_state for due/overdue flags. If/when this moves off SQLite
onto Supabase Postgres, only get_conn()'s connection layer changes — the
SQL below is intentionally plain (no SQLite-only functions) so it ports
as-is.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

import db

# Mastery bands, keyed to the app's existing design tokens so this feature
# never introduces a fifth color language on top of the grading legend.
_BANDS = [
    # (floor, css_var, label)
    (0.90, "var(--sage)",      "Mastered"),
    (0.70, "var(--gilding)",   "Solid — needs upkeep"),
    (0.40, "var(--terracotta)", "Weak"),
    (0.00, "var(--terracotta-dim)", "Struggling"),
]

_NOT_STARTED_COLOR = "var(--ink-700)"


def _band_for(score: float) -> tuple:
    for floor, color, label in _BANDS:
        if score >= floor:
            return color, label
    return _BANDS[-1][1], _BANDS[-1][2]


def get_surah_mastery(user_id: int) -> dict:
    """
    Returns {surah_num: {"score": float 0-1, "attempted": int, "due": int}}

    score is computed from the *latest* session covering each ayah in that
    surah (latest session_id per (surah, ayah), weighted by word-level
    correctness in that attempt) — deliberately not an all-time average,
    per the module docstring above.
    """
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            WITH latest_session_per_ayah AS (
                SELECT sw.surah, sw.ayah, MAX(sw.session_id) AS session_id
                FROM session_words sw
                JOIN sessions s ON s.id = sw.session_id
                WHERE s.user_id = ?
                GROUP BY sw.surah, sw.ayah
            )
            SELECT sw.surah, sw.ayah,
                   SUM(CASE WHEN sw.status IN ('correct','close') THEN 1 ELSE 0 END) AS correct_n,
                   SUM(CASE WHEN sw.status != 'not_recited' THEN 1 ELSE 0 END) AS scored_n
            FROM session_words sw
            JOIN latest_session_per_ayah l
              ON l.surah = sw.surah AND l.ayah = sw.ayah AND l.session_id = sw.session_id
            GROUP BY sw.surah, sw.ayah
            """,
            (user_id,),
        ).fetchall()

        due_rows = conn.execute(
            "SELECT surah, COUNT(*) AS n FROM review_state "
            "WHERE user_id = ? AND next_due_date <= ? GROUP BY surah",
            (user_id, date.today().isoformat()),
        ).fetchall()

    per_surah: dict = {}
    for r in rows:
        d = per_surah.setdefault(r["surah"], {"correct": 0, "scored": 0, "ayahs": 0})
        if r["scored_n"]:
            d["correct"] += r["correct_n"]
            d["scored"] += r["scored_n"]
            d["ayahs"] += 1

    due_by_surah = {r["surah"]: r["n"] for r in due_rows}

    result = {}
    for surah, d in per_surah.items():
        score = (d["correct"] / d["scored"]) if d["scored"] else 0.0
        result[surah] = {
            "score": score,
            "attempted": d["ayahs"],
            "due": due_by_surah.get(surah, 0),
        }
    return result


def render_progress_map(quran: dict, surah_numbers: list) -> None:
    """Renders the full 114-surah mastery grid plus a summary strip."""
    user_id = st.session_state.user_id
    mastery = get_surah_mastery(user_id)

    attempted = [s for s in surah_numbers if s in mastery]
    if not attempted:
        st.info(
            "Your progress map fills in as you recite. Head to **Listen & "
            "Correct**, complete a few ayahs, and this page will start "
            "coloring in — that's the whole point: nothing to look at "
            "until there's something real behind it."
        )
        return

    overall_score = sum(mastery[s]["score"] for s in attempted) / len(attempted)
    mastered_n = sum(1 for s in attempted if mastery[s]["score"] >= 0.90)
    weak_n = sum(1 for s in attempted if mastery[s]["score"] < 0.40)
    due_n = sum(mastery[s]["due"] for s in attempted)

    st.markdown(
        f"""
        <div class="progress-summary-row">
          <div class="progress-stat">
            <div class="progress-stat-value" style="color:var(--gilding);">{len(attempted)}<span class="progress-stat-unit">/114</span></div>
            <div class="progress-stat-label">Surahs started</div>
          </div>
          <div class="progress-stat">
            <div class="progress-stat-value" style="color:var(--sage);">{mastered_n}</div>
            <div class="progress-stat-label">Mastered</div>
          </div>
          <div class="progress-stat">
            <div class="progress-stat-value" style="color:var(--terracotta);">{weak_n}</div>
            <div class="progress-stat-label">Weak — at risk</div>
          </div>
          <div class="progress-stat">
            <div class="progress-stat-value" style="color:var(--parchment);">{overall_score * 100:.0f}%</div>
            <div class="progress-stat-label">Overall strength</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if due_n > 0:
        st.markdown(
            f'<p class="progress-due-banner">🔔 <strong>{due_n}</strong> ayah(s) across your '
            f"memorized surahs are due for review today — check <strong>Today's Review</strong> above.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="progress-legend">'
        '<span><i style="background:var(--sage);"></i> Mastered</span>'
        '<span><i style="background:var(--gilding);"></i> Solid</span>'
        '<span><i style="background:var(--terracotta);"></i> Weak</span>'
        '<span><i style="background:var(--terracotta-dim);"></i> Struggling</span>'
        '<span><i style="background:var(--ink-700);border:1px dashed var(--muted);"></i> Not started</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    tiles = []
    for s in surah_numbers:
        name = quran.get(s, {}).get("name", f"Surah {s}")
        if s in mastery:
            score = mastery[s]["score"]
            color, band_label = _band_for(score)
            due = mastery[s]["due"]
            due_dot = '<span class="tile-due-dot"></span>' if due else ""
            title = f"{name} — {score * 100:.0f}% ({band_label})" + (f" — {due} due" if due else "")
            tiles.append(
                f'<div class="surah-tile" style="background:{color};" title="{title}">'
                f'<span class="tile-num">{s}</span>{due_dot}</div>'
            )
        else:
            tiles.append(
                f'<div class="surah-tile surah-tile-empty" title="{name} — not started">'
                f'<span class="tile-num">{s}</span></div>'
            )

    st.markdown(f'<div class="surah-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)

    if weak_n > 0:
        st.markdown("---")
        st.markdown("#### Weakest surahs — recite these next")
        weakest = sorted(
            (s for s in attempted if mastery[s]["score"] < 0.70),
            key=lambda s: mastery[s]["score"],
        )[:5]
        for s in weakest:
            name = quran.get(s, {}).get("name", f"Surah {s}")
            score = mastery[s]["score"]
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f'<div class="weak-row"><span>{name} <span class="muted">'
                    f"(Surah {s})</span></span>"
                    f'<span class="weak-row-score" style="color:{_band_for(score)[0]};">'
                    f"{score * 100:.0f}%</span></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Practice", key=f"practice_weak_{s}", use_container_width=True):
                    st.session_state.jump_from_surah = s
                    st.session_state.jump_to_surah = s
                    st.session_state.app_mode = "🎙 Listen & Correct"
                    st.rerun()
