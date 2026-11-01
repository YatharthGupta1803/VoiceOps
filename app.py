"""Streamlit Voice & Text UI for ZentroMart VoiceOps AI.

Incorporates the 'Luminous Resonance' editorial design system:
- High-editorial serif typography (Newsreader) and geometric sans (Plus Jakarta Sans)
- Frosted glassmorphic cards and floating pill containers
- Luminous ambient mesh palette: Primary Violet (#8B5CF6), Petal Magenta (#EC4899), Soft Iris (#A78BFA)
- Real-time speech input (browser microphone) & Gnani TTS synthesis playback
- Live caller verification & active order telemetrics
"""

import os
import re
import sys
import tempfile
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.voice_agent import VoiceOpsAgent, normalize_text_for_speech
from src.retriever import HybridRetriever

# Page configuration
st.set_page_config(
    page_title="ZentroMart VoiceOps — Ethereal Acoustic Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS ("Luminous Resonance")
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Background Canvas with Subtle Ambient Mesh */
    .stApp {
        background-color: #fbf8fc;
        background-image: 
            radial-gradient(at 0% 0%, rgba(243, 232, 255, 0.6) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(252, 231, 243, 0.5) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(233, 221, 255, 0.4) 0px, transparent 50%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1b1b1e;
    }

    /* Editorial Display Typography */
    .hero-eyebrow {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b38d4;
        background: rgba(233, 221, 255, 0.7);
        padding: 5px 14px;
        border-radius: 9999px;
        display: inline-block;
        margin-bottom: 12px;
        border: 1px solid rgba(208, 188, 255, 0.5);
    }
    .hero-headline {
        font-family: 'Newsreader', serif;
        font-size: 2.8rem;
        font-weight: 400;
        line-height: 1.15;
        letter-spacing: -0.02em;
        color: #18181B;
        margin-bottom: 0.5rem;
    }
    .hero-subtext {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        color: #494454;
        max-width: 720px;
        margin-bottom: 1.8rem;
        line-height: 1.6;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 20px 40px -15px rgba(107, 56, 212, 0.07), 0 1px 3px 0 rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }

    /* Chat Bubbles */
    .user-bubble {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(228, 217, 245, 0.8);
        border-radius: 20px 20px 20px 4px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.03);
    }
    .agent-bubble {
        background: linear-gradient(135deg, rgba(245, 240, 255, 0.85), rgba(253, 242, 248, 0.75));
        border: 1px solid rgba(208, 188, 255, 0.6);
        border-radius: 20px 20px 4px 20px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 6px 20px -4px rgba(107, 56, 212, 0.08);
    }

    /* Badges & Tags */
    .badge-tool {
        background: rgba(107, 56, 212, 0.1);
        color: #5516be;
        font-family: monospace;
        font-weight: 600;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(107, 56, 212, 0.2);
        display: inline-block;
        margin-right: 6px;
        margin-top: 6px;
    }
    .badge-cite {
        background: rgba(253, 86, 167, 0.12);
        color: #920050;
        font-weight: 600;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid rgba(253, 86, 167, 0.25);
        display: inline-block;
        margin-right: 6px;
        margin-top: 6px;
    }

    /* Sidebar Status Card */
    .telemetry-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(220, 217, 221, 0.6);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session-Scoped Agent (Preserves conversation state across reruns)
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = VoiceOpsAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_audio_ids" not in st.session_state:
    st.session_state.processed_audio_ids = set()

voice_agent = st.session_state.agent_instance

# Sidebar: Controls & Live Session Telemetrics
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
        <div style="width:36px; height:36px; border-radius:50%; background:linear-gradient(135deg, #6b38d4, #b4136d); display:flex; align-items:center; justify-center; box-shadow:0 4px 12px rgba(107,56,212,0.3);">
        </div>
        <div>
            <div style="font-weight:700; font-size:16px; color:#18181B;">ZentroMart VoiceOps</div>
            <div style="font-size:11px; color:#7b7486; text-transform:uppercase; letter-spacing:0.05em;">Acoustic Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎙️ Voice & Model Controls")
    voice_choice = st.selectbox("Gnani Speech Persona", ["Kaveri", "Trupti", "Pranav", "Shlok"], index=0)
    tts_model = st.selectbox("TTS Architecture", ["timbre-v2.5", "timbre-v3.0"], index=0)
    lang_choice = st.selectbox("Acoustic Language", ["en-IN (English / Hinglish)", "hi-IN (Hindi)"], index=0)
    stt_lang = "hi-IN" if "hi-IN" in lang_choice else "en-IN"

    st.markdown("---")
    st.subheader("📊 Live Telemetrics")
    ag = voice_agent.agent
    order_val = ag.active_order_id or "None Detected"
    verif_badge = "✅ Verified Caller" if ag.verified else ("⏳ Attempt 1/2" if ag.verification_attempts > 0 else "🔒 Gated (Unverified)")
    ticket_val = f"🎫 {ag.ticket_created}" if ag.ticket_created else "None"

    st.markdown(f"""
    <div class="telemetry-card">
        <div style="font-size:12px; color:#7b7486; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Active Session Order</div>
        <div style="font-size:16px; font-weight:700; color:#18181B; margin-bottom:10px;">#{order_val}</div>
        
        <div style="font-size:12px; color:#7b7486; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Identity Gate Status</div>
        <div style="font-size:13px; font-weight:700; color:{'#059669' if ag.verified else '#b4136d'}; margin-bottom:10px;">{verif_badge}</div>
        
        <div style="font-size:12px; color:#7b7486; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Escalation Dispatch</div>
        <div style="font-size:13px; font-weight:600; color:#494454;">{ticket_val}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="telemetry-card" style="font-size:12px; color:#494454; line-height:1.5;">
        <b>Engine:</b> Groq (gpt-oss-120b)<br>
        <b>Speech Stack:</b> Gnani STT & TTS<br>
        <b>Reference Date:</b> 20 Oct 2026
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Reset Dialog Session", use_container_width=True):
        voice_agent.reset()
        st.session_state.messages = []
        st.session_state.processed_audio_ids = set()
        st.rerun()

# Hero Header
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <span class="hero-eyebrow">Acoustic Intelligence • ZentroMart Customer Support</span>
    <h1 class="hero-headline">VoiceOps 2026</h1>
    <p class="hero-subtext" style="margin: 0 auto 24px auto;">
        Real-time conversational voice agent grounded in company documents, gated identity verification, and neural speech synthesis.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Two-Column Layout
col_studio, col_stream = st.columns([1, 1.2], gap="large")

with col_studio:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Customer Voice Input")
    st.markdown('<p style="font-size:13px; color:#7b7486; margin-bottom:12px;">Speak through your microphone to test Gnani STT transcription and multi-turn reasoning.</p>', unsafe_allow_html=True)
    
    # Audio recorder input
    audio_file = st.audio_input("Record audio clip")
    
    st.markdown("---")
    st.markdown("### 💬 Or Type Query")
    text_input = st.chat_input("Enter customer query (e.g. 'Order number 8978 2837' or 'How long do refunds take?')...")

    st.markdown('</div>', unsafe_allow_html=True)

    # 1. Process Voice Input
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_id = hash(audio_bytes)

        if audio_id not in st.session_state.processed_audio_ids:
            st.session_state.processed_audio_ids.add(audio_id)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name

            with st.spinner("🎧 Transcribing with Gnani STT and reasoning over policies..."):
                out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                turn_res = voice_agent.voice_turn(
                    audio_path=tmp_in_path,
                    output_wav_path=out_wav,
                    language_code=stt_lang,
                    voice=voice_choice,
                    model=tts_model,
                )

            transcript = turn_res["transcript"] or "[Unclear audio / silence]"
            st.session_state.messages.append({
                "role": "user",
                "content": transcript,
                "is_voice": True,
            })
            st.session_state.messages.append({
                "role": "agent",
                "content": turn_res["reply_text"],
                "audio_file": turn_res["audio_file"],
                "tools": turn_res["executed_tools"],
                "citations": turn_res["retrieved_sources"],
            })
            st.rerun()

    # 2. Process Text Input
    elif text_input:
        st.session_state.messages.append({
            "role": "user",
            "content": text_input,
            "is_voice": False,
        })

        with st.spinner("🤖 Reasoning and retrieving policies..."):
            reply_text, tools = voice_agent.agent.respond(text_input)

            # Synthesize speech
            spoken_text = normalize_text_for_speech(reply_text)
            if not spoken_text or not spoken_text.strip():
                spoken_text = "I am here to assist you. Could you please say that again?"

            out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            tts_lang_target = "hi-IN" if any(ch in reply_text for ch in "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") else "en-IN"

            try:
                voice_agent.tts.synthesize(
                    text=spoken_text,
                    voice=voice_choice,
                    model=tts_model,
                    language=tts_lang_target,
                    output_file=out_wav,
                )
            except Exception as e:
                print(f"[WARN] TTS error in app.py: {e}")
                out_wav = None

            citations = list(set(re.findall(r"\[([A-Z0-9_.-]+)\]", reply_text))) or ["KB-00.1"]

        st.session_state.messages.append({
            "role": "agent",
            "content": reply_text,
            "audio_file": out_wav,
            "tools": tools,
            "citations": citations,
        })
        st.rerun()

with col_stream:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Live Call Stream & Session Inspector")
    st.markdown('<p style="font-size:13px; color:#7b7486; margin-bottom:16px;">Real-time conversation stream, grounded citations, and tool footprints.</p>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color:#7b7486;">
            <span class="material-symbols-outlined" style="font-size: 40px; color:#cbc3d7;">record_voice_over</span>
            <div style="font-weight:600; margin-top:8px;">No messages yet</div>
            <div style="font-size:13px;">Record audio on the left or type a customer inquiry to start the conversation.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                voice_label = "🗣️ Voice (Gnani STT)" if msg.get("is_voice") else "💬 Text"
                st.markdown(f"""
                <div class="user-bubble">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:12px; font-weight:600; color:#494454;">
                        <span>Customer</span>
                        <span style="color:#7b7486;">{voice_label}</span>
                    </div>
                    <div style="font-size:15px; color:#18181B; line-height:1.5;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-bubble">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:12px; font-weight:600; color:#6b38d4;">
                        <span>Kaveri (ZentroMart VoiceOps AI)</span>
                        <span style="color:#b4136d;">Gnani Speech Synthesized</span>
                    </div>
                    <div style="font-size:15px; color:#18181B; line-height:1.5;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Audio playback widget
                if msg.get("audio_file") and os.path.exists(msg["audio_file"]):
                    st.audio(msg["audio_file"], format="audio/wav")

                # Footprint badges
                badges_html = ""
                if msg.get("tools"):
                    for t in msg["tools"]:
                        badges_html += f'<span class="badge-tool">🛠️ {t}</span>'
                if msg.get("citations"):
                    for c in msg["citations"]:
                        badges_html += f'<span class="badge-cite">📄 {c}</span>'
                if badges_html:
                    st.markdown(f'<div style="margin-top:-6px; margin-bottom:16px;">{badges_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
