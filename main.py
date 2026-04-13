"""Streamlit application for real-time AI workplace safety monitoring."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from alerts.notifier import AlertManager
from analytics.aggregator import SafetyAnalytics
from analytics.comparison import compare_periods
from core.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MULTI_CAMERA_COUNT,
    DEFAULT_STREAM_SECONDS,
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
    EVENTS_CSV,
    MACHINERY_DANGER_DISTANCE_PX,
    SUMMARY_CSV,
)
from core.detector import YoloDetector
from core.overlay import annotate_frame
from core.pipeline import SafetyPipeline
from storage.export import export_summary_to_csv
from storage.logger import EventLogger
from ui.dashboard import make_trend_figure, make_violation_figure
from video.sources import build_sources


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def inject_streamlit_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 8% 10%, #dff5ff 0%, transparent 28%),
                    radial-gradient(circle at 90% 0%, #ffe7cf 0%, transparent 24%),
                    linear-gradient(145deg, #f5fbff 0%, #eaf3fb 50%, #e5eef7 100%);
            }
            .hero-wrap {
                border: 1px solid rgba(255,255,255,0.55);
                background: linear-gradient(160deg, rgba(255,255,255,0.66), rgba(255,255,255,0.26));
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 14px 34px rgba(20, 48, 78, 0.12);
                margin-bottom: 0.8rem;
            }
            .hero-title {
                margin: 0;
                color: #15314c;
                font-weight: 800;
                letter-spacing: 0.02em;
            }
            .hero-sub {
                margin: 0.2rem 0 0;
                color: #355875;
                font-size: 0.92rem;
            }
            .mini-chip {
                display: inline-block;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                background: rgba(14, 156, 180, 0.16);
                border: 1px solid rgba(14, 156, 180, 0.32);
                color: #165f74;
                font-size: 0.72rem;
                font-weight: 700;
                margin-top: 0.45rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_streamlit_frontend(summary_history: list[dict]) -> None:
    latest = summary_history[-1] if summary_history else {}
    workers = int(latest.get("total_workers_detected", 0))
    violations = int(latest.get("total_violations", 0))
    compliance = float(latest.get("compliance_rate", 0.0)) * 100.0

    st.markdown(
        """
        <div class="hero-wrap">
            <h2 class="hero-title">AI Workplace Safety Monitoring</h2>
            <p class="hero-sub">Live detection, explainable risk scoring, and operational analytics in one Streamlit dashboard.</p>
            <span class="mini-chip">Streamlit Frontend Active</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Workers Seen", workers)
    c2.metric("Violations", violations)
    c3.metric("Compliance", f"{compliance:.2f}%")


def _to_checkup_row(result, item) -> dict:
    distance = item.nearest_machinery_distance
    is_danger = distance is not None and distance < MACHINERY_DANGER_DISTANCE_PX
    return {
        "timestamp": result.timestamp.isoformat(timespec="seconds"),
        "camera_id": result.camera_id,
        "person_id": item.person_id,
        "status": item.status,
        "helmet": "Yes" if item.has_helmet else "No",
        "vest": "Yes" if item.has_vest else "No",
        "machinery_distance_px": None if distance is None else round(float(distance), 2),
        "machinery_risk": "Danger" if is_danger else "OK",
        "score": item.score,
        "confidence": round(float(item.confidence_context), 3),
        "reason": item.reason,
    }


def default_image_folders() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "Safe Folder": root / "WorkSafety" / "safe",
        "Unsafe Folder": root / "WorkSafety" / "unsafe",
    }


def collect_images_from_folders(folder_paths: list[Path], limit_per_folder: int) -> list[tuple[str, Path]]:
    collected: list[tuple[str, Path]] = []
    for folder in folder_paths:
        if not folder.exists() or not folder.is_dir():
            continue
        images = sorted(
            [
                item
                for item in folder.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        for image_path in images[:limit_per_folder]:
            collected.append((folder.name, image_path))
    return collected


@st.cache_resource(show_spinner=False)
def build_pipeline(model_path: str, conf: float, iou: float) -> SafetyPipeline:
    detector = YoloDetector(model_path=model_path, conf_threshold=conf, iou_threshold=iou)
    detector.load()
    return SafetyPipeline(detector=detector)


def _save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.read())
    temp_file.flush()
    temp_file.close()
    return temp_file.name


@st.cache_data(show_spinner=False, ttl=2)
def _read_logs_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def run_monitoring(
    pipeline: SafetyPipeline,
    mode: str,
    video_path: str | None,
    camera_count: int,
    seconds: int,
    alert_manager: AlertManager,
):
    sources = build_sources(mode=mode, video_path=video_path, camera_count=camera_count)
    if not sources:
        st.error("No active sources available. Select a valid webcam or video input.")
        return None

    analytics = SafetyAnalytics()
    logger = EventLogger(EVENTS_CSV)
    checkup_rows: list[dict] = []

    frame_index = 0
    placeholders = [st.empty() for _ in sources]
    metrics_placeholder = st.empty()
    alerts_placeholder = st.empty()

    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        loop_start = time.perf_counter()
        alert_messages: list[str] = []

        for source_idx, source in enumerate(sources):
            video_frame = source.read(frame_index=frame_index)
            if video_frame is None:
                continue

            elapsed = max(1e-6, time.perf_counter() - loop_start)
            fps = 1.0 / elapsed
            result = pipeline.process_frame(
                frame_bgr=video_frame.frame_bgr,
                camera_id=video_frame.camera_id,
                frame_index=frame_index,
                fps=fps,
            )
            logger.log(result)
            analytics.update(result)
            for item in result.assessments:
                checkup_rows.append(_to_checkup_row(result, item))

            if any(item.severity.value == "red" for item in result.violations):
                alert_messages.append(
                    alert_manager.trigger(
                        f"HIGH RISK: {video_frame.camera_id} has active dangerous condition(s).",
                        high_risk=True,
                    )
                )

            annotated = annotate_frame(video_frame.frame_bgr, result)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            placeholders[source_idx].image(
                rgb,
                caption=f"{video_frame.camera_id} | Frame {frame_index} | FPS {fps:.2f}",
                use_container_width=True,
            )

        summary = analytics.live_metrics()
        metrics_placeholder.markdown(
            (
                f"**Workers:** {summary['total_workers_detected']} | "
                f"**Violations:** {summary['total_violations']} | "
                f"**Compliance:** {summary['compliance_rate'] * 100:.2f}%"
            )
        )

        if alert_messages:
            alerts_placeholder.error("\n".join(alert_messages))
        else:
            alerts_placeholder.info("No active high-risk alerts.")

        frame_index += 1

    for source in sources:
        source.release()

    return {
        **analytics.summary(),
        "checkup_rows": checkup_rows[-1200:],
    }


def run_image_monitoring(
    pipeline: SafetyPipeline,
    image_items: list[tuple[str, Path]],
    alert_manager: AlertManager,
):
    if not image_items:
        st.error("No images found. Add files to WorkSafety/safe or WorkSafety/unsafe.")
        return None

    analytics = SafetyAnalytics()
    logger = EventLogger(EVENTS_CSV)
    checkup_rows: list[dict] = []

    image_placeholder = st.empty()
    metrics_placeholder = st.empty()
    alerts_placeholder = st.empty()

    for frame_index, (group_name, image_path) in enumerate(image_items, start=1):
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            continue

        result = pipeline.process_frame(
            frame_bgr=frame_bgr,
            camera_id=f"{group_name}-images",
            frame_index=frame_index,
            fps=0.0,
        )
        logger.log(result)
        analytics.update(result)
        for item in result.assessments:
            checkup_rows.append(_to_checkup_row(result, item))

        red_events = [item for item in result.violations if item.severity.value == "red"]
        if red_events:
            alerts_placeholder.error(
                alert_manager.trigger(
                    f"HIGH RISK: {image_path.name} has dangerous condition(s).",
                    high_risk=True,
                )
            )
        else:
            alerts_placeholder.info("No active high-risk alerts.")

        annotated = annotate_frame(frame_bgr, result)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image_placeholder.image(
            rgb,
            caption=f"{group_name} | {image_path.name} | Item {frame_index}/{len(image_items)}",
            use_container_width=True,
        )

        summary = analytics.live_metrics()
        metrics_placeholder.markdown(
            (
                f"**Workers:** {summary['total_workers_detected']} | "
                f"**Violations:** {summary['total_violations']} | "
                f"**Compliance:** {summary['compliance_rate'] * 100:.2f}%"
            )
        )

    return {
        **analytics.summary(),
        "checkup_rows": checkup_rows[-1200:],
    }


def render_analytics(summary: dict) -> None:
    st.subheader("Analytics Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Workers", summary.get("total_workers_detected", 0))
    col2.metric("Total Violations", summary.get("total_violations", 0))
    col3.metric("Compliance Rate", f"{summary.get('compliance_rate', 0.0) * 100:.2f}%")

    fig_dist = make_violation_figure(summary.get("violation_distribution", {}))
    st.pyplot(fig_dist, use_container_width=True)

    fig_score = make_trend_figure(summary.get("score_trend", []), "Safety Score Trend", "Average Safety Score")
    st.pyplot(fig_score, use_container_width=True)

    fig_compliance = make_trend_figure(summary.get("compliance_trend", []), "Compliance Trend", "Compliance Rate")
    st.pyplot(fig_compliance, use_container_width=True)

    st.subheader("Helmet, Vest, and Machinery Checkups")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Helmet Compliance", f"{summary.get('helmet_compliance_rate', 0.0) * 100:.2f}%")
    k2.metric("Vest Compliance", f"{summary.get('vest_compliance_rate', 0.0) * 100:.2f}%")
    k3.metric("Machinery Exposure", int(summary.get("machinery_exposure_count", 0)))
    k4.metric("Machinery Danger", int(summary.get("machinery_danger_count", 0)))


def render_checkup_table(checkup_rows: list[dict]) -> None:
    st.subheader("Detailed PPE and Machinery Checkups")
    if not checkup_rows:
        st.info("No checkup data available yet. Start monitoring to generate helmet, vest, and machinery checks.")
        return

    rows = pd.DataFrame(checkup_rows)
    col1, col2, col3 = st.columns([1.2, 1.2, 2.0])
    status_filter = col1.selectbox("Status Filter", ["All", "Safe", "Warning", "Unsafe", "Dangerous"], key="check_status_filter")
    danger_only = col2.checkbox("Show Machinery Danger Only", value=False, key="check_danger_only")
    search = col3.text_input("Search Person/Reason", value="", key="check_search")

    filtered = rows.copy()
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]
    if danger_only:
        filtered = filtered[filtered["machinery_risk"] == "Danger"]
    if search.strip():
        needle = search.strip().lower()
        filtered = filtered[
            filtered["person_id"].str.lower().str.contains(needle, na=False)
            | filtered["reason"].str.lower().str.contains(needle, na=False)
        ]

    st.dataframe(filtered.sort_values(by=["timestamp", "camera_id", "person_id"], ascending=[False, True, True]), use_container_width=True)


def render_before_after(snapshot_history: list[dict]) -> None:
    st.subheader("Before vs After Analysis")
    if len(snapshot_history) < 2:
        st.info("Run at least two monitoring sessions to compare improvements.")
        return

    before = snapshot_history[-2]
    after = snapshot_history[-1]
    comparison = compare_periods(before=before, after=after)

    col1, col2 = st.columns(2)
    col1.metric("Violation Improvement", comparison["violation_improvement"])
    col2.metric("Compliance Delta", f"{comparison['compliance_delta'] * 100:.2f}%")

    st.json(comparison)


def render_logs_table() -> None:
    st.subheader("Violation Logs")
    csv_path = Path(EVENTS_CSV)
    if not csv_path.exists():
        st.info("No violation events logged yet.")
        return

    logs = _read_logs_csv(str(csv_path))
    def _status_style(value: object) -> str:
        text = str(value).strip().lower()
        if text in {"unsafe", "dangerous", "danger", "red"}:
            return "color: #ff3b30; font-weight: 700"
        if text in {"warning", "yellow", "amber"}:
            return "color: #ffb020; font-weight: 600"
        if text in {"safe", "green"}:
            return "color: #2ecc71; font-weight: 600"
        return ""

    styled = logs.style
    if "status" in logs.columns:
        styled = styled.map(_status_style, subset=["status"])
    if "severity" in logs.columns:
        styled = styled.map(_status_style, subset=["severity"])

    st.dataframe(styled, use_container_width=True)


def app() -> None:
    st.set_page_config(page_title="AI Workplace Safety Monitoring", layout="wide")
    inject_streamlit_theme()

    if "summary_history" not in st.session_state:
        st.session_state["summary_history"] = []
    if "last_checkup_rows" not in st.session_state:
        st.session_state["last_checkup_rows"] = []

    render_streamlit_frontend(st.session_state["summary_history"])

    st.sidebar.header("Monitoring Controls")
    mode_label = st.sidebar.selectbox(
        "Input Mode",
        ["Webcam", "Video File", "Multi-Camera Simulation", "Image Folders (safe/unsafe)"],
    )
    mode = {
        "Webcam": "webcam",
        "Video File": "video",
        "Multi-Camera Simulation": "multi",
        "Image Folders (safe/unsafe)": "images",
    }[mode_label]

    uploaded = None
    video_path = None
    if mode in {"video", "multi"}:
        uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            video_path = _save_uploaded_video(uploaded)

    image_items: list[tuple[str, Path]] = []
    if mode == "images":
        defaults = default_image_folders()
        available_labels = [label for label, path in defaults.items() if path.exists()]
        selected_labels = st.sidebar.multiselect(
            "Dataset Folders",
            options=list(defaults.keys()),
            default=available_labels,
        )
        per_folder_limit = st.sidebar.slider("Images Per Folder", min_value=1, max_value=500, value=100, step=1)

        selected_paths = [defaults[label] for label in selected_labels]
        image_items = collect_images_from_folders(selected_paths, limit_per_folder=per_folder_limit)
        st.sidebar.caption(f"Images ready: {len(image_items)}")

    model_path = st.sidebar.text_input("YOLO Model Path", value=DEFAULT_MODEL_PATH)
    alert_profile = st.sidebar.selectbox(
        "Monitoring Profile",
        ["Balanced", "Strict PPE", "High Machinery Risk"],
    )
    conf = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=float(DETECTION_CONFIDENCE), step=0.05)
    iou = st.sidebar.slider("IoU Threshold", min_value=0.1, max_value=0.9, value=float(DETECTION_IOU), step=0.05)
    if alert_profile == "Strict PPE":
        conf = max(conf, 0.60)
    if alert_profile == "High Machinery Risk":
        iou = max(iou, 0.50)
    st.sidebar.caption(f"Effective thresholds -> confidence: {conf:.2f}, IoU: {iou:.2f}")
    seconds = DEFAULT_STREAM_SECONDS
    if mode != "images":
        seconds = st.sidebar.slider("Session Duration (seconds)", min_value=5, max_value=120, value=DEFAULT_STREAM_SECONDS, step=5)

    camera_count = DEFAULT_MULTI_CAMERA_COUNT
    if mode == "multi":
        camera_count = st.sidebar.slider("Simulated Camera Count", min_value=2, max_value=6, value=DEFAULT_MULTI_CAMERA_COUNT, step=1)
    sound_enabled = st.sidebar.checkbox("Enable Sound Alerts", value=False)

    start_clicked = st.sidebar.button("Start Monitoring")

    if start_clicked:
        with st.spinner("Loading model and processing live frames..."):
            pipeline = build_pipeline(model_path=model_path, conf=conf, iou=iou)
            detector = pipeline.detector
            available_classes_fn = getattr(detector, "available_classes", None)
            if callable(available_classes_fn):
                available_classes = available_classes_fn()
            else:
                raw_names = getattr(detector, "_names", {}) or {}
                normalize_fn = getattr(detector, "_normalize_label", lambda x: str(x).strip().lower())
                available_classes = sorted({normalize_fn(str(name)) for name in raw_names.values()})
            missing_checks = []
            if not detector.supports_any_class({"helmet", "hardhat", "hard_hat"}):
                missing_checks.append("helmet")
            if not detector.supports_any_class({"vest", "safety_vest", "safety-vest"}):
                missing_checks.append("vest")
            if missing_checks:
                st.warning(
                    "Current model does not include PPE classes for: "
                    + ", ".join(missing_checks)
                    + ". PPE checks for those classes are skipped to avoid false violations."
                )
                if available_classes:
                    display_classes = ", ".join(available_classes[:20])
                    suffix = " ..." if len(available_classes) > 20 else ""
                    st.caption(f"Model classes detected: {display_classes}{suffix}")
                if set(missing_checks) == {"helmet", "vest"}:
                    st.info(
                        "This usually means the selected model is a generic detector (for example COCO). "
                        "Use a PPE-trained YOLO model (best.pt) that includes helmet and vest classes."
                    )
            alert_manager = AlertManager(sound_enabled=sound_enabled)
            if mode == "images":
                summary = run_image_monitoring(
                    pipeline=pipeline,
                    image_items=image_items,
                    alert_manager=alert_manager,
                )
            else:
                summary = run_monitoring(
                    pipeline=pipeline,
                    mode=mode,
                    video_path=video_path,
                    camera_count=camera_count,
                    seconds=seconds,
                    alert_manager=alert_manager,
                )

        if summary is not None:
            st.session_state["summary_history"].append(summary)
            st.session_state["last_checkup_rows"] = summary.get("checkup_rows", [])
            render_analytics(summary)

            export_path = export_summary_to_csv(summary, SUMMARY_CSV)
            csv_bytes = Path(export_path).read_bytes()
            st.download_button(
                label="Download Analytics CSV",
                data=csv_bytes,
                file_name="analytics_summary.csv",
                mime="text/csv",
            )

            if st.session_state["last_checkup_rows"]:
                checkup_df = pd.DataFrame(st.session_state["last_checkup_rows"])
                st.download_button(
                    label="Download Detailed Checkups CSV",
                    data=checkup_df.to_csv(index=False).encode("utf-8"),
                    file_name="detailed_checkups.csv",
                    mime="text/csv",
                )

    st.divider()
    if st.session_state["summary_history"]:
        render_before_after(st.session_state["summary_history"])

    st.divider()
    render_logs_table()

    st.divider()
    render_checkup_table(st.session_state["last_checkup_rows"])

    st.markdown("### Deployment")
    st.code("streamlit run main.py", language="bash")


if __name__ == "__main__":
    app()

