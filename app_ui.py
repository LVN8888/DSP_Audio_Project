from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from experiments.runner import predict_with_classical


st.set_page_config(page_title="DSP Audio Predictor", page_icon="🎧", layout="wide")


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


@st.cache_data(show_spinner=False)
def run_prediction(
    file_path: str,
    pipeline: str,
    artifact_path: str,
    sr: int,
    filter_type: str,
    duration: float,
    segment_duration: float,
    hop_duration: float,
    threshold: float,
) -> Dict[str, Any]:
    return predict_with_classical(
        file_path=file_path,
        pipeline=pipeline,
        artifact_path=artifact_path,
        sr=sr,
        filter_type=filter_type,
        duration=duration,
        segment_duration=segment_duration,
        hop_duration=hop_duration,
        threshold=threshold,
    )



def top3_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    top3 = result.get("top3", [])
    rows = []
    for rank, item in enumerate(top3, start=1):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label, prob = item[0], item[1]
        else:
            label, prob = str(item), None
        rows.append(
            {
                "Rank": rank,
                "Label": label,
                "Probability": round(float(prob), 4) if prob is not None else None,
            }
        )
    return pd.DataFrame(rows)



def render_result(title: str, result: Optional[Dict[str, Any]], error: Optional[str] = None) -> None:
    st.subheader(title)

    if error:
        st.error(error)
        return

    if not result:
        st.info("Chưa có kết quả.")
        return

    prediction = result.get("prediction", "N/A")
    confidence = result.get("confidence", 0.0)

    c1, c2 = st.columns(2)
    c1.metric("Prediction", str(prediction))
    c2.metric("Confidence", f"{float(confidence):.4f}")

    df = top3_dataframe(result)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Raw result object"):
        st.json(result)


st.title("🎧 DSP Audio Classification UI")
st.caption("Upload 1 file âm thanh và chạy so sánh RAW vs DSP từ artifact đã train.")

with st.sidebar:
    st.header("Cấu hình")
    raw_artifact = st.text_input(
        "RAW artifact path",
        value="outputs/artifacts/raw_svm.joblib",
    )
    dsp_artifact = st.text_input(
        "DSP artifact path",
        value="outputs/artifacts/dsp_svm.joblib",
    )

    sr = st.number_input("Sample rate (sr)", min_value=8000, max_value=96000, value=22050, step=50)
    duration = st.number_input("Duration", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
    filter_type = st.selectbox("Filter type", options=["iir", "fir"], index=0)
    segment_duration = st.number_input("Segment duration", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
    hop_duration = st.number_input("Hop duration", min_value=0.1, max_value=20.0, value=2.0, step=0.5)
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

uploaded = st.file_uploader(
    "Chọn file âm thanh",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    accept_multiple_files=False,
)

if uploaded is not None:
    st.audio(uploaded)
    st.write(f"**File:** {uploaded.name}")

    temp_path = save_uploaded_file(uploaded)

    c1, c2, c3 = st.columns([1, 1, 2])
    run_both = c1.button("▶️ Chạy cả RAW + DSP", use_container_width=True)
    run_raw_only = c2.button("RAW only", use_container_width=True)
    run_dsp_only = c3.button("DSP only", use_container_width=True)

    raw_result = None
    dsp_result = None
    raw_error = None
    dsp_error = None

    if run_both or run_raw_only or run_dsp_only:
        if (run_both or run_raw_only) and not Path(raw_artifact).exists():
            raw_error = f"Không tìm thấy RAW artifact: {raw_artifact}"
        if (run_both or run_dsp_only) and not Path(dsp_artifact).exists():
            dsp_error = f"Không tìm thấy DSP artifact: {dsp_artifact}"

        with st.spinner("Đang chạy dự đoán..."):
            if (run_both or run_raw_only) and raw_error is None:
                try:
                    raw_result = run_prediction(
                        file_path=str(temp_path),
                        pipeline="raw",
                        artifact_path=raw_artifact,
                        sr=int(sr),
                        filter_type=filter_type,
                        duration=float(duration),
                        segment_duration=float(segment_duration),
                        hop_duration=float(hop_duration),
                        threshold=float(threshold),
                    )
                except Exception as exc:
                    raw_error = f"RAW failed: {exc}"

            if (run_both or run_dsp_only) and dsp_error is None:
                try:
                    dsp_result = run_prediction(
                        file_path=str(temp_path),
                        pipeline="dsp",
                        artifact_path=dsp_artifact,
                        sr=int(sr),
                        filter_type=filter_type,
                        duration=float(duration),
                        segment_duration=float(segment_duration),
                        hop_duration=float(hop_duration),
                        threshold=float(threshold),
                    )
                except Exception as exc:
                    dsp_error = f"DSP failed: {exc}"

        if run_both:
            left, right = st.columns(2)
            with left:
                render_result("RAW result", raw_result, raw_error)
            with right:
                render_result("DSP result", dsp_result, dsp_error)

            if raw_result and dsp_result:
                st.markdown("### So sánh nhanh")
                compare_df = pd.DataFrame(
                    [
                        {
                            "Pipeline": "RAW",
                            "Prediction": raw_result.get("prediction", "N/A"),
                            "Confidence": round(float(raw_result.get("confidence", 0.0)), 4),
                        },
                        {
                            "Pipeline": "DSP",
                            "Prediction": dsp_result.get("prediction", "N/A"),
                            "Confidence": round(float(dsp_result.get("confidence", 0.0)), 4),
                        },
                    ]
                )
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

        elif run_raw_only:
            render_result("RAW result", raw_result, raw_error)
        elif run_dsp_only:
            render_result("DSP result", dsp_result, dsp_error)
else:
    st.info("Hãy upload file âm thanh để bắt đầu.")


