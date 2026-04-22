"""
app.py — Visual Book QA
Simple layout: left controls, right slide image output.
"""

import gradio as gr
from pathlib import Path

from pipeline.ingestion import PDFIngester
from pipeline.retrieval import HybridRetriever
from pipeline.generation import AnswerGenerator
from pipeline.renderer import HTMLRenderer

ingester  = PDFIngester()
retriever = HybridRetriever()
generator = AnswerGenerator()
renderer  = HTMLRenderer()

_loaded_pdf: str | None = None


def load_pdf(pdf_file) -> str:
    global _loaded_pdf
    if pdf_file is None:
        return '<div class="status-idle">No file loaded</div>'
    path = pdf_file if isinstance(pdf_file, str) else (
        pdf_file.get("name") or pdf_file.get("path") if isinstance(pdf_file, dict)
        else getattr(pdf_file, "name", str(pdf_file))
    )
    try:
        chunks, idx, bm25 = ingester.ingest(path)
        retriever.load(chunks, idx, bm25, ingester.embed_model)
        _loaded_pdf = path
        name = Path(path).name
        return f'<div class="status-ok">✅ &nbsp;<strong>{name}</strong> &nbsp;·&nbsp; {len(chunks)} chunks indexed</div>'
    except Exception as e:
        return f'<div class="status-err">❌ {e}</div>'


def answer_question(question: str, _state: str):
    if not _loaded_pdf:
        return None, '<div class="status-err">⚠ Please upload a PDF first.</div>', ""
    if not question.strip():
        return None, '<div class="status-err">⚠ Please enter a question.</div>', ""
    try:
        chunks = retriever.retrieve(question)
        if not chunks:
            return None, '<div class="status-err">No relevant content found.</div>', ""
        html = generator.generate(question, chunks)
        img_path = renderer.render(html)
        return str(img_path), '<div class="status-ok">✅ Slide generated.</div>', html
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f'<div class="status-err">❌ {e}</div>', ""


def regenerate(question: str, _state: str):
    return answer_question(question, _state)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .gradio-container, .main { background: #0f1623 !important; }
footer { display: none !important; }
.gradio-container { max-width: 100% !important; padding: 0; margin: 0 !important; }

/* ── Header ── */
#app-header {
    padding: 22px 32px 18px;
    border-bottom: 1px solid #1c2e44;
    margin-bottom: 0;
    display: flex; align-items: center; gap: 12px;
}
#app-header h1 {
    font-family: 'Inter', sans-serif; font-weight: 700;
    font-size: 1.25rem; color: #f0f6ff; margin: 0;
}
#app-header p {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    color: #3a5a7a; margin: 3px 0 0;
}
#app-header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #3ee8b5; flex-shrink: 0;
    box-shadow: 0 0 8px #3ee8b5;
}

/* ── Panels ── */
#left-panel  { padding: 24px 14px 24px 28px; }
#right-panel { padding: 24px 28px 24px 14px; }

/* ── Labels ── */
.lbl {
    font-family: 'Space Mono', monospace; font-size: 0.62rem;
    font-weight: 700; letter-spacing: 0.12em; color: #2a4a6a;
    text-transform: uppercase; margin-bottom: 8px;
}

/* ── Upload area ── */
#upload-box {
    border: 1.5px dashed #1e4060 !important;
    border-radius: 10px !important;
    background: #0c1825 !important;
    transition: border-color 0.2s ease !important;
}
#upload-box:hover { border-color: #3ee8b5 !important; }
#upload-box label { color: #3a5a7a !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; }

/* ── Status ── */
.status-ok  { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #3ee8b5; padding: 6px 0; }
.status-err { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #f87171; padding: 6px 0; }
.status-idle{ font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #2a4a6a; padding: 6px 0; }

/* ── Question box ── */
textarea {
    background: #0c1825 !important;
    color: #d8eaf8 !important;
    border: 1.5px solid #1a3050 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 12px 14px !important;
    line-height: 1.55 !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}
textarea::placeholder { color: #2a4a6a !important; }
textarea:focus {
    border-color: #3ee8b5 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(62,232,181,0.07) !important;
}
label { color: #3a5a7a !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; }

/* ── Buttons ── */
#ask-btn button {
    width: 100% !important; padding: 12px !important;
    background: #3ee8b5 !important; color: #031018 !important;
    border: none !important; border-radius: 9px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
    margin-top: 2px !important;
}
#ask-btn button:hover  { background: #62f0c6 !important; transform: translateY(-1px) !important; }
#ask-btn button:active { transform: translateY(0px) !important; }

#regen-btn button {
    width: 100% !important; padding: 11px !important;
    background: transparent !important; color: #3ee8b5 !important;
    border: 1.5px solid #1a4030 !important; border-radius: 9px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important; cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s !important;
    margin-top: 6px !important;
}
#regen-btn button:hover {
    border-color: #3ee8b5 !important;
    background: rgba(62,232,181,0.05) !important;
}

/* ── Status bar ── */
#status-bar { min-height: 28px; padding: 2px 0; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1c2e44; margin: 18px 0; }

/* ── Hint box ── */
.hint-box {
    background: #0c1825; border: 1px solid #1c2e44;
    border-radius: 8px; padding: 12px 14px; margin-top: 14px;
}
.hint-box p {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    color: #2a4a6a; margin: 0; line-height: 1.7;
}
.hint-box p strong { color: #3a6a5a; }

/* ── Image output ── */
#image-out { background: transparent !important; border: none !important; }
#image-out label { display: none !important; }
#image-out img {
    border-radius: 14px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5) !important;
    width: 100% !important;
}
"""


def build_ui():
    with gr.Blocks(title="Visual Book QA") as demo:

        gr.HTML(f"<style>{CSS}</style>")

        # Header
        gr.HTML("""
        <div id="app-header">
            <div class="dot"></div>
            <div>
                <h1>Visual Book QA</h1>
                <p>Upload a PDF · Ask a question · Get a visual slide</p>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── LEFT: controls ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300, elem_id="left-panel"):

                gr.HTML('<div class="lbl">1 · Upload PDF</div>')
                pdf_input = gr.File(
                    label="Drop your PDF here",
                    file_types=[".pdf"],
                    file_count="single",
                    type="filepath",
                    elem_id="upload-box",
                )
                pdf_status = gr.HTML('<div class="status-idle">No file loaded</div>')

                gr.HTML('<hr class="divider"><div class="lbl">2 · Ask a Question</div>')
                question_input = gr.Textbox(
                    label="",
                    lines=4,
                    max_lines=6,
                    placeholder="e.g. What is the attention mechanism?\nHow does backpropagation work?",
                    elem_id="question-box",
                )

                ask_btn   = gr.Button("⚡  Generate Slide",    elem_id="ask-btn",   variant="primary")
                regen_btn = gr.Button("↺  Regenerate",         elem_id="regen-btn")
                status_bar = gr.HTML('<div id="status-bar"></div>')

                gr.HTML("""
                <div class="hint-box">
                  <p>
                    <strong>Tips:</strong><br>
                    · Ask about any concept in the book<br>
                    · The slide adapts its layout to the topic<br>
                    · Hit Regenerate if the diagram needs improvement
                  </p>
                </div>
                """)

            # ── RIGHT: slide output ───────────────────────────────────────
            with gr.Column(scale=2, elem_id="right-panel"):

                gr.HTML('<div class="lbl">Visual Slide Output</div>')
                image_out = gr.Image(
                    label="",
                    type="filepath",
                    elem_id="image-out",
                    height=600,
                )

        raw_state = gr.State("")

        pdf_input.change(fn=load_pdf, inputs=[pdf_input], outputs=[pdf_status])

        ask_btn.click(
            fn=answer_question,
            inputs=[question_input, raw_state],
            outputs=[image_out, status_bar, raw_state],
        )
        question_input.submit(
            fn=answer_question,
            inputs=[question_input, raw_state],
            outputs=[image_out, status_bar, raw_state],
        )
        regen_btn.click(
            fn=regenerate,
            inputs=[question_input, raw_state],
            outputs=[image_out, status_bar, raw_state],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)
