from __future__ import annotations

import base64
import hashlib
import html
import io
import os
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import streamlit as st
import pypdfium2 as pdfium
from dotenv import load_dotenv
from openai import OpenAI

from analyzer import (
    SUBJECTS,
    Paper,
    Question,
    available_chapters,
    build_study_plan,
    generate_practice_paper,
    load_papers,
    rank_patterns,
    split_question_options,
)


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="KCET Paper Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#182526; --muted:#647875; --green:#176b54; --mint:#e8f4ed; --cream:#f6f8f2; }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      .stApp { background:linear-gradient(180deg,#f2f7f1 0%,#fbfcf9 34rem,#fbfcf9 100%); }
      h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
      .hero { border-radius:24px; padding:2.1rem 2.2rem; color:#fff;
        background:radial-gradient(circle at 88% 8%,#70b790 0,transparent 27%),
        linear-gradient(120deg,#143e38,#176b54 64%,#398367); margin-bottom:1.25rem; }
      .hero h1 { color:#fff; font-size:2.35rem; margin:0 0 .45rem; }
      .hero p { color:#e2f3e7; max-width:760px; margin:0; font-size:1.03rem; }
      .eyebrow { color:#d1edda; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; }
      .metric { background:#fff; border:1px solid #e6ece5; border-radius:18px; padding:1.1rem 1.2rem; min-height:112px; box-shadow:0 5px 18px #183d2710; }
      .metric-label { color:var(--muted); font-size:.84rem; }
      .metric-value { color:var(--green); font:800 1.8rem 'Manrope',sans-serif; margin-top:.3rem; }
      .section-note { color:var(--muted); margin-top:-.45rem; margin-bottom:1.1rem; }
      .priority-high { color:#176b54; font-weight:700; }
      .priority-medium { color:#a36717; font-weight:700; }
      .priority-low { color:#63706d; font-weight:700; }
      div[data-testid="stExpander"] { border:1px solid #e6ece5; border-radius:14px; background:#fff; }
      .question-stem { line-height:1.75; font-size:1rem; margin:.5rem 0 .85rem; white-space:pre-wrap; }
      .question-option { border:1px solid #e8eee8; border-radius:10px; padding:.55rem .75rem;
        margin:.42rem 0; background:#f8faf7; line-height:1.65; }
      .question-option strong { color:#176b54; margin-right:.35rem; }
      .source-note { color:#536862; font-size:.9rem; }
      .stButton button[kind="primary"] { border-radius:10px; }
      [data-testid="stSidebar"] { background:#f4f7f2; }
      a { color:#176b54 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading the papers in your subject folders…")
def get_analysis(folder: str) -> tuple[list[Paper], list[str]]:
    return load_papers(Path(folder))


@st.cache_data
def get_source_pdf_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data
def get_source_page_images(path: str, first_page: int, last_page: int) -> tuple[bytes, ...]:
    page_images: list[bytes] = []
    document = pdfium.PdfDocument(path)
    try:
        for page_number in range(first_page, last_page + 1):
            page = document[page_number - 1]
            try:
                bitmap = page.render(scale=1.8)
                image = bitmap.to_pil().convert("RGB")
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=90, optimize=True)
                page_images.append(output.getvalue())
            finally:
                page.close()
    finally:
        document.close()
    return tuple(page_images)


def explain_question(
    question: Question,
    api_key: str,
    page_images: tuple[bytes, ...],
    solution_images: tuple[bytes, ...],
    model: str,
) -> str:
    client = OpenAI(api_key=api_key)
    answer_note = f"The source paper's marked answer is {question.answer}." if question.answer else (
        "No marked answer was reliably extracted. Solve independently from the question if it contains enough information, "
        "label the result as your derivation rather than an official key, and explain any ambiguity instead of inventing missing text."
    )
    content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Subject: {question.subject}\nChapter estimate: {question.chapter}\n"
            f"Text extracted from the paper (may have lost symbols): {question.text}\n{answer_note}\n\n"
            "The attached original question-page image(s) are authoritative for the exact wording, symbols, diagrams, tables, "
            "and answer choices. First transcribe the visible question and options accurately, retaining original "
            "chemical formulas, subscripts, superscripts, Greek letters, operators, units, reaction arrows, and labels. "
            "Do not silently repair ambiguous source printing: identify uncertainty explicitly. Then give a clear, "
            "detailed KCET-level solution with numbered steps, readable LaTeX for equations, the final answer, and "
            "a short exam tip. Treat an extracted marked answer as source-provided; if no marked answer is supplied, "
            "solve independently and do not claim your result is an official key. If a printed solution-page image "
            "is attached after the question images, compare it with your derivation and clearly identify it as the "
            "source's printed solution."
        ),
    }]
    image_groups = [
        ("Original question page", page_images),
        ("Original printed solution page", solution_images),
    ]
    for label, images in image_groups:
        if images:
            if label == "Original printed solution page":
                content.append({
                    "type": "text",
                    "text": (
                        "The following image(s) are the original paper's printed solution for this question, "
                        "if legible. Preserve its notation and use it to check the derivation."
                    ),
                })
            for image in images:
                encoded = base64.b64encode(image).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                })
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise visual-reading KCET tutor. Read the original scanned or printed page before "
                    "trusting extracted text. Preserve notation faithfully and provide a self-contained, accessible "
                    "worked explanation. Use Markdown headings and display math where useful. Never invent missing "
                    "question content, diagram labels, or an official answer."
                ),
            },
            {"role": "user", "content": content},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("The AI service returned an empty explanation.")
    return content


def priority_class(value: str) -> str:
    return {
        "High": "priority-high",
        "Medium": "priority-medium",
        "Build foundations": "priority-low",
    }.get(value, "")


def chapter_resources(subject: str, chapter: str) -> list[tuple[str, str]]:
    search = quote_plus(f"KCET {subject} {chapter} lesson")
    return [
        ("NCERT textbooks (official)", "https://ncert.nic.in/textbook.php"),
        ("KEA CET information (official)", "https://cetonline.karnataka.gov.in/kea/"),
        (f"Find a {chapter} lesson on YouTube", f"https://www.youtube.com/results?search_query={search}"),
    ]


papers, notices = get_analysis(str(ROOT))
usable_papers = [paper for paper in papers if paper.questions]
all_questions = [question for paper in papers for question in paper.questions]
patterns = rank_patterns(papers)
paper_paths = {(paper.subject, paper.path.name): paper.path for paper in papers}
vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your KCET preparation companion</div>
      <h1>Study smarter, one question at a time.</h1>
      <p>Explore previous papers by subject and chapter, spot recurring question patterns,
      build a focused study plan, and request a worked solution whenever you need one.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Your study plan")
    weeks = st.slider("Weeks until your exam", min_value=2, max_value=16, value=8)
    hours_per_week = st.slider("Study hours per week", min_value=3, max_value=35, value=12)
    st.divider()
    st.markdown("### Optional AI tutor")
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Only sent with the selected question and its original PDF page when you request a vision explanation.",
    )
    st.caption(
        f"Vision model: `{vision_model}`. Your key is used only for an explanation request; "
        "the selected question and relevant question/solution page images are sent to OpenAI. "
        "API usage may incur charges."
    )
    st.divider()
    st.caption("Question indexing is local. Original diagrams and notation remain visible in the source-page preview.")
    if st.button("Refresh paper analysis", use_container_width=True):
        get_analysis.clear()
        st.rerun()

question_count = len(all_questions)
subjects_with_papers = len({paper.subject for paper in usable_papers})
high_count = sum(row["priority"] == "High" for row in patterns)
metrics = st.columns(4)
for column, label, value in zip(
    metrics,
    ("Questions extracted", "Usable past papers", "Subjects covered", "High-repeat patterns"),
    (f"{question_count:,}", len(usable_papers), f"{subjects_with_papers} / 3", high_count),
):
    column.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
overview_tab, bank_tab, prediction_tab, plan_tab, resources_tab = st.tabs(
    ["✨ Pattern insights", "📖 Question bank", "🎯 Next-year practice papers", "🗓️ Study plan", "🔗 Learning resources"]
)

with overview_tab:
    st.subheader("Recurring patterns, ranked by year-to-year consistency")
    st.markdown(
        '<p class="section-note">Rank reflects how many analyzed paper years include this chapter/topic pattern; it is a study signal, not a prediction.</p>',
        unsafe_allow_html=True,
    )
    selected_subjects = st.multiselect(
        "Show subjects",
        SUBJECTS,
        default=list(SUBJECTS),
        key="pattern_subjects",
    )
    if not selected_subjects:
        st.info("Choose at least one subject to see its recurring patterns.")
    for subject in selected_subjects:
        subject_patterns = [row for row in patterns if row["subject"] == subject]
        with st.expander(f"{subject} · {len(subject_patterns)} detected patterns", expanded=True):
            if not subject_patterns:
                st.info("No question patterns were extracted for this subject.")
                continue
            for rank, row in enumerate(subject_patterns[:15], start=1):
                year_text = ", ".join(map(str, row["years"]))
                st.markdown(
                    f"**#{rank} · {row['chapter']} — {row['topic']}**  \n"
                    f"<span class='{priority_class(row['priority'])}'>{row['priority']} priority</span>"
                    f" · seen in {len(row['years'])} of {sum(1 for p in usable_papers if p.subject == subject)} "
                    f"usable paper-years · {row['count']} question(s) · {year_text}",
                    unsafe_allow_html=True,
                )
                if row["examples"]:
                    with st.expander("Example questions from this pattern"):
                        for sample in row["examples"]:
                            st.caption(f"{sample.year}, Q{sample.number}: {sample.text[:360]}{'…' if len(sample.text) > 360 else ''}")

with bank_tab:
    st.subheader("Browse every extracted question by chapter")
    subject = st.selectbox("Choose a subject", SUBJECTS, key="bank_subject")
    subject_questions = [question for question in all_questions if question.subject == subject]
    chapter_rows = available_chapters(papers, subject)
    chapter_rank = {row["chapter"]: row for row in chapter_rows}
    questions_by_chapter: dict[str, list[Question]] = defaultdict(list)
    for question in subject_questions:
        questions_by_chapter[question.chapter].append(question)
    search_text = st.text_input("Search questions", placeholder="Try: resistor, probability, half-life…")
    if not subject_questions:
        st.info("No questions were extracted. Check the notes below; some PDFs are answer keys or image-only papers.")
    for chapter, questions in sorted(
        questions_by_chapter.items(),
        key=lambda item: (
            chapter_rank.get(item[0], {}).get("coverage", 0),
            chapter_rank.get(item[0], {}).get("count", 0),
            item[0],
        ),
        reverse=True,
    ):
        visible = [
            question for question in questions
            if not search_text or search_text.casefold() in question.text.casefold()
        ]
        if not visible:
            continue
        rank = chapter_rank.get(chapter)
        priority = rank["priority"] if rank else "Build foundations"
        label = f"{chapter} · {len(visible)} question(s)"
        if rank:
            label += f" · {priority} repeat priority"
        with st.expander(label):
            for question in sorted(visible, key=lambda item: (item.year, item.number)):
                key_suffix = hashlib.sha1(
                    f"{question.source}:{question.number}".encode("utf-8")
                ).hexdigest()[:10]
                solution_key = f"solution_{key_suffix}"
                with st.container(border=True):
                    st.markdown(f"**{question.year} · Question {question.number}**")
                    stem, options = split_question_options(question.text)
                    st.markdown(
                        f'<div class="question-stem">{html.escape(stem)}</div>',
                        unsafe_allow_html=True,
                    )
                    source_path = paper_paths.get((question.subject, question.source))
                    first_page = question.page_number
                    last_page = max(first_page, question.last_page_number or first_page)
                    solution_first_page = question.solution_page_number
                    solution_last_page = question.solution_last_page_number or solution_first_page
                    page_label = f"🖼️ View original PDF page · {first_page}"
                    if last_page > first_page:
                        page_label += f"–{last_page}"
                    with st.expander(page_label):
                        if source_path is None:
                            st.warning("The source PDF for this question could not be located.")
                        else:
                            try:
                                source_images = get_source_page_images(
                                    str(source_path),
                                    first_page,
                                    last_page,
                                )
                                for page_number, image in enumerate(source_images, start=first_page):
                                    st.image(
                                        image,
                                        caption=f"{question.year} source paper · page {page_number}",
                                        use_container_width=True,
                                    )
                                if solution_first_page is not None and solution_last_page is not None:
                                    st.markdown("**Original printed solution**")
                                    solution_images = get_source_page_images(
                                        str(source_path),
                                        solution_first_page,
                                        solution_last_page,
                                    )
                                    for page_number, image in enumerate(
                                        solution_images,
                                        start=solution_first_page,
                                    ):
                                        st.image(
                                            image,
                                            caption=f"{question.year} solution · page {page_number}",
                                            use_container_width=True,
                                        )
                            except Exception as exc:
                                st.error(f"Could not render the original PDF page: {exc}")
                    if any(unicodedata.category(char) == "Co" for char in question.text):
                        st.info(
                            "This PDF uses custom embedded glyphs without a Unicode map. "
                            "The extracted symbols are preserved unchanged; use the original page above for exact printed notation."
                        )
                        if source_path is not None:
                            st.download_button(
                                "Download original paper",
                                data=get_source_pdf_bytes(str(source_path)),
                                file_name=source_path.name,
                                mime="application/pdf",
                                key=f"source_{key_suffix}",
                            )
                    if options:
                        for label, option_text in options:
                            st.markdown(
                                '<div class="question-option">'
                                f'<strong>({label})</strong>{html.escape(option_text)}'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                    st.caption(f"Topic estimate: {question.topic} · Source: {question.source}")
                    if question.answer:
                        st.success(f"Marked answer found in source: **{question.answer}**")
                    if st.button("✨ Read symbols & explain from page", key=f"explain_{key_suffix}", type="primary"):
                        if not api_key:
                            st.info("Add your OpenAI API key in the sidebar to request a vision-based worked explanation.")
                        elif source_path is None:
                            st.error("The source PDF is unavailable, so a page-aware explanation cannot be generated.")
                        else:
                            try:
                                with st.spinner("Reading the original symbols and working through the solution…"):
                                    source_images = get_source_page_images(
                                        str(source_path),
                                        first_page,
                                        last_page,
                                    )
                                    solution_images = (
                                        get_source_page_images(
                                            str(source_path),
                                            solution_first_page,
                                            solution_last_page,
                                        )
                                        if solution_first_page is not None and solution_last_page is not None
                                        else ()
                                    )
                                    st.session_state[solution_key] = explain_question(
                                        question,
                                        api_key,
                                        source_images,
                                        solution_images,
                                        vision_model,
                                    )
                            except Exception as exc:
                                st.error(f"Could not generate the explanation: {exc}")
                    if solution_key in st.session_state:
                        st.markdown("**Step-by-step explanation**")
                        st.markdown(st.session_state[solution_key])

with prediction_tab:
    target_year = date.today().year + 1
    st.subheader(f"{target_year} KCET analysis-weighted practice papers")
    st.markdown(
        '<p class="section-note">These are practice mocks assembled from your past-paper questions. '
        'They emphasize recurring chapters; they are not predictions of the actual exam or a guarantee of questions.</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Each subject paper selects up to 60 distinct questions from the available extracted papers, "
        "weighted toward patterns seen repeatedly across years. Questions retain their source year and number. "
        "Check the current KEA syllabus and exam instructions before using this as a timed mock."
    )
    if st.button(
        f"Build / refresh all three {target_year} practice papers",
        type="primary",
        key="build_prediction_papers",
    ):
        st.session_state["prediction_papers"] = {
            subject: generate_practice_paper(papers, subject, target_year, question_count=60)
            for subject in SUBJECTS
        }
        st.session_state["prediction_papers_year"] = target_year

    prediction_papers = st.session_state.get("prediction_papers", {})
    if not prediction_papers:
        st.caption("Select the button above to build the Chemistry, Mathematics, and Physics practice papers.")
    else:
        subject_tabs = st.tabs(list(SUBJECTS))
        for subject_tab, subject in zip(subject_tabs, SUBJECTS):
            with subject_tab:
                selected_questions = prediction_papers.get(subject, [])
                st.markdown(f"#### {subject} · {len(selected_questions)} questions")
                if not selected_questions:
                    st.warning("There are no extracted source questions available for this subject.")
                    continue

                chapter_counts = Counter(question.chapter for question in selected_questions)
                top_chapters = sorted(chapter_counts.items(), key=lambda item: (-item[1], item[0]))
                st.markdown(
                    "**Chapter mix:** " + " · ".join(
                        f"{chapter}: {count}" for chapter, count in top_chapters[:8]
                    )
                )

                paper_markdown = [
                    f"# {target_year} KCET {subject} — Practice Mock",
                    "",
                    "> Analysis-weighted practice from previous papers; not an actual or guaranteed prediction.",
                    "",
                ]
                for number, question in enumerate(selected_questions, start=1):
                    stem, options = split_question_options(question.text)
                    paper_markdown.extend([
                        f"## Question {number}",
                        "",
                        stem,
                        "",
                    ])
                    for label, option_text in options:
                        paper_markdown.append(f"- **({label})** {option_text}")
                    paper_markdown.extend([
                        "",
                        f"*Chapter estimate: {question.chapter} · Source: {question.year}, Q{question.number}*",
                        "",
                    ])
                    with st.expander(
                        f"Question {number} · {question.chapter} · source {question.year}, Q{question.number}"
                    ):
                        st.markdown(f"**{stem}**")
                        if options:
                            for label, option_text in options:
                                st.markdown(f"- **({label})** {option_text}")
                        source_path = paper_paths.get((question.subject, question.source))
                        first_page = question.page_number
                        last_page = max(first_page, question.last_page_number or first_page)
                        if source_path is not None:
                            source_label = f"View original question page(s) {first_page}"
                            if last_page > first_page:
                                source_label += f"–{last_page}"
                            with st.expander(source_label):
                                try:
                                    source_images = get_source_page_images(
                                        str(source_path),
                                        first_page,
                                        last_page,
                                    )
                                    for page_number, image in enumerate(source_images, start=first_page):
                                        st.image(
                                            image,
                                            caption=f"{question.year} source paper · page {page_number}",
                                            use_container_width=True,
                                        )
                                except Exception as exc:
                                    st.error(f"Could not render the original PDF page: {exc}")
                        if question.answer:
                            st.caption(f"Source paper marked answer: {question.answer}")

                st.download_button(
                    f"Download {subject} mock (.md)",
                    data="\n".join(paper_markdown),
                    file_name=f"KCET_{target_year}_{subject.replace(' ', '_')}_practice_mock.md",
                    mime="text/markdown",
                    key=f"download_prediction_{subject}",
                )

with plan_tab:
    st.subheader(f"A focused {weeks}-week preparation plan")
    st.markdown(
        f'<p class="section-note">Built around {hours_per_week} study hours per week and the chapter recurrence in your papers. '
        "Use your official current-year syllabus to confirm topic coverage.</p>",
        unsafe_allow_html=True,
    )
    plan_subject = st.selectbox("Prioritize a subject", SUBJECTS, key="plan_subject")
    plan_chapters = available_chapters(papers, plan_subject)
    if plan_chapters:
        st.markdown("#### What to concentrate on")
        for row in plan_chapters:
            with st.expander(f"{row['chapter']} · {row['priority']} priority · {len(row['years'])} paper-years"):
                st.write(
                    f"Detected **{row['count']} question(s)** across "
                    f"{', '.join(map(str, row['years']))}. Work through the recurring problem style, "
                    "then revisit missed questions from your error log."
                )
                topics = sorted({
                    question.topic for question in all_questions
                    if question.subject == plan_subject and question.chapter == row["chapter"]
                })
                st.markdown("**Topics to practice:** " + "; ".join(topics))
    else:
        st.info("Chapter priorities will appear once questions are extracted for this subject.")

    st.markdown("#### Week-by-week rhythm")
    for item in build_study_plan(papers, weeks=weeks, hours_per_week=hours_per_week):
        with st.expander(f"Week {item['week']} · about {item['hours']} hours", expanded=item["week"] == 1):
            for task in item["focus"]:
                st.markdown(f"- {task}")
            st.caption(item["revision"])

    with st.expander("A repeatable weekly routine"):
        st.markdown(
            "- **Learn:** revise the chapter concepts and formulas from your textbook or class notes.\n"
            "- **Practice:** solve the chapter's past KCET questions under a timer before checking answers.\n"
            "- **Review:** record errors by cause (concept, calculation, reading, or time) and redo them after a few days.\n"
            "- **Simulate:** reserve regular sessions for mixed-subject, full-length timed practice and post-test review."
        )
    st.warning("Past-paper frequency cannot guarantee a question or a rank. Cover the current official syllabus and protect time for every subject.")

with resources_tab:
    st.subheader("Trusted starting points and chapter-specific lessons")
    st.markdown("Official syllabus and textbook references come first; video links open a YouTube search for the selected topic.")
    resource_subject = st.selectbox("Choose a subject for resources", SUBJECTS, key="resource_subject")
    chapters_for_resources = available_chapters(papers, resource_subject)
    if chapters_for_resources:
        for row in chapters_for_resources:
            with st.expander(f"{row['chapter']} · {row['priority']} priority"):
                for label, url in chapter_resources(resource_subject, row["chapter"]):
                    st.markdown(f"- [{label}]({url})")
                st.caption("Check that each lesson matches the latest KEA syllabus before relying on it.")
    else:
        for label, url in chapter_resources(resource_subject, "KCET syllabus"):
            st.markdown(f"- [{label}]({url})")

if notices:
    with st.expander(f"Paper-reading notes · {len(notices)}"):
        for notice in notices:
            st.markdown(f"- {notice}")
        st.caption("Question extraction is text-based. A PDF that is a scanned image needs OCR before its questions can be analyzed.")
