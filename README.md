# KCET Paper Analyzer

A local Streamlit study app for the PDF papers in `chemistry/`, `maths/`, and
`physics/`. It extracts available question text, groups it by an estimated
chapter, ranks recurring chapters and question styles by the number of usable
paper years, and builds a study schedule from that evidence.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The `.env` file is optional. To request step-by-step explanations, replace the
placeholder in `.env` with your OpenAI API key. You can instead paste a key
into the app's password field for the current session. A key is used only when
you click **Read symbols & explain from page**. The app sends the selected
question text and rendered question/solution page images to the configured
OpenAI vision model; it does not upload the complete paper. API usage may
incur charges. Keep `.env` private and do not commit it. Set
`OPENAI_VISION_MODEL` to choose another compatible vision model; the default
is `gpt-4.1`.

## How the analysis works

- PDFs are read locally with `pypdf`. Question text is extracted from the
  document's text layer; image-only scans need OCR and are reported rather than
  silently treated as complete. Papers with fewer than 60 extracted questions
  are flagged in the app for review.
- Subject-specific keyword rules estimate chapters and selected subtopics.
  Review those estimates, especially where the extracted PDF text is damaged.
- In the question bank, recognized four-choice questions display the question
  stem separately and place each answer option on its own line. Scanned PDFs
  with damaged or missing option text may still need manual review.
- Text normalization preserves Unicode punctuation, mathematical notation,
  and private-use glyphs exactly as extracted. Some older PDFs use embedded
  fonts without Unicode maps; affected questions show a notice, an original
  page preview, and a download button for checking their printed notation.
- Original PDF pages are rendered for each question so diagrams, graphs,
  tables, and source typography remain available. On request, the vision model
  reads that page image alongside the extracted text, then provides a detailed
  explanation with readable mathematical notation. Model transcription should
  still be checked against the displayed source page when the scan is unclear.
- Papers that include printed worked solutions have those solution pages
  mapped back to their questions; the original solution page is shown and sent
  with the question page for visual explanations.
- The separate next-year practice-paper tab assembles up to 60 distinct
  historical questions per subject with recurrence-weighted chapter selection.
  Each paper can be downloaded as Markdown and includes its source year and
  question number. It is a practice mock, not a prediction of the actual exam.
- Pattern priority is based on the number of distinct usable paper years in
  which the chapter/topic appears. It describes observed recurrence, not a
  forecast, official weightage, or guaranteed rank.
- Some 2022/2023 files in the supplied folders are answer-key sheets, not
  question papers. They are called out in the paper-reading notes and do not
  contribute invented questions to the rankings.
- The app only displays a marked answer when it can extract an answer embedded
  beside the question. An AI explanation will explicitly note when no source
  answer was found.

## Tests

```bash
python -m unittest -v
```

Always confirm chapter priorities and planned topics against the current
official KEA syllabus. Past papers are useful practice, not a promise about the
next exam.
