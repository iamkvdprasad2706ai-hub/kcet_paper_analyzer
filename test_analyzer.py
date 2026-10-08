import unittest
from pathlib import Path

from analyzer import (
    Paper,
    Question,
    build_study_plan,
    classify_question,
    extract_questions,
    generate_practice_paper,
    load_papers,
    normalize_text,
    page_number_for_offset,
    page_spans,
    rank_patterns,
    split_question_options,
)


ROOT = Path(__file__).resolve().parent


class QuestionExtractionTests(unittest.TestCase):
    def test_practice_paper_is_repeatable_unique_and_recurrence_weighted(self):
        papers = []
        for year in (2021, 2022, 2023, 2024):
            questions = [
                Question(
                    "Physics", year, number, f"recurring chapter problem {year} {number}",
                    "Current Electricity", "Circuit calculations", None, f"{year}.pdf",
                )
                for number in range(1, 21)
            ]
            if year == 2024:
                questions.extend(
                    Question(
                        "Physics", year, number, f"rare chapter problem {number}",
                        "Wave Optics", "Diffraction", None, f"{year}.pdf",
                    )
                    for number in range(21, 31)
                )
            papers.append(Paper("Physics", year, Path(f"{year}.pdf"), tuple(questions)))

        first = generate_practice_paper(papers, "Physics", 2027, question_count=60)
        second = generate_practice_paper(papers, "Physics", 2027, question_count=60)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 60)
        self.assertEqual(len({question.text for question in first}), 60)
        self.assertGreater(
            sum(question.chapter == "Current Electricity" for question in first),
            sum(question.chapter == "Wave Optics" for question in first),
        )
        self.assertEqual(generate_practice_paper([], "Physics", 2027), [])

    def test_page_spans_map_question_text_to_original_pdf_pages(self):
        spans = page_spans(["Question 1: first half", "second half Question 2: next"])
        self.assertEqual(page_number_for_offset(0, spans), 1)
        self.assertEqual(page_number_for_offset(spans[1][0], spans), 2)
        questions = extract_questions(
            "Question 1: first half\nsecond half\nQuestion 2: next question body",
            "Chemistry",
            2018,
            "fixture.pdf",
            spans,
        )
        self.assertEqual(questions[0].page_number, 1)
        self.assertEqual(questions[0].last_page_number, 2)
        self.assertEqual(questions[1].page_number, 2)

    def test_text_normalization_preserves_original_special_characters(self):
        symbols = "PCl₅ H₂SO₄ Δ→ − α β ° ‟ \uf03d\uf02b\uf02d\uf0b0\uf0ce"
        self.assertEqual(normalize_text(f"  {symbols}  "), symbols)

    def test_extracts_labelled_questions_and_inline_answers(self):
        text = (
            "Question 1: What is the answer to this chemistry question? "
            "(A) one (B) two Answer 1: (B) "
            "Question 2: Which concept is tested in this question? "
            "(A) three (B) four Answer 2: (A)"
        )
        questions = extract_questions(text, "Chemistry", 2018, "fixture.pdf")
        self.assertEqual([question.number for question in questions], [1, 2])
        self.assertEqual([question.answer for question in questions], ["B", "A"])

    def test_stops_at_second_numbered_answer_key_sequence(self):
        first = " ".join(
            f"{number}. Question {number} describes a useful physics concept? (1) A (2) B"
            for number in range(1, 61)
        )
        key = " ".join(f"{number}. (A)" for number in range(1, 61))
        questions = extract_questions(first + " Answers " + key, "Physics", 2016, "fixture.pdf")
        self.assertEqual(len(questions), 60)
        self.assertEqual(questions[-1].number, 60)

    def test_classifies_chapter_and_topic(self):
        chapter, topic = classify_question(
            "Physics",
            "A projectile is projected at an angle. Find its range.",
        )
        self.assertEqual(chapter, "Motion in a Plane")
        self.assertEqual(topic, "Projectile motion")

    def test_splits_parenthesized_numbered_options_onto_separate_choices(self):
        stem, options = split_question_options(
            "Which property applies? (1) Osmotic pressure (2) Optical activity "
            "(3) Freezing point (4) Boiling point"
        )
        self.assertEqual(stem, "Which property applies?")
        self.assertEqual(
            options,
            [
                ("1", "Osmotic pressure"),
                ("2", "Optical activity"),
                ("3", "Freezing point"),
                ("4", "Boiling point"),
            ],
        )

    def test_splits_lettered_options_and_preserves_formula_text(self):
        stem, options = split_question_options(
            "Find the value. A. x + 1 B. x - 1 C. x = 0 D. Does not exist"
        )
        self.assertEqual(stem, "Find the value.")
        self.assertEqual(
            options,
            [
                ("A", "x + 1"),
                ("B", "x - 1"),
                ("C", "x = 0"),
                ("D", "Does not exist"),
            ],
        )

    def test_does_not_treat_abbreviation_in_option_text_as_another_choice(self):
        stem, options = split_question_options(
            "Choose the true statement. (A) are in A.P. (B) are not in A.P. "
            "(C) are not in G.P. (D) are in G.P."
        )
        self.assertEqual(stem, "Choose the true statement.")
        self.assertEqual(options[0], ("A", "are in A.P."))
        self.assertEqual(options[-1], ("D", "are in G.P."))

    def test_leaves_text_intact_if_four_options_are_not_detected(self):
        text = "Consider the point A and the function f(x)."
        self.assertEqual(split_question_options(text), (text, []))

    def test_ranks_recurrence_before_single_year_patterns(self):
        repeated = [
            Question("Physics", year, 1, "projectile range", "Motion in a Plane", "Projectile motion", None, "x")
            for year in (2019, 2020, 2023)
        ]
        once = [
            Question("Physics", 2023, 2, "capacitor", "Electrostatic Potential and Capacitance",
                     "Core concepts and standard applications", None, "x")
        ]
        papers = [
            Paper("Physics", year, Path(f"{year}.pdf"), tuple(q for q in repeated + once if q.year == year))
            for year in (2019, 2020, 2023)
        ]
        ranked = rank_patterns(papers)
        self.assertEqual(ranked[0]["chapter"], "Motion in a Plane")
        self.assertEqual(ranked[0]["priority"], "High")
        schedule = build_study_plan(papers, weeks=4, hours_per_week=10)
        self.assertEqual(len(schedule), 4)
        self.assertTrue(all(any(subject in task for task in schedule[0]["focus"])
                            for subject in ("Chemistry", "Mathematics", "Physics")))

    def test_reads_workspace_papers_and_reports_answer_key_sheets(self):
        papers, notices = load_papers(ROOT)
        self.assertEqual(len(papers), 21)
        self.assertTrue(any("answer key" in notice.lower() for notice in notices))
        maths_2022 = next(paper for paper in papers if paper.subject == "Mathematics" and paper.year == 2022)
        self.assertEqual(maths_2022.questions, ())
        chemistry_2016_q18 = next(
            question for paper in papers
            if paper.subject == "Chemistry" and paper.year == 2016
            for question in paper.questions
            if question.number == 18
        )
        self.assertEqual((chemistry_2016_q18.page_number, chemistry_2016_q18.last_page_number), (7, 8))
        self.assertEqual(
            (chemistry_2016_q18.solution_page_number, chemistry_2016_q18.solution_last_page_number),
            (32, 33),
        )
        self.assertGreater(sum(len(paper.questions) for paper in papers), 500)
        self.assertTrue(all(question.text for paper in papers for question in paper.questions))
        for subject in ("Chemistry", "Mathematics", "Physics"):
            mock = generate_practice_paper(papers, subject, 2027, question_count=60)
            self.assertEqual(len(mock), 60)
            self.assertEqual(len({question.text.casefold() for question in mock}), 60)
            self.assertTrue(all(question.subject == subject for question in mock))
        total = sum(len(paper.questions) for paper in papers)
        classified = sum(
            question.chapter != "Needs review"
            for paper in papers
            for question in paper.questions
        )
        self.assertGreater(classified / total, 0.55)


if __name__ == "__main__":
    unittest.main()
