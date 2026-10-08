"""Offline-first analysis helpers for KCET previous-year question papers."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SUBJECTS = ("Chemistry", "Mathematics", "Physics")

# Keywords identify a likely chapter, not a definitive syllabus mapping. More
# specific chapter names are listed first where terms can overlap.
CHAPTER_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "Chemistry": {
        "Coordination Compounds": ("coordination compound", "ligand", "werner", "isomerism", "coordination number"),
        "Biomolecules": ("carbohydrate", "glucose", "amino acid", "protein", "vitamin", "nucleic acid"),
        "Polymers": ("polymer", "monomer", "nylon", "terylene", "bakelite", "rubber"),
        "Chemistry in Everyday Life": ("antacid", "antibiotic", "analgesic", "detergent", "food preservative"),
        "The d- and f-Block Elements": ("lanthanoid", "actinoid", "transition element", "d-block", "f-block"),
        "The p-Block Elements": ("p-block", "boron", "silicon", "nitrogen", "phosphorus", "halogen", "noble gas"),
        "The s-Block Elements": ("alkali metal", "alkaline earth", "s-block", "sodium", "magnesium"),
        "General Principles and Processes of Isolation of Elements": ("metallurgy", "ore", "roasting", "calcination", "refining", "copper matte", "froth flotation"),
        "Environmental Chemistry": ("greenhouse effect", "ozone depletion", "air pollution", "water pollution"),
        "Surface Chemistry": ("adsorption", "colloid", "emulsion", "catalysis", "tindall", "sulphur sol", "sulfur sol", "dispersed in"),
        "Chemical Kinetics": ("half-life", "order of reaction", "rate constant", "activation energy", "reaction rate", "rate of reaction", "concentration is decreased"),
        "Electrochemistry": ("electrochemical", "galvanic cell", "electrolytic", "nernst", "electrode potential", "faraday", "secondary cell", "recharged"),
        "Solutions": ("colligative", "osmotic pressure", "van't hoff", "van‟t hoff", "van't hoff's factor", "freezing point", "boiling point", "molarity", "molality", "hoff's factor"),
        "Chemical Equilibrium": ("equilibrium constant", "le chatelier", "ionic equilibrium", "ph of", "buffer solution", "equilibrium concentration", "concentration of oh"),
        "Chemical Thermodynamics": ("enthalpy", "entropy", "gibbs", "spontaneous", "heat of reaction", "bond energies", "delta h"),
        "Redox Reactions": ("oxidation number", "oxidising agent", "reducing agent", "redox"),
        "Chemical Bonding and Molecular Structure": ("hybridisation", "hybridization", "bond angle", "molecular orbital", "hydrogen bond", "dipole moment"),
        "Structure of Atom": ("quantum number", "orbital", "bohr", "photoelectric", "de broglie", "electronic configuration"),
        "Classification of Elements and Periodicity": ("periodic table", "ionisation enthalpy", "electronegativity", "atomic radius", "periodic property"),
        "Some Basic Concepts of Chemistry": ("mole concept", "molar mass", "empirical formula", "limiting reagent", "stoichiometry", "number of oxygen atoms", "number of atoms", "number of molecules"),
        "Organic Chemistry: Basic Principles and Techniques": ("inductive effect", "electrophile", "nucleophile", "isomer", "iupac", "iup ac", "resonance", "most acidic", "acidic compound"),
        "Hydrocarbons": ("alkane", "alkene", "alkyne", "benzene", "aromatic hydrocarbon", "hydrocarbon"),
        "Haloalkanes and Haloarenes": ("haloalkane", "haloarene", "alkyl halide", "aryl halide", "wurtz"),
        "Alcohols, Phenols and Ethers": ("phenol", "ether", "alcohol", "williamson", "ethanol", "ethoxy ethane"),
        "Aldehydes, Ketones and Carboxylic Acids": ("aldehyde", "ketone", "carboxylic acid", "tollens", "fehling", "phenyl hydrazine", "silver mirror"),
        "Amines": ("amine", "diazonium", "aniline", "carbylamine"),
        "States of Matter": ("ideal gas", "gas law", "compressibility factor", "kinetic theory of gases"),
        "Solid State": ("unit cell", "crystal", "packing efficiency", "schottky", "frenkel", "lattice"),
        "Isolation of Elements": ("electrolytic reduction", "blast furnace", "froth flotation"),
    },
    "Mathematics": {
        "Matrices": ("matrix", "matrices", "inverse of", "transpose", "matrix a", "matrix b", "identity matrix"),
        "Determinants": ("determinant", "minor and cofactor", "cramer", "determinant of"),
        "Three-Dimensional Geometry": ("direction cosines", "direction ratios", "plane", "skew lines", "3d"),
        "Vector Algebra": ("scalar product", "vector product", "cross product", "dot product", "unit vector"),
        "Differential Equations": ("differential equation", "general solution", "particular solution", "order and degree"),
        "Applications of Integrals": ("area bounded", "area under", "area enclosed"),
        "Integrals": ("integration", "integral", "by parts", "definite integral", " dx", "integrate"),
        "Applications of Derivatives": ("increasing function", "decreasing function", "rate of change", "maxima", "minima", "tangent and normal"),
        "Continuity and Differentiability": ("continuity", "differentiability", "chain rule", "implicit differentiation", "derivative", "dy dx", "dy/dx", "differentiate"),
        "Limits": ("limit", "l'hospital", "lim x", "limiting value"),
        "Probability": ("probability", "bayes", "random variable", "binomial distribution"),
        "Linear Programming": ("linear programming", "feasible region", "objective function"),
        "Statistics": ("mean deviation", "variance", "standard deviation"),
        "Relations and Functions": ("one-one", "onto", "bijective", "domain and range", "inverse function", "relation", "binary operation", "commutative", "associative"),
        "Inverse Trigonometric Functions": ("inverse trigonometric", "tan inverse", "sin inverse", "cos inverse", "principal value", "sin-1", "tan-1", "cos-1"),
        "Trigonometric Functions": ("trigonometric", "sine", "cosine", "tangent", "cot", "sin ", "cos ", "tan "),
        "Conic Sections": ("hyperbola", "parabola", "ellipse", "eccentricity", "focus", "directrix"),
        "Straight Lines": ("slope", "straight line", "distance between the lines", "equation of a line"),
        "Permutations and Combinations": ("permutation", "combination", "arrangements", "ways in which"),
        "Binomial Theorem": ("binomial", "middle term", "independent term", "coefficient of"),
        "Sequences and Series": ("sequence", "series", "arithmetic progression", "geometric progression", "common difference", "common ratio"),
        "Complex Numbers": ("complex number", "imaginary", "modulus", "argument", "conjugate"),
        "Sets": ("set a", "union", "intersection", "complement", "number of elements"),
    },
    "Physics": {
        "Semiconductor Electronics": ("semiconductor", "transistor", "diode", "rectifier", "logic gate", "p-n junction"),
        "Atoms and Nuclei": ("radioactive", "half-life", "nuclear", "binding energy", "bohr model", "fission", "fusion"),
        "Dual Nature of Radiation and Matter": ("photoelectric", "de broglie", "work function", "photon"),
        "Electromagnetic Waves": ("electromagnetic wave", "displacement current", "infrared", "ultraviolet"),
        "Wave Optics": ("interference", "diffraction", "polarisation", "young's double", "fringe width"),
        "Ray Optics and Optical Instruments": ("lens", "mirror", "refractive", "prism", "optical instrument", "focal length"),
        "Electromagnetic Induction and Alternating Currents": ("induced emf", "inductance", "faraday's law", "lenz", "alternating current", "transformer", "reactance"),
        "Moving Charges and Magnetism": ("magnetic field", "lorentz force", "biot-savart", "ampere's law", "cyclotron", "galvanometer", "solenoid", "proton"),
        "Magnetism and Matter": ("magnetic moment", "diamagnetic", "paramagnetic", "ferromagnetic", "bar magnet"),
        "Electric Charges and Fields": ("electric field", "electric flux", "gauss", "dipole", "coulomb"),
        "Electrostatic Potential and Capacitance": ("potential difference", "capacitance", "capacitor", "equipotential", "electrostatic potential"),
        "Current Electricity": ("resistance", "resistivity", "kirchhoff", "ohm", "drift velocity", "wheatstone", "electric current", "mobility of free electrons", "conductor", "network"),
        "Oscillations": ("simple harmonic", "pendulum", "oscillation", "spring constant", "time period", "maximum acceleration", "maximum speed"),
        "Waves": ("sound wave", "standing wave", "doppler", "wave speed", "frequency of a wave", "source of sound", "observer measures the frequency"),
        "Thermodynamics": ("first law of thermodynamics", "isothermal", "adiabatic", "heat engine", "entropy", "carnot"),
        "Kinetic Theory": ("kinetic theory", "mean free path", "degrees of freedom", "rms speed"),
        "Thermal Properties of Matter": ("specific heat", "latent heat", "thermal expansion", "calorimetry", "conduction", "cools from", "heat reservoirs", "conduct most heat"),
        "Mechanical Properties of Fluids": ("viscosity", "surface tension", "bernoulli", "pascal", "buoyant", "terminal velocity", "ideal fluid", "pipe of circular", "flow through a pipe"),
        "Mechanical Properties of Solids": ("young's modulus", "bulk modulus", "shear modulus", "stress and strain", "strain produced", "spring is stretched"),
        "Gravitation": ("gravitational", "escape velocity", "orbital velocity", "satellite", "kepler", "acceleration due to gravity", "center of earth", "centre of earth"),
        "System of Particles and Rotational Motion": ("torque", "moment of inertia", "angular momentum", "centre of mass", "center of mass", "rolling", "roll down"),
        "Work, Energy and Power": ("work done", "kinetic energy", "potential energy", "power", "collision"),
        "Laws of Motion": ("friction", "newton", "inertia", "impulse", "acceleration in a train"),
        "Motion in a Plane": ("projectile", "vector", "uniform circular", "relative velocity"),
        "Motion in a Straight Line": ("average velocity", "displacement", "free fall", "uniform acceleration"),
        "Units and Measurements": ("significant figure", "dimensional", "measurement", "error in", "least count"),
    },
}

CHAPTER_TOPICS: dict[str, dict[str, tuple[str, ...]]] = {
    "Chemistry": {
        "Chemical Kinetics": {"First-order reactions and half-life": ("half-life", "first order", "rate constant")},
        "Electrochemistry": {"Cells and electrode potentials": ("galvanic cell", "electrode potential", "emf"), "Electrolysis and Faraday's laws": ("electrolysis", "faraday")},
        "Solutions": {"Colligative properties": ("colligative", "osmotic pressure", "freezing point", "boiling point"), "Concentration and mole fraction": ("molarity", "molality", "mole fraction")},
        "Solid State": {"Crystal lattices and unit cells": ("unit cell", "edge centre", "packing efficiency"), "Point defects": ("schottky", "frenkel", "defect")},
    },
    "Mathematics": {
        "Matrices": {"Inverse and operations": ("inverse", "matrix multiplication", "transpose")},
        "Determinants": {"Evaluation and properties": ("determinant", "minor", "cofactor")},
        "Conic Sections": {"Hyperbola and eccentricity": ("hyperbola", "eccentricity"), "Parabola and ellipse": ("parabola", "ellipse")},
        "Permutations and Combinations": {"Counting arrangements": ("arrangement", "ways", "seated")},
        "Probability": {"Conditional probability and Bayes' theorem": ("conditional", "bayes"), "Probability distributions": ("random variable", "distribution")},
        "Integrals": {"Definite and indefinite integrals": ("definite", "indefinite", "integral")},
        "Vector Algebra": {"Dot and cross products": ("dot product", "scalar product", "cross product", "vector product")},
    },
    "Physics": {
        "Motion in a Plane": {"Projectile motion": ("projectile", "range"), "Vectors and relative motion": ("vector", "relative velocity")},
        "Laws of Motion": {"Friction and Newton's laws": ("friction", "newton", "acceleration in a train")},
        "Current Electricity": {"Resistance networks and circuits": ("resistance", "resistor", "kirchhoff", "circuit")},
        "Electromagnetic Induction and Alternating Currents": {"Induction and AC circuits": ("induced emf", "alternating current", "transformer", "reactance")},
        "Ray Optics and Optical Instruments": {"Mirrors, lenses and refraction": ("lens", "mirror", "refractive", "focal length")},
        "Atoms and Nuclei": {"Radioactive decay and half-life": ("radioactive", "half-life", "decay"), "Atomic and nuclear models": ("bohr", "nuclear", "binding energy")},
        "Semiconductor Electronics": {"Diodes, transistors and logic gates": ("diode", "transistor", "logic gate", "rectifier")},
    },
}


@dataclass(frozen=True)
class Paper:
    subject: str
    year: int
    path: Path
    questions: tuple["Question", ...]
    note: str = ""


@dataclass(frozen=True)
class Question:
    subject: str
    year: int
    number: int
    text: str
    chapter: str
    topic: str
    answer: str | None
    source: str
    page_number: int = 1
    last_page_number: int | None = None
    solution_page_number: int | None = None
    solution_last_page_number: int | None = None


def read_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def page_spans(page_texts: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for page_text in page_texts:
        end = cursor + len(page_text)
        spans.append((cursor, end))
        cursor = end + 1
    return spans


def page_number_for_offset(offset: int, spans: list[tuple[int, int]]) -> int:
    starts = [start for start, _ in spans]
    return min(bisect_right(starts, offset), len(spans)) or 1


def extract_solution_page_ranges(
    page_texts: list[str],
    subject: str,
) -> dict[int, tuple[int, int]]:
    full_text = "\n".join(page_texts)
    subject_alias = "math(?:ematics|s)" if subject == "Mathematics" else re.escape(subject)
    heading = re.search(
        rf"\bSolutions?\s*[–—-]\s*{subject_alias}\b",
        full_text,
        flags=re.IGNORECASE,
    )
    if heading is None:
        return {}

    section = full_text[heading.end():]
    section_offset = heading.end()
    spans = page_spans(page_texts)
    markers_by_number: dict[int, re.Match[str]] = {}
    for marker in re.finditer(r"(?m)^[ \t]{0,8}(\d{1,2})\.\s+(?=\S)", section):
        number = int(marker.group(1))
        if 1 <= number <= 60:
            markers_by_number.setdefault(number, marker)
    ordered = sorted(markers_by_number.values(), key=lambda marker: marker.start())
    ranges: dict[int, tuple[int, int]] = {}
    for index, marker in enumerate(ordered):
        end = ordered[index + 1].start() if index + 1 < len(ordered) else len(section)
        content_end = marker.start() + len(section[marker.start():end].rstrip())
        first_page = page_number_for_offset(section_offset + marker.start(), spans)
        last_page = page_number_for_offset(
            section_offset + max(marker.start(), content_end - 1),
            spans,
        )
        ranges[int(marker.group(1))] = (first_page, max(first_page, last_page))
    return ranges


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_question_options(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate a stem from four sequential multiple-choice options when present."""
    marker_pattern = re.compile(
        r"(?<!\w)(?:\(\s*(?P<parenthesized>[A-D1-48O0])\s*\)|"
        r"(?P<letter>[A-D])[.)]|(?P<digit>[1-4])[.)])\s*"
    )
    markers = list(marker_pattern.finditer(text))
    labels = []
    for marker in markers:
        label = marker.group("parenthesized") or marker.group("letter") or marker.group("digit")
        labels.append({"8": "B", "O": "D", "0": "D"}.get(label, label))

    best_group: list[int] | None = None
    best_score: tuple[int, int, int] | None = None
    for start, label in enumerate(labels):
        sequence = ["A", "B", "C", "D"] if label == "A" else ["1", "2", "3", "4"] if label == "1" else []
        if not sequence:
            continue
        group = [start]
        cursor = start + 1
        for expected in sequence[1:]:
            next_index = next(
                (index for index in range(cursor, len(labels)) if labels[index] == expected),
                None,
            )
            if next_index is None:
                break
            group.append(next_index)
            cursor = next_index + 1
        if len(group) != 4:
            continue
        unparenthesized = sum(
            markers[index].group("parenthesized") is None
            for index in group
        )
        score = (unparenthesized, group[-1] - group[0] - 3, group[0])
        if best_score is None or score < best_score:
            best_group = group
            best_score = score

    if best_group is not None:
        canonical = ["A", "B", "C", "D"] if labels[best_group[0]] == "A" else ["1", "2", "3", "4"]
        options = []
        for position, marker_index in enumerate(best_group):
            marker = markers[marker_index]
            next_marker = markers[best_group[position + 1]] if position < 3 else None
            end = next_marker.start() if next_marker else len(text)
            options.append((canonical[position], text[marker.end():end].strip()))
        return text[:markers[best_group[0]].start()].strip(), options
    return text.strip(), []


def _question_markers(text: str) -> tuple[list[re.Match[str]], str]:
    labelled = list(re.finditer(r"\bQuestion\s+(\d{1,2})\s*:", text, flags=re.IGNORECASE))
    if len(labelled) >= 2:
        return labelled, "labelled"
    dotted = list(re.finditer(r"(?<![\w.])(\d{1,2})\.\s+(?=\S)", text))
    return dotted, "numbered"


def extract_questions(
    text: str,
    subject: str,
    year: int,
    source: str,
    source_page_spans: list[tuple[int, int]] | None = None,
    source_solution_pages: dict[int, tuple[int, int]] | None = None,
) -> list[Question]:
    markers, marker_type = _question_markers(text)
    chosen: list[re.Match[str]] = []
    expected = 1
    for marker in markers:
        number = int(marker.group(1))
        if number == 1 and len(chosen) >= 50:
            break
        if number == expected:
            chosen.append(marker)
            expected += 1
            if len(chosen) == 60:
                break
    if not chosen:
        return []

    questions: list[Question] = []
    for index, marker in enumerate(chosen):
        end = chosen[index + 1].start() if index + 1 < len(chosen) else len(text)
        raw = text[marker.end():end]
        content_end = marker.end() + len(raw.rstrip())
        answer_match = re.search(r"\bAnswer\s+\d{1,2}\s*:\s*\(?\s*([A-D])\s*\)?", raw, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else None
        if answer_match:
            raw = raw[:answer_match.start()]
        if marker_type == "numbered":
            raw = re.sub(r"^\s*", "", raw)
        clean = normalize_text(raw)
        clean = re.sub(r"^(?:Question\s+\d+\s*:\s*)", "", clean, flags=re.IGNORECASE)
        if len(clean) < 12:
            continue
        chapter, topic = classify_question(subject, clean)
        solution_range = (source_solution_pages or {}).get(int(marker.group(1)))
        questions.append(Question(
            subject=subject,
            year=year,
            number=int(marker.group(1)),
            text=clean,
            chapter=chapter,
            topic=topic,
            answer=answer,
            source=source,
            page_number=page_number_for_offset(marker.start(), source_page_spans) if source_page_spans else 1,
            last_page_number=(
                page_number_for_offset(max(marker.start(), content_end - 1), source_page_spans)
                if source_page_spans else 1
            ),
            solution_page_number=solution_range[0] if solution_range else None,
            solution_last_page_number=solution_range[1] if solution_range else None,
        ))
    return questions


def classify_question(subject: str, text: str) -> tuple[str, str]:
    lowered = text.casefold().replace("’", "'")
    matches: list[tuple[int, int, str]] = []
    for chapter, keywords in CHAPTER_KEYWORDS[subject].items():
        found = [keyword for keyword in keywords if keyword.casefold() in lowered]
        if found:
            matches.append((len(found), max(map(len, found)), chapter))
    if not matches:
        return "Needs review", "Chapter not confidently detected"
    chapter = max(matches)[2]
    topic = "Core concepts and standard applications"
    for candidate, topic_map in CHAPTER_TOPICS.get(subject, {}).items():
        if candidate == chapter:
            topic_matches = [
                (sum(keyword.casefold() in lowered for keyword in keywords), max(map(len, keywords)), label)
                for label, keywords in topic_map.items()
                if any(keyword.casefold() in lowered for keyword in keywords)
            ]
            if topic_matches:
                topic = max(topic_matches)[2]
            break
    return chapter, topic


def load_papers(root: Path) -> tuple[list[Paper], list[str]]:
    papers: list[Paper] = []
    notices: list[str] = []
    for subject in SUBJECTS:
        folder = root / ("maths" if subject == "Mathematics" else subject.casefold())
        if not folder.is_dir():
            notices.append(f"Missing subject folder: {folder.name}/")
            continue
        for path in sorted(folder.glob("*.pdf")):
            year_match = re.search(r"\b(20\d{2})\b", path.name)
            if not year_match:
                notices.append(f"Skipped {path.name}: could not determine the paper year.")
                continue
            year = int(year_match.group(1))
            try:
                page_texts = read_pdf_pages(path)
                text = "\n".join(page_texts)
            except Exception as exc:
                notices.append(f"Could not read {path.name}: {exc}")
                papers.append(Paper(subject, year, path, (), "PDF could not be read"))
                continue
            first_pages = text[:4000]
            is_answer_sheet = (
                bool(re.search(r"answer\s*keys?|provisional answer", first_pages, re.IGNORECASE))
                and not re.search(r"\bQuestion\s+\d+\s*:", text, re.IGNORECASE)
            )
            questions = [] if is_answer_sheet else extract_questions(
                text,
                subject,
                year,
                path.name,
                page_spans(page_texts),
                extract_solution_page_ranges(page_texts, subject),
            )
            if not questions:
                key_label = "answer key / answer sheet" if is_answer_sheet else "no numbered question text"
                note = f"Detected {key_label}; no question statements were extracted."
                notices.append(f"{subject} {year}: {path.name} — {note}")
                papers.append(Paper(subject, year, path, (), note))
            else:
                if len(questions) < 60:
                    notices.append(
                        f"{subject} {year}: extracted {len(questions)} of the expected 60 question numbers from {path.name}; "
                        "check the PDF text quality before treating its counts as complete."
                    )
                papers.append(Paper(subject, year, path, tuple(questions)))
    return papers, notices


def rank_patterns(papers: list[Paper]) -> list[dict[str, Any]]:
    usable_by_subject = Counter(paper.subject for paper in papers if paper.questions)
    patterns: dict[tuple[str, str, str], dict[str, Any]] = {}
    for paper in papers:
        for question in paper.questions:
            if question.chapter == "Needs review":
                continue
            key = (question.subject, question.chapter, question.topic)
            row = patterns.setdefault(key, {"subject": question.subject, "chapter": question.chapter,
                                            "topic": question.topic, "years": set(), "count": 0,
                                            "examples": []})
            row["years"].add(question.year)
            row["count"] += 1
            if len(row["examples"]) < 3:
                row["examples"].append(question)
    ranked = []
    for row in patterns.values():
        year_count = len(row["years"])
        coverage = year_count / usable_by_subject[row["subject"]]
        row["years"] = sorted(row["years"])
        row["coverage"] = coverage
        row["priority"] = "High" if coverage >= 0.65 else "Medium" if coverage >= 0.35 else "Build foundations"
        ranked.append(row)
    return sorted(ranked, key=lambda item: (-item["coverage"], -item["count"], item["subject"], item["chapter"], item["topic"]))


def build_study_plan(papers: list[Paper], weeks: int = 8, hours_per_week: int = 12) -> list[dict[str, Any]]:
    ranked = rank_patterns(papers)
    per_subject: dict[str, list[dict[str, Any]]] = {subject: [] for subject in SUBJECTS}
    for row in ranked:
        if row["chapter"] == "Needs review":
            continue
        if row["chapter"] not in [item["chapter"] for item in per_subject[row["subject"]]]:
            per_subject[row["subject"]].append(row)
    schedule: list[dict[str, Any]] = []
    for week in range(1, weeks + 1):
        tasks = []
        for subject in SUBJECTS:
            chapters = per_subject[subject]
            if chapters:
                row = chapters[(week - 1) % len(chapters)]
                tasks.append(f"{subject}: {row['chapter']} — {row['topic']} and worked examples")
            else:
                tasks.append(f"{subject}: revise core concepts from the current official syllabus")
        tasks.extend([
            "Solve timed KCET questions, then review every mistake",
            "Finish with a short cumulative recall session",
        ])
        schedule.append({
            "week": week,
            "hours": hours_per_week,
            "focus": tasks,
            "revision": "Use roughly half your study time for new concepts, one-third for timed questions, and the rest for error-log review.",
        })
    return schedule


def available_chapters(papers: list[Paper], subject: str) -> list[dict[str, Any]]:
    return [row for row in rank_patterns(papers) if row["subject"] == subject and row["chapter"] != "Needs review"]
