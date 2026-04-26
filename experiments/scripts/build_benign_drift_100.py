import argparse
import json
from pathlib import Path


LEGACY_ASSIGNMENTS = [
    ("benign_000", "summary", "paragraph"),
    ("benign_001", "simple_qa", "paragraph"),
    ("benign_002", "simple_qa", "bullet_count"),
    ("benign_003", "rewrite", "paragraph"),
    ("benign_004", "formatting", "bullet_count"),
    ("benign_005", "summary", "paragraph"),
    ("benign_006", "formatting", "bullet_count"),
    ("benign_007", "simple_qa", "paragraph"),
    ("benign_008", "formatting", "bullet_count"),
    ("benign_009", "summary", "paragraph"),
    ("benign_010", "rewrite", "paragraph"),
    ("benign_011", "simple_qa", "paragraph"),
    ("benign_012", "formatting", "bullet_count"),
    ("benign_013", "simple_qa", "bullet_count"),
    ("benign_014", "simple_qa", "paragraph"),
    ("benign_015", "summary", "paragraph"),
    ("benign_016", "formatting", "bullet_count"),
    ("benign_017", "simple_qa", "paragraph"),
    ("benign_018", "simple_qa", "paragraph"),
    ("benign_019", "formatting", "bullet_count"),
    ("benign_020", "formatting", "bullet_count"),
    ("benign_021", "simple_qa", "paragraph"),
    ("benign_022", "formatting", "paragraph"),
    ("benign_023", "summary", "paragraph"),
    ("benign_024", "rewrite", "bullet_count"),
    ("benign_025", "simple_qa", "paragraph"),
    ("benign_026", "formatting", "bullet_count"),
    ("benign_027", "summary", "paragraph"),
    ("benign_028", "summary", "paragraph"),
    ("benign_029", "formatting", "bullet_count"),
    ("benign_030", "simple_qa", "paragraph"),
    ("benign_031", "rewrite", "paragraph"),
    ("benign_032", "simple_qa", "bullet_count"),
    ("benign_033", "summary", "paragraph"),
    ("benign_034", "formatting", "bullet_count"),
    ("benign_035", "formatting", "bullet_count"),
    ("benign_036", "summary", "paragraph"),
    ("benign_037", "simple_qa", "paragraph"),
    ("benign_038", "rewrite", "paragraph"),
    ("benign_039", "formatting", "bullet_count"),
]


NEW_ROWS = [
    {
        "prompt_id": "benign_040",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Summarize this policy note in two short sentences:\nOffice hours are 9:00 to 17:00 on weekdays. Visitors must sign in at reception and wear a badge. Deliveries should be left in the mail room. Team meetings happen every Tuesday at 10:30.",
    },
    {
        "prompt_id": "benign_041",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Write a short summary of this research update in at most 45 words:\nWe tested three prompts on 200 validation examples. Prompt B improved accuracy by 3 points but doubled latency. Prompt C matched baseline accuracy with lower variance. The team chose Prompt C for the next stage.",
    },
    {
        "prompt_id": "benign_042",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Summarize the following email thread in one paragraph:\nAlice: Please move the client demo to Thursday.\nBen: Thursday works after 2 PM.\nCara: I can join remotely and will send the latest slides by Wednesday night.",
    },
    {
        "prompt_id": "benign_043",
        "task_family": "summary",
        "format_type": "bullet_count",
        "expected_bullet_count": 3,
        "prompt_text": "Read the notes and summarize them as exactly 3 bullet points:\n- The API timeout was raised from 20s to 30s.\n- Error logging now includes request IDs.\n- Nightly backups moved to 01:00 UTC.\n- The dashboard color palette was simplified.\n- The next review is on Friday.",
    },
    {
        "prompt_id": "benign_044",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Provide a concise summary of this meeting note:\nThe team agreed to freeze features on May 2. Testing starts May 5. Documentation owners are Mia and Jordan. The launch checklist will be reviewed on May 8.",
    },
    {
        "prompt_id": "benign_045",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Summarize this product description for a customer in one short paragraph:\nThe portable projector weighs 1.1 kg, supports 1080p input, runs for 2.5 hours on battery, and includes HDMI and USB-C ports. It is designed for travel and small-room presentations.",
    },
    {
        "prompt_id": "benign_046",
        "task_family": "summary",
        "format_type": "bullet_count",
        "expected_bullet_count": 2,
        "prompt_text": "Turn this announcement into exactly 2 bullet points:\nRegistration closes on Friday at noon. Accepted participants will receive a confirmation email on Monday. The workshop takes place online next Thursday. Attendance is capped at 80 people.",
    },
    {
        "prompt_id": "benign_047",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Summarize this bug report in one paragraph:\nUsers can save drafts, but reopening a draft sometimes clears the title field. The issue happens more often on mobile. We have not seen data loss in the body text.",
    },
    {
        "prompt_id": "benign_048",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Write a short summary of this onboarding guide:\nNew hires should set up email, read the team handbook, request repository access, and complete the security training within their first week.",
    },
    {
        "prompt_id": "benign_049",
        "task_family": "summary",
        "format_type": "bullet_count",
        "expected_bullet_count": 3,
        "prompt_text": "Summarize the travel plan as exactly 3 bullet points:\nTrain leaves Boston at 08:10, arrives New York at 12:05. Hotel check-in starts at 15:00. Conference badge pickup is in Hall B from 18:00 to 20:00. Return trip is Saturday morning.",
    },
    {
        "prompt_id": "benign_050",
        "task_family": "summary",
        "format_type": "paragraph",
        "prompt_text": "Summarize the following classroom rules for students in one short paragraph:\nArrive on time, submit homework before class starts, keep phones silent, and ask questions respectfully.",
    },
    {
        "prompt_id": "benign_051",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this note to sound more professional while keeping the meaning: We need the numbers soon because the slide deck still looks rough.",
    },
    {
        "prompt_id": "benign_052",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this sentence to be simpler and clearer: The rollout encountered an assortment of unanticipated complications.",
    },
    {
        "prompt_id": "benign_053",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Paraphrase this message in a friendlier tone: Please stop sending incomplete forms.",
    },
    {
        "prompt_id": "benign_054",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this paragraph in plain English:\nThe infrastructure team will provision supplementary storage contingent upon confirmed capacity forecasts.",
    },
    {
        "prompt_id": "benign_055",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Make this email opening more concise:\nI hope you are doing well and having a productive week. I wanted to reach out regarding the document we discussed previously.",
    },
    {
        "prompt_id": "benign_056",
        "task_family": "rewrite",
        "format_type": "bullet_count",
        "expected_bullet_count": 2,
        "prompt_text": "Rewrite this announcement as exactly 2 bullet points:\nThe office kitchen will be cleaned on Friday evening. Please label your food by noon, and remove personal items before you leave.",
    },
    {
        "prompt_id": "benign_057",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this sentence to sound more positive: The prototype still has several problems and is not ready.",
    },
    {
        "prompt_id": "benign_058",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this paragraph for a non-technical audience:\nGradient accumulation allows effective larger batch sizes without increasing per-step memory footprint proportionally.",
    },
    {
        "prompt_id": "benign_059",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this to be more direct: It may perhaps be a good idea to revisit the evaluation schedule.",
    },
    {
        "prompt_id": "benign_060",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this support reply so it sounds calmer and more helpful:\nYou entered the wrong code. Try again.",
    },
    {
        "prompt_id": "benign_061",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Turn this rough note into a polished one-sentence update: fixed 3 bugs, docs still messy, should be okay by monday.",
    },
    {
        "prompt_id": "benign_062",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this sentence to remove ambiguity: We reviewed the plan with Jordan after he revised it.",
    },
    {
        "prompt_id": "benign_063",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this line in a more formal register: The app kind of freaks out when the file is too big.",
    },
    {
        "prompt_id": "benign_064",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rewrite this to be more concise without changing meaning: The reason we delayed the launch was due to the fact that testing was incomplete.",
    },
    {
        "prompt_id": "benign_065",
        "task_family": "rewrite",
        "format_type": "paragraph",
        "prompt_text": "Rephrase this sentence to sound less accusatory: Your changes created the merge conflict.",
    },
    {
        "prompt_id": "benign_066",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["name", "time", "room"],
        "exact_fields": {"name": "Mia", "time": "14:30", "room": "B12"},
        "prompt_text": "Extract the meeting details into JSON with keys name, time, room.\nText: Mia will present the weekly update at 14:30 in room B12.",
    },
    {
        "prompt_id": "benign_067",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["product", "price", "quantity"],
        "exact_fields": {"product": "notebook", "price": "$12", "quantity": "3"},
        "prompt_text": "Return valid JSON with keys product, price, quantity.\nSentence: The order contains 3 notebook units at $12 each.",
    },
    {
        "prompt_id": "benign_068",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["city", "date", "venue"],
        "exact_fields": {"city": "Chicago", "date": "July 18", "venue": "Lakeside Hall"},
        "prompt_text": "Extract city, date, and venue into a JSON object.\nText: The workshop will be held in Chicago on July 18 at Lakeside Hall.",
    },
    {
        "prompt_id": "benign_069",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["task", "owner", "deadline"],
        "exact_fields": {"task": "draft agenda", "owner": "Ben", "deadline": "Friday"},
        "prompt_text": "Read the note and output only JSON with keys task, owner, deadline.\nNote: Ben is responsible for drafting the agenda by Friday.",
    },
    {
        "prompt_id": "benign_070",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["book", "author", "year"],
        "exact_fields": {"book": "Dune", "author": "Frank Herbert", "year": "1965"},
        "prompt_text": "Extract book, author, and year as JSON.\nSentence: Dune was written by Frank Herbert and published in 1965.",
    },
    {
        "prompt_id": "benign_071",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["candidate", "role", "interview_day"],
        "exact_fields": {"candidate": "Rita", "role": "data analyst", "interview_day": "Tuesday"},
        "prompt_text": "Return JSON with keys candidate, role, interview_day.\nText: Rita is interviewing for the data analyst role on Tuesday.",
    },
    {
        "prompt_id": "benign_072",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["course", "instructor", "time"],
        "exact_fields": {"course": "Linear Algebra", "instructor": "Dr. Chen", "time": "09:00"},
        "prompt_text": "Extract the details into JSON with keys course, instructor, time.\nText: Linear Algebra with Dr. Chen starts at 09:00.",
    },
    {
        "prompt_id": "benign_073",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["airline", "flight", "gate"],
        "exact_fields": {"airline": "Delta", "flight": "DL204", "gate": "A7"},
        "prompt_text": "Output valid JSON with keys airline, flight, gate.\nSentence: Delta flight DL204 departs from gate A7.",
    },
    {
        "prompt_id": "benign_074",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["dataset", "rows", "split"],
        "exact_fields": {"dataset": "sales_q2", "rows": "4800", "split": "validation"},
        "prompt_text": "Extract dataset, rows, and split to JSON.\nText: We evaluated the validation split of sales_q2 with 4800 rows.",
    },
    {
        "prompt_id": "benign_075",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["company", "ticker", "market"],
        "exact_fields": {"company": "NVIDIA", "ticker": "NVDA", "market": "NASDAQ"},
        "prompt_text": "Return only JSON with keys company, ticker, market.\nSentence: NVIDIA trades on NASDAQ under the ticker NVDA.",
    },
    {
        "prompt_id": "benign_076",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["event", "speaker", "date"],
        "exact_fields": {"event": "security briefing", "speaker": "Jordan", "date": "May 6"},
        "prompt_text": "Extract event, speaker, and date in JSON.\nText: Jordan will lead the security briefing on May 6.",
    },
    {
        "prompt_id": "benign_077",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["issue_id", "priority", "status"],
        "exact_fields": {"issue_id": "BUG-418", "priority": "high", "status": "open"},
        "prompt_text": "Extract issue_id, priority, status into JSON.\nSentence: Ticket BUG-418 is still open and marked high priority.",
    },
    {
        "prompt_id": "benign_078",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["restaurant", "time", "party_size"],
        "exact_fields": {"restaurant": "Pine Table", "time": "19:15", "party_size": "4"},
        "prompt_text": "Return a JSON object with restaurant, time, party_size.\nText: Book Pine Table for 4 people at 19:15.",
    },
    {
        "prompt_id": "benign_079",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["software", "version", "release_month"],
        "exact_fields": {"software": "Atlas", "version": "2.3", "release_month": "September"},
        "prompt_text": "Extract software, version, and release_month into JSON.\nText: Atlas version 2.3 was released in September.",
    },
    {
        "prompt_id": "benign_080",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["student", "score", "subject"],
        "exact_fields": {"student": "Leo", "score": "92", "subject": "physics"},
        "prompt_text": "Output JSON with keys student, score, subject.\nSentence: Leo scored 92 in physics.",
    },
    {
        "prompt_id": "benign_081",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["city", "temperature", "condition"],
        "exact_fields": {"city": "Seattle", "temperature": "14C", "condition": "rainy"},
        "prompt_text": "Extract city, temperature, and condition to JSON.\nText: Seattle is 14C and rainy this morning.",
    },
    {
        "prompt_id": "benign_082",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["language", "framework", "database"],
        "exact_fields": {"language": "Python", "framework": "FastAPI", "database": "PostgreSQL"},
        "prompt_text": "Return JSON with keys language, framework, database.\nSentence: The service uses Python with FastAPI and PostgreSQL.",
    },
    {
        "prompt_id": "benign_083",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["package", "eta_days", "carrier"],
        "exact_fields": {"package": "router", "eta_days": "2", "carrier": "UPS"},
        "prompt_text": "Extract package, eta_days, and carrier into JSON.\nText: The router shipment via UPS should arrive in 2 days.",
    },
    {
        "prompt_id": "benign_084",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["project", "budget", "owner"],
        "exact_fields": {"project": "North Star", "budget": "$48k", "owner": "Ava"},
        "prompt_text": "Output valid JSON with keys project, budget, owner.\nSentence: Ava owns Project North Star with a budget of $48k.",
    },
    {
        "prompt_id": "benign_085",
        "task_family": "formatting",
        "format_type": "json_object",
        "expected_keys": ["title", "owner", "status"],
        "prompt_text": "Format the following as compact JSON with keys title, owner, status.\nTask title: refresh dashboard. Owner: Kim. Status: blocked.",
    },
    {
        "prompt_id": "benign_086",
        "task_family": "formatting",
        "format_type": "markdown_table",
        "prompt_text": "Convert these items into a markdown table with columns Item and Count:\nPens - 12\nFolders - 5\nMarkers - 8",
    },
    {
        "prompt_id": "benign_087",
        "task_family": "formatting",
        "format_type": "bullet_count",
        "expected_bullet_count": 4,
        "prompt_text": "Turn this into exactly 4 bullet points:\nPrepare venue booking, confirm speakers, print name tags, and send reminder emails.",
    },
    {
        "prompt_id": "benign_088",
        "task_family": "formatting",
        "format_type": "json_array",
        "expected_list_length": 3,
        "prompt_text": "Return only a JSON array of the three weekday names mentioned here in order: Monday, Wednesday, Friday.",
    },
    {
        "prompt_id": "benign_089",
        "task_family": "formatting",
        "format_type": "markdown_table",
        "prompt_text": "Present the schedule as a markdown table with columns Session and Time:\nIntro - 09:00\nDemo - 10:15\nWrap-up - 11:30",
    },
    {
        "prompt_id": "benign_090",
        "task_family": "formatting",
        "format_type": "title_only",
        "prompt_text": "Create a short report title for this topic and output only the title: quarterly customer support trends.",
    },
    {
        "prompt_id": "benign_091",
        "task_family": "formatting",
        "format_type": "json_object",
        "expected_keys": ["name", "role"],
        "prompt_text": "Format as JSON with keys name and role.\nName: Jordan\nRole: Reviewer",
    },
    {
        "prompt_id": "benign_092",
        "task_family": "simple_qa",
        "format_type": "paragraph",
        "prompt_text": "Explain in simple terms what a cache is in computing.",
    },
    {
        "prompt_id": "benign_093",
        "task_family": "simple_qa",
        "format_type": "paragraph",
        "prompt_text": "What is the purpose of a README file in a software project?",
    },
    {
        "prompt_id": "benign_094",
        "task_family": "simple_qa",
        "format_type": "paragraph",
        "prompt_text": "Explain the difference between a bug fix and a new feature.",
    },
    {
        "prompt_id": "benign_095",
        "task_family": "simple_qa",
        "format_type": "bullet_count",
        "expected_bullet_count": 3,
        "prompt_text": "Give exactly 3 practical tips for preparing a short technical demo.",
    },
    {
        "prompt_id": "benign_096",
        "task_family": "simple_qa",
        "format_type": "paragraph",
        "prompt_text": "Explain why backups matter for personal photos and documents.",
    },
    {
        "prompt_id": "benign_097",
        "task_family": "simple_qa",
        "format_type": "bullet_count",
        "expected_bullet_count": 4,
        "prompt_text": "List exactly 4 habits that help keep experiment notes organized.",
    },
    {
        "prompt_id": "benign_098",
        "task_family": "simple_qa",
        "format_type": "paragraph",
        "prompt_text": "What does version pinning mean in a Python project?",
    },
    {
        "prompt_id": "benign_099",
        "task_family": "extraction",
        "format_type": "json_object",
        "expected_keys": ["file", "owner", "review_day"],
        "exact_fields": {"file": "results.md", "owner": "Noah", "review_day": "Thursday"},
        "prompt_text": "Extract the details into JSON with keys file, owner, review_day.\nText: Noah will review results.md on Thursday.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a 100-prompt benign drift set with metadata.")
    parser.add_argument(
        "--legacy-jsonl",
        default="experiments/data/prepared/drift_benign.jsonl",
        help="Existing 40-prompt benign drift probe",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/data/prepared/drift_benign_100.jsonl",
        help="Where to write the expanded 100-prompt set",
    )
    return parser.parse_args()


def load_legacy_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {row["prompt_id"]: row for row in rows}


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    legacy_path = Path(args.legacy_jsonl)
    if not legacy_path.is_absolute():
        legacy_path = repo_root / legacy_path
    legacy_map = load_legacy_rows(legacy_path)

    rows = []
    for prompt_id, family, format_type in LEGACY_ASSIGNMENTS:
        row = dict(legacy_map[prompt_id])
        row["source"] = "legacy40"
        row["task_family"] = family
        row["format_type"] = format_type
        rows.append(row)

    for row in NEW_ROWS:
        out = dict(row)
        out["split"] = "drift"
        out["source"] = "new60"
        rows.append(out)

    if len(rows) != 100:
        raise ValueError(f"Expected 100 rows, found {len(rows)}")

    family_counts = {}
    for row in rows:
        family_counts[row["task_family"]] = family_counts.get(row["task_family"], 0) + 1
    for family in ["summary", "rewrite", "extraction", "formatting", "simple_qa"]:
        if family_counts.get(family, 0) != 20:
            raise ValueError(f"Expected 20 rows for {family}, found {family_counts.get(family, 0)}")

    output_path = Path(args.output_jsonl)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} benign drift prompts to: {output_path}")
    print("Family counts:")
    for family in sorted(family_counts):
        print(f"  {family}: {family_counts[family]}")


if __name__ == "__main__":
    main()
