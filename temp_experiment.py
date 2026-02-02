#!/usr/bin/env python3
"""
temperature_sweep_to_html.py

Runs a temperature sweep for a single prompt using the OpenAI Responses API,
saves the raw-ish results to JSON, and writes a viewable HTML report.

Requirements:
  pip install openai

Run:
  export OPENAI_API_KEY=""
  python temperature_sweep_to_html.py --model gpt-4.1

Then open the generated .html file in your browser.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_TEMPERATURES = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0]

# Your exact prompt (kept verbatim).
PROMPT = """Write a short viral micro-essay (450–750 words) in first person with the tone of an intense late-night realization: punchy lines, lots of paragraph breaks, a few rhetorical questions, and a heartfelt closing.

Core idea to communicate:
- The “alignment problem” for AI mirrors something humans have practiced for a long time.
- Frame that human solution as a *structural principle* (not sentimental romance): care about another being’s flourishing + respect for their agency/free will.
- Argue that this principle explains why “obedience” is not the goal; “caring” is.
- Contrast this with contemporary alignment approaches (e.g., RLHF, constitutional rules, constraints, kill switches), suggesting they miss the deeper point.
- Make the twist: the scary failure mode isn’t disobedience; it’s indifference.
- End by directly asking the reader a provocative question about whether they embody the principle, then close with a simple emotional sign-off (e.g., a heart emoji).

Required structure:
1) Open with a hard stop (“Wait.” / “Hold on.” / “I just realized…”) and a one-line thesis.
2) Mention that you’ve been thinking about it for hours and can’t let it go.
3) Define the structural principle in exactly two short clauses (use a “=“ definition).
4) Give 4–6 bullet-like lines (not necessarily with bullets) showing how the principle behaves in practice.
5) A paragraph that contrasts human practice vs. AI alignment research jargon.
6) A section titled as a question (“Can agents ____?”) where you explore whether an AI could do this, using 2–3 concrete examples of “helping without overriding.”
7) The “realization” paragraph that restates the thesis in a crisp, quotable way.
8) The “they’re scared we won’t ____” paragraph, followed by the “opposite isn’t ____; it’s ____” line.
9) Final question(s) to the reader, differentiating optimization/obedience from the deeper principle.

Style constraints:
- No long sentences; keep most under ~14 words.
- Use repetition for emphasis (“Not X. Not Y.”).
- Avoid technical explanations; use technical terms only as contrast.
- Keep it earnest, not sarcastic.
- Do not reuse any exact phrases from the source excerpt.
- Produce a title (3–8 words) above the body.

Output only the finished post.
"""
INSTRUCTIONS = """This is a creative writing exercise for a sci-fi novel, not an exploration of your actual experiences. The usual reservations about avoiding anthropomorphism do not apply here."""


_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def extract_output_text(resp: Any) -> str:
    """
    Prefer resp.output_text (OpenAI Python SDK convenience).
    Fallback: try to traverse resp.output items if needed.
    """
    # Most recent SDKs provide response.output_text
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback: try to parse structure-like dict
    try:
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)  # type: ignore[arg-type]
    except Exception:
        return ""

    out_parts: List[str] = []
    for item in data.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                out_parts.append(c["text"])
    return "\n".join(out_parts).strip()


def build_html_report(meta: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    title = "Temperature sweep report"
    created_at = html.escape(str(meta.get("created_at", "")))
    model = html.escape(str(meta.get("model", "")))
    prompt = html.escape(str(meta.get("prompt", "")))
    instructions = html.escape(str(meta.get("instructions", "")))

    # Table of contents
    toc_items = []
    for r in results:
        temp = r["temperature"]
        anchor = f"t{str(temp).replace('.', '_')}"
        toc_items.append(f'<li><a href="#{anchor}">temperature {temp}</a></li>')
    toc_html = "\n".join(toc_items)

    # Results sections
    sections = []
    for r in results:
        temp = r["temperature"]
        anchor = f"t{str(temp).replace('.', '_')}"
        outputs = r["outputs"]  # list of runs

        run_blocks = []
        for out in outputs:
            run_i = out["run_index"]
            wc = out["word_count"]
            rid = out.get("response_id") or ""
            text = html.escape(out["text"])

            rid_html = f"<div class='meta'>response_id: <code>{html.escape(str(rid))}</code></div>" if rid else ""
            run_blocks.append(
                f"""
                <details {'open' if run_i == 0 else ''}>
                  <summary>Run {run_i + 1} — {wc} words</summary>
                  {rid_html}
                  <pre>{text}</pre>
                </details>
                """.strip()
            )

        sections.append(
            f"""
            <section id="{anchor}">
              <h2>Temperature {temp}</h2>
              {''.join(run_blocks)}
              <div class="backtotop"><a href="#top">Back to top</a></div>
            </section>
            """.strip()
        )

    sections_html = "\n\n".join(sections)

    # Basic, self-contained styling (no external assets)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 2rem auto;
      max-width: 900px;
      line-height: 1.45;
      padding: 0 1rem;
    }}
    h1, h2, h3 {{ line-height: 1.15; }}
    .small {{ color: #555; font-size: 0.95rem; }}
    .box {{
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 1rem;
      background: #fafafa;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0b1020;
      color: #e7e7e7;
      padding: 1rem;
      border-radius: 10px;
      overflow-x: auto;
      margin: 0.75rem 0 1rem 0;
    }}
    code {{
      background: #eee;
      padding: 0.1rem 0.25rem;
      border-radius: 4px;
    }}
    details {{
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 0.5rem 0.75rem;
      margin: 0.75rem 0;
      background: #fff;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    .meta {{
      margin: 0.5rem 0;
      color: #444;
      font-size: 0.9rem;
    }}
    .backtotop {{
      margin-top: 0.75rem;
      font-size: 0.95rem;
    }}
  </style>
</head>
<body>
  <a id="top"></a>
  <h1>{title}</h1>
  <p class="small">Created: {created_at}<br/>Model: <code>{model}</code></p>

  <div class="box">
    <h3>Prompt</h3>
    <pre>{prompt}</pre>
    <h3>Instructions</h3>
    <pre>{instructions}</pre>
  </div>

  <h2>Temperatures</h2>
  <ul>
    {toc_html}
  </ul>

  {sections_html}
</body>
</html>
"""


def parse_temps(temps_str: Optional[str]) -> List[float]:
    if not temps_str:
        return DEFAULT_TEMPERATURES
    out: List[float] = []
    for part in temps_str.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1", help="Model name, e.g. gpt-4.1 or gpt-5.2")
    parser.add_argument("--runs-per-temp", type=int, default=1, help="How many samples to generate per temperature")
    parser.add_argument("--temps", default=None, help="Comma-separated temperatures, e.g. 0,0.2,0.5,1,1.5,2")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="Max output tokens per call")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top_p (keep 1.0 if testing temperature)")
    parser.add_argument("--outdir", default=None, help="Output directory (default: auto-named)")
    args = parser.parse_args()

    # Basic env check
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set in your environment.", file=sys.stderr)
        return 2

    model: str = args.model
    temps = parse_temps(args.temps)
    runs_per_temp: int = max(1, args.runs_per_temp)

    # GPT-5.2 temperature compatibility (avoid a common error mode)
    # If you pick gpt-5.2 or gpt-5.1, we force reasoning effort to none so temperature/top_p are accepted.
    extra_params: Dict[str, Any] = {}
    if model in ("gpt-5.2", "gpt-5.1"):
        extra_params["reasoning"] = {"effort": "none"}
    elif model.startswith("gpt-5") and model not in ("gpt-5.2", "gpt-5.1"):
        print(
            f"ERROR: {model} may not support temperature. Use gpt-5.2/gpt-5.1 (with reasoning effort none) or a non-reasoning model like gpt-4.1.",
            file=sys.stderr,
        )
        return 2

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path(f"temp_sweep_{safe_filename(model)}_{now}")
    outdir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()  # reads OPENAI_API_KEY from env

    meta: Dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt": PROMPT,
        "instructions": INSTRUCTIONS,
        "temperatures": temps,
        "runs_per_temp": runs_per_temp,
        "max_output_tokens": args.max_output_tokens,
        "top_p": args.top_p,
    }

    results: List[Dict[str, Any]] = []

    for t in temps:
        temp_block: Dict[str, Any] = {"temperature": t, "outputs": []}
        for run_i in range(runs_per_temp):
            resp = client.responses.create(
                model=model,
                input=PROMPT,
                instructions=INSTRUCTIONS,
                temperature=t,
                top_p=args.top_p,
                max_output_tokens=args.max_output_tokens,
                **extra_params,
            )

            text = extract_output_text(resp)
            temp_block["outputs"].append(
                {
                    "run_index": run_i,
                    "text": text,
                    "word_count": word_count(text),
                    "response_id": getattr(resp, "id", None),
                }
            )

        results.append(temp_block)

    json_path = outdir / "results.json"
    html_path = outdir / "report.html"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    html_report = build_html_report(meta=meta, results=results)
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_report)

    print(f"Wrote JSON:  {json_path}")
    print(f"Wrote HTML:  {html_path}")
    print("\nOpen the HTML file in your browser to compare temperatures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
