#!/usr/bin/env python3
"""
Exporte les rapports Markdown SecurAI en HTML et PDF.

Usage :
  python export_rapport.py membre2          # rapport individuel M2
  python export_rapport.py global           # rapport global équipe
  python export_rapport.py all              # les deux
  python export_rapport.py membre2 --html   # HTML seulement (pas de PDF)

Prérequis : pip install markdown
PDF : Microsoft Edge ou Google Chrome (mode headless)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "-q"])
    import markdown

ROOT = Path(__file__).resolve().parent

REPORTS = {
    "membre2": {
        "md": ROOT / "securai_report_membre2_traore.md",
        "html": ROOT / "securai_report_membre2_traore.html",
        "pdf": ROOT / "securai_report_membre2_traore.pdf",
        "title": "Rapport individuel - Attaques par évasion - SecurAI",
    },
    "global": {
        "md": ROOT / "securai_report_draft.md",
        "html": ROOT / "securai_report_global.html",
        "pdf": ROOT / "securai_report_global.pdf",
        "title": "Rapport technique - Attaques par évasion - SecurAI",
    },
}

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1a1a2e;
  margin: 0;
  padding: 0;
}
.content {
  border-left: 4px solid #00a8c9;
  padding: 0.5rem 0 0.5rem 1rem;
}
.rapport-entete {
  text-align: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #dde3ea;
  page-break-after: avoid;
}
.rapport-entete h1 { text-align: center; }
.rapport-entete p { text-align: center; margin: 0.35rem 0; }
.rapport-entete table { margin: 1rem auto 0 auto; max-width: 520px; text-align: left; }
h1 { font-size: 1.55rem; color: #0B0F19; margin: 0 0 0.75rem 0; }
h2 {
  font-size: 1.15rem;
  color: #0B0F19;
  border-bottom: 2px solid #00F0FF;
  padding-bottom: 0.2rem;
  margin-top: 1.6rem;
  page-break-after: avoid;
}
h3 { font-size: 1rem; color: #1565a8; margin-top: 1.1rem; page-break-after: avoid; }
p { margin: 0.5rem 0 0.75rem 0; text-align: justify; }
ul, ol { margin: 0.4rem 0 0.8rem 1.2rem; }
li { margin-bottom: 0.25rem; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 9.5pt;
  margin: 0.75rem 0 1rem 0;
  page-break-inside: avoid;
}
th, td { border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }
th { background: #eef8fb; color: #0B0F19; }
code, pre { font-family: Consolas, monospace; font-size: 8.5pt; background: #f4f6f8; }
pre {
  padding: 0.6rem 0.75rem;
  border-radius: 4px;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0.6rem auto 1rem auto;
  border: 1px solid #dde3ea;
  border-radius: 4px;
  page-break-inside: avoid;
}
img[alt="Logo EHTP"] {
  max-height: 90px;
  width: auto;
  border: none;
  margin: 0 auto 1.5rem auto;
  display: block;
}
.rapport-entete img[alt="Logo EHTP"] {
  margin-bottom: 1.75rem;
}
strong { color: #0B0F19; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.2rem 0; }
.thanks { text-align: center; margin: 2rem 0 1rem 0; font-size: 1.1rem; }
blockquote { color: #555; border-left: 3px solid #ccc; margin: 0.5rem 0; padding-left: 0.75rem; }
@media print {
  h2, h3 { page-break-after: avoid; }
  img { max-height: 220mm; object-fit: contain; }
}
"""


def strip_legacy_blocks(text: str) -> str:
    if text.startswith("# PROMPT DIRECTIVE"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1].lstrip()
    # Retire les anciens marqueurs d'images non remplis (rapport global)
    text = re.sub(r"^> \*\*\[📷 IMAGE[^\n]*\]\*\*\s*\n?", "", text, flags=re.MULTILINE)
    return text


def find_browser() -> Path:
    for p in EDGE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Edge ou Chrome introuvable. Installez Microsoft Edge ou exportez le HTML "
        "manuellement : Ctrl+P → Enregistrer au format PDF."
    )


def build_html(md_body: str, title: str) -> str:
    body = markdown.markdown(
        md_body,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    if "<h2>Remerciements</h2>" in body and "Merci de votre attention" not in body:
        body = body.replace(
            "<h2>Remerciements</h2>",
            '<h2>Remerciements</h2><p class="thanks"><strong>Merci de votre attention.</strong></p>',
        )
    body = re.sub(
        r'(<p class="thanks"><strong>Merci de votre attention\.</strong></p>\s*)'
        r'<p><strong>Merci de votre attention\.</strong></p>',
        r"\1",
        body,
    )
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
<div class="content">
{body}
</div>
</body>
</html>"""


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    browser = find_browser()
    url = html_path.resolve().as_uri()
    if pdf_path.exists():
        pdf_path.unlink()
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Échec export PDF:\n{result.stderr}")
    if not pdf_path.exists():
        raise RuntimeError("Le PDF n'a pas été généré.")


def export_one(key: str, html_only: bool = False) -> None:
    cfg = REPORTS[key]
    if not cfg["md"].exists():
        raise FileNotFoundError(f"Fichier introuvable : {cfg['md']}")

    md_clean = strip_legacy_blocks(cfg["md"].read_text(encoding="utf-8"))
    html = build_html(md_clean, cfg["title"])
    cfg["html"].write_text(html, encoding="utf-8")
    print(f"[{key}] HTML → {cfg['html']}")

    if html_only:
        print("       (PDF ignoré : option --html)")
        return

    try:
        export_pdf(cfg["html"], cfg["pdf"])
        print(f"[{key}] PDF  → {cfg['pdf']}")
    except FileNotFoundError as e:
        print(f"[{key}] PDF  → ignoré ({e})")
        print(f"       Ouvrez {cfg['html']} dans le navigateur, puis Ctrl+P → PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rapports SecurAI MD → HTML → PDF")
    parser.add_argument(
        "rapport",
        choices=["membre2", "global", "all"],
        help="Quel rapport exporter",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Générer uniquement le HTML (sans PDF)",
    )
    args = parser.parse_args()

    keys = list(REPORTS) if args.rapport == "all" else [args.rapport]
    for key in keys:
        export_one(key, html_only=args.html)


if __name__ == "__main__":
    main()
