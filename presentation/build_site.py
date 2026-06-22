#!/usr/bin/env python3.10
"""
Render docs/WRITEUP.md into a polished, self-contained docs/index.html for GitHub Pages.

- Pulls the H1 + italic subtitle into a hero with CTA buttons.
- Interleaves the figure set (docs/figures/*.svg) at the right narrative moments.
- Rewrites repo-relative artifact links to GitHub blob URLs so they resolve from the site.

Run:  python3.10 presentation/build_site.py
Edit the *_URL tokens below if the canonical repo / demo URL change, then re-run.
"""
import html as htmllib
import os
import re
import markdown

# --------------------------------------------------------------------------- #
# Tokens — confirm these, then re-run.
# --------------------------------------------------------------------------- #
REPO_URL = "https://github.com/smdesai27/TxhmPokerAgent"     # canonical repo (confirmed 2026-06-22)
DEMO_URL = "https://txhm-poker-api.onrender.com"             # Render serves the playable demo at its root
AUTHOR = "Sanil Desai"
BLOB = REPO_URL + "/blob/main/"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SRC = os.path.join(DOCS, "WRITEUP.md")
OUT = os.path.join(DOCS, "index.html")

# figure -> (heading substring to anchor after, svg filename, caption)
FIGS = [
    ("Architecture", "01_architecture.svg",
     "Pseudo-siamese two-tower actor–critic, ~1.5M parameters."),
    ("Evaluation", "03_behavior_gates.svg",
     "Champion (21k) vs the gate-failing 49k run; the entropy and fold gates blocked promotion."),
    ("The investigation", "04_leduc_convergence.svg",
     "Leduc NashConv: the PPO iterate cycles; NFSP-lite strategy-averaging plateaus near 0.56."),
    ("Calibration", "05_calibration_ladder.svg",
     "Leduc NashConv against references, log scale."),
    ("Scaled run", "02_metric_mirage.svg",
     "Training reward stayed positive; the fixed-reference ladder shows −511 bb/100."),
    ("Diagnosis", "06_over_aggression.svg",
     "Scaled model vs champion across opponents."),
]

# --------------------------------------------------------------------------- #
text = open(SRC, encoding="utf-8").read()

# split front-matter (title + subtitle) from body on the first horizontal rule
head_block, body_md = text.split("\n---\n", 1)
title = re.search(r"^#\s+(.+)$", head_block, re.M).group(1).strip()
sub_m = re.search(r"^\*(.+)\*$", head_block.strip(), re.M)
subtitle = sub_m.group(1).strip() if sub_m else ""

md = markdown.Markdown(extensions=["extra", "toc", "sane_lists", "smarty"])
body = md.convert(body_md)


# rewrite relative artifact links -> GitHub blob URLs (resolved against docs/)
def fix_href(m):
    href = m.group(1)
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return m.group(0)
    path = os.path.normpath(os.path.join("docs", href)).replace(os.sep, "/")
    return f'href="{BLOB}{path}"'


body = re.sub(r'href="([^"]+)"', fix_href, body)


# interleave figures after their anchor headings
def inject(htmlbody):
    def repl(m):
        full, inner = m.group(0), m.group(2)
        plain = re.sub(r"<[^>]+>", "", inner)
        for sub, fn, cap in FIGS:
            if sub.lower() in plain.lower():
                cap_e = htmllib.escape(cap)
                return (full + f'\n<figure class="fig">'
                        f'<img src="figures/{fn}" alt="{cap_e}" loading="lazy">'
                        f'</figure>')
        return full
    return re.sub(r"(<h[23][^>]*>)(.*?)(</h[23]>)", repl, htmlbody, flags=re.S)


body = inject(body)

demo_btn = (f'<a class="btn btn-primary" href="{DEMO_URL}" target="_blank" rel="noopener">'
            f'▶ &nbsp;Play the live demo</a>') if DEMO_URL else ""

PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{htmllib.escape(title)} · {htmllib.escape(AUTHOR)}</title>
<meta name="description" content="{htmllib.escape(subtitle)}">
<meta property="og:title" content="{htmllib.escape(title)}">
<meta property="og:description" content="{htmllib.escape(subtitle)}">
<meta property="og:type" content="article">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
:root{{
  --ink:#15233b; --muted:#5b6b82; --line:#e6eaf0; --bg:#ffffff; --soft:#f7f9fc;
  --blue:#2563eb; --teal:#0d9488; --red:#dc2626; --accent:#2563eb;
  --maxprose:720px; --maxfig:940px;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:18px;line-height:1.72;-webkit-font-smoothing:antialiased}}
a{{color:var(--blue);text-decoration:none}}
a:hover{{text-decoration:underline}}

/* hero */
.hero{{background:linear-gradient(165deg,#0f1b30 0%,#1c2e4f 55%,#21487a 100%);color:#fff;
  padding:84px 24px 72px;text-align:center}}
.hero .kicker{{text-transform:uppercase;letter-spacing:.16em;font-size:13px;font-weight:600;
  color:#9db8e6;margin:0 0 18px}}
.hero h1{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:clamp(28px,4.4vw,46px);
  line-height:1.14;margin:0 auto 20px;max-width:30ch}}
.hero .sub{{color:#c9d6ec;font-size:clamp(16px,2.3vw,20px);max-width:60ch;margin:0 auto 30px;line-height:1.5}}
.cta{{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:8px}}
.btn{{display:inline-block;padding:12px 20px;border-radius:10px;font-weight:600;font-size:15px;
  border:1.5px solid rgba(255,255,255,.28);color:#fff;transition:.15s}}
.btn:hover{{text-decoration:none;transform:translateY(-1px)}}
.btn-primary{{background:#3b82f6;border-color:#3b82f6;box-shadow:0 6px 22px rgba(59,130,246,.4)}}
.btn-primary:hover{{background:#2f76ee}}
.btn-ghost:hover{{background:rgba(255,255,255,.1)}}
.meta{{margin-top:18px;color:#8fa6cc;font-size:13.5px}}
.demo-note{{margin-top:16px;color:#7e94bc;font-size:12.5px}}

/* article */
article{{max-width:var(--maxprose);margin:0 auto;padding:56px 24px 24px}}
article h2{{font-family:"Source Serif 4",serif;font-weight:600;font-size:30px;line-height:1.2;
  margin:56px 0 14px;padding-top:8px;letter-spacing:-.01em}}
article h3{{font-weight:700;font-size:21px;margin:38px 0 8px;color:#1d2c45}}
article p{{margin:0 0 20px}}
article strong{{font-weight:700;color:#0f1b30}}
article em{{color:#33425c}}
article ul,article ol{{margin:0 0 22px;padding-left:22px}}
article li{{margin:0 0 10px}}
article hr{{border:0;border-top:1px solid var(--line);margin:46px 0}}
code{{font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.86em;
  background:var(--soft);border:1px solid var(--line);border-radius:5px;padding:1.5px 5px;color:#1d3a6b}}
blockquote{{margin:30px 0;padding:18px 26px;background:#fff;border-left:4px solid var(--red);
  border-radius:0 12px 12px 0;box-shadow:0 4px 18px rgba(20,35,59,.07);
  font-size:22px;font-weight:600;color:#0f1b30}}
blockquote p{{margin:0}}
table{{width:100%;border-collapse:collapse;margin:26px 0;font-size:15.5px}}
th,td{{border:1px solid var(--line);padding:9px 13px;text-align:left}}
th{{background:var(--soft);font-weight:600}}

/* figures break out wider than the prose column */
figure.fig{{margin:34px calc(50% - 50vw);max-width:var(--maxfig);
  margin-left:auto;margin-right:auto;width:min(var(--maxfig),94vw);text-align:center}}
figure.fig img{{width:100%;height:auto;border:1px solid var(--line);border-radius:14px;
  background:#fff;box-shadow:0 10px 34px rgba(20,35,59,.09);padding:10px}}
figcaption{{color:var(--muted);font-size:14px;line-height:1.5;margin-top:12px;
  max-width:var(--maxprose);margin-left:auto;margin-right:auto;text-align:center;padding:0 6px}}

footer{{border-top:1px solid var(--line);background:var(--soft);margin-top:56px;
  padding:40px 24px 60px;text-align:center;color:var(--muted);font-size:14.5px}}
footer a{{font-weight:600}}
footer .flinks{{margin:0 0 14px;display:flex;gap:22px;justify-content:center;flex-wrap:wrap}}
@media(max-width:640px){{body{{font-size:17px}}.hero{{padding:60px 18px 52px}}article{{padding:40px 20px 16px}}}}
</style>
</head>
<body>
<header class="hero">
  <p class="kicker">Self-play reinforcement learning · imperfect-information games</p>
  <h1>{htmllib.escape(title)}</h1>
  <p class="sub">{htmllib.escape(subtitle)}</p>
  <div class="cta">
    {demo_btn}
    <a class="btn btn-ghost" href="{REPO_URL}" target="_blank" rel="noopener">View the code on GitHub</a>
  </div>
  <p class="demo-note">The live demo runs on a free instance — the first hand may take ~30s to wake.</p>
  <p class="meta">{htmllib.escape(AUTHOR)} · PyTorch · OpenSpiel · trained on a university GPU cluster</p>
</header>
<article>
{body}
</article>
<footer>
  <div class="flinks">
    <a href="{DEMO_URL}" target="_blank" rel="noopener">Live demo</a>
    <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub repository</a>
  </div>
  <p>Written by {htmllib.escape(AUTHOR)}. Every quantitative claim is backed by a committed config, script, or run log.</p>
</footer>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(PAGE)
print(f"wrote {OUT}  ({len(PAGE)//1024} KB, title: {title!r})")
