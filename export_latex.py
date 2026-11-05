"""
Automated Markdown-to-LaTeX (`.tex`) Manuscript Exporter for CUFG
=================================================================
Converts all 9 core Markdown manuscripts in the repository root into
self-contained, Overleaf- and arXiv-ready LaTeX (`.tex`) files inside `latex/`.

Usage:
    python -B export_latex.py
"""

from __future__ import annotations

import re
from pathlib import Path

MANUSCRIPTS = [
    ("cufg_summary_paper.md", "cufg_summary_paper.tex"),
    (
        "cufg_trilogy_part1_emergent_gravity_from_overlapping_uncertainty.md",
        "cufg_trilogy_part1_emergent_gravity.tex",
    ),
    (
        "cufg_trilogy_part2_galaxies_clusters_and_zero_gravity_cosmology.md",
        "cufg_trilogy_part2_galaxies_and_cosmology.tex",
    ),
    (
        "cufg_trilogy_part3_subatomic_and_nuclear_scale_foundation.md",
        "cufg_trilogy_part3_subatomic_and_nuclear.tex",
    ),
    (
        "cufg_paper1_macroscopic_uncertainty_dynamics.md",
        "cufg_paper1_macroscopic_uncertainty_dynamics.tex",
    ),
    (
        "cufg_paper2_subatomic_and_nuclear_hierarchy.md",
        "cufg_paper2_subatomic_and_nuclear_hierarchy.tex",
    ),
    (
        "cufg_theoretical_and_numerical_report.md",
        "cufg_theoretical_and_numerical_report.tex",
    ),
    (
        "causal_dependency_graph_and_predictions.md",
        "causal_dependency_graph_and_predictions.tex",
    ),
    (
        "cufg_cosmic_chronology_and_scifi_companion.md",
        "cufg_cosmic_chronology_and_scifi_companion.tex",
    ),
]

UNICODE_MATH_MAP = {
    "−": "-",
    "–": "--",
    "—": "---",
    "×": r"\times ",
    "·": r"\cdot ",
    "±": r"\pm ",
    "∓": r"\mp ",
    "≈": r"\approx ",
    "≃": r"\simeq ",
    "≅": r"\cong ",
    "≡": r"\equiv ",
    "≠": r"\neq ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≪": r"\ll ",
    "≫": r"\gg ",
    "→": r"\to ",
    "←": r"\leftarrow ",
    "↔": r"\leftrightarrow ",
    "⇒": r"\implies ",
    "⇔": r"\iff ",
    "↦": r"\mapsto ",
    "∝": r"\propto ",
    "∞": r"\infty ",
    "∂": r"\partial ",
    "∇": r"\nabla ",
    "√": r"\sqrt{}",
    "⊕": r"\oplus ",
    "⊗": r"\otimes ",
    "⊖": r"\ominus ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∀": r"\forall ",
    "∃": r"\exists ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "∫": r"\int ",
    "∮": r"\oint ",
    "°": r"^\circ ",
    "′": "'",
    "″": "''",
    "ħ": r"\hbar ",
    "⊙": r"\odot ",
    "⊥": r"\perp ",
    "∥": r"\parallel ",
    "⟨": r"\langle ",
    "⟩": r"\rangle ",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\varepsilon ",
    "ϵ": r"\epsilon ",
    "ζ": r"\zeta ",
    "η": r"\eta ",
    "θ": r"\theta ",
    "ϑ": r"\vartheta ",
    "ι": r"\iota ",
    "κ": r"\kappa ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "ν": r"\nu ",
    "ξ": r"\xi ",
    "π": r"\pi ",
    "ρ": r"\rho ",
    "σ": r"\sigma ",
    "ς": r"\varsigma ",
    "τ": r"\tau ",
    "υ": r"\upsilon ",
    "φ": r"\phi ",
    "ϕ": r"\varphi ",
    "χ": r"\chi ",
    "ψ": r"\psi ",
    "ω": r"\omega ",
    "Γ": r"\Gamma ",
    "Δ": r"\Delta ",
    "Θ": r"\Theta ",
    "Λ": r"\Lambda ",
    "Ξ": r"\Xi ",
    "Π": r"\Pi ",
    "Σ": r"\Sigma ",
    "Υ": r"\Upsilon ",
    "Φ": r"\Phi ",
    "Ψ": r"\Psi ",
    "Ω": r"\Omega ",
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "^+",
    "⁻": "^-",
    "ℓ": r"\ell ",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℤ": r"\mathbb{Z}",
    "ℕ": r"\mathbb{N}",
    "𝕀": r"\mathbb{I}",
}

UNICODE_TEXT_MAP = {
    "−": "-",
    "–": "--",
    "—": "---",
    "’": "'",
    "‘": "`",
    "“": "``",
    "”": "''",
    "•": r"\textbullet{}",
    "·": r"\textperiodcentered{}",
    "×": r"$\times$",
    "±": r"$\pm$",
    "∓": r"$\mp$",
    "≈": r"$\approx$",
    "≃": r"$\simeq$",
    "≡": r"$\equiv$",
    "≠": r"$\neq$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "≪": r"$\ll$",
    "≫": r"$\gg$",
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\implies$",
    "⇔": r"$\iff$",
    "∝": r"$\propto$",
    "∞": r"$\infty$",
    "∂": r"$\partial$",
    "∇": r"$\nabla$",
    "√": r"$\sqrt{}$",
    "⊕": r"$\oplus$",
    "⊗": r"$\otimes$",
    "∈": r"$\in$",
    "∀": r"$\forall$",
    "∑": r"$\sum$",
    "∏": r"$\prod$",
    "∫": r"$\int$",
    "∮": r"$\oint$",
    "°": r"$^\circ$",
    "′": "'",
    "″": "''",
    "ħ": r"$\hbar$",
    "⊙": r"$\odot$",
    "⊥": r"$\perp$",
    "∥": r"$\parallel$",
    "⟨": r"$\langle$",
    "⟩": r"$\rangle$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "γ": r"$\gamma$",
    "δ": r"$\delta$",
    "ε": r"$\varepsilon$",
    "ϵ": r"$\epsilon$",
    "ζ": r"$\zeta$",
    "η": r"$\eta$",
    "θ": r"$\theta$",
    "κ": r"$\kappa$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "ν": r"$\nu$",
    "ξ": r"$\xi$",
    "π": r"$\pi$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "τ": r"$\tau$",
    "υ": r"$\upsilon$",
    "φ": r"$\phi$",
    "χ": r"$\chi$",
    "ψ": r"$\psi$",
    "ω": r"$\omega$",
    "Γ": r"$\Gamma$",
    "Δ": r"$\Delta$",
    "Θ": r"$\Theta$",
    "Λ": r"$\Lambda$",
    "Ξ": r"$\Xi$",
    "Π": r"$\Pi$",
    "Σ": r"$\Sigma$",
    "Υ": r"$\Upsilon$",
    "Φ": r"$\Phi$",
    "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$",
    "₀": r"$_0$",
    "₁": r"$_1$",
    "₂": r"$_2$",
    "₃": r"$_3$",
    "₄": r"$_4$",
    "₅": r"$_5$",
    "₆": r"$_6$",
    "₇": r"$_7$",
    "₈": r"$_8$",
    "₉": r"$_9$",
    "⁰": r"$^0$",
    "¹": r"$^1$",
    "²": r"$^2$",
    "³": r"$^3$",
    "⁴": r"$^4$",
    "⁵": r"$^5$",
    "⁶": r"$^6$",
    "⁷": r"$^7$",
    "⁸": r"$^8$",
    "⁹": r"$^9$",
    "⁺": r"$^+$",
    "⁻": r"$^-$",
    "ℓ": r"$\ell$",
    "ℝ": r"$\mathbb{R}$",
    "ℂ": r"$\mathbb{C}$",
    "ℤ": r"$\mathbb{Z}$",
    "▼": r"$\blacktriangledown$",
    "▲": r"$\blacktriangle$",
    "★": r"$\star$",
}

UNICODE_ASCII_CODE_MAP = {
    "−": "-",
    "–": "-",
    "—": "--",
    "×": "x",
    "·": "*",
    "±": "+/-",
    "∓": "-/+",
    "≈": "~",
    "≃": "~",
    "≡": "==",
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    "≪": "<<",
    "≫": ">>",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    "⇔": "<=>",
    "∝": "~",
    "∞": "inf",
    "∂": "d",
    "∇": "grad",
    "√": "sqrt",
    "⊕": "(+)",
    "⊗": "(x)",
    "∈": "in",
    "∀": "forall",
    "∑": "sum",
    "∏": "prod",
    "∫": "int",
    "∮": "oint",
    "°": "deg",
    "′": "'",
    "″": "''",
    "ħ": "hbar",
    "⊙": "sun",
    "⊥": "perp",
    "∥": "par",
    "⟨": "<",
    "⟩": ">",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "eps",
    "ϵ": "eps",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "u",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ξ": "Xi",
    "Π": "Pi",
    "Σ": "Sigma",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
    "⁺": "+",
    "⁻": "-",
    "ℓ": "l",
    "ℝ": "R",
    "ℂ": "C",
    "ℤ": "Z",
    "▼": "v",
}


def _escape_latex_text(s: str) -> str:
    """Escape plain-text characters for LaTeX outside math/code tokens."""
    out = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch in UNICODE_TEXT_MAP:
            out.append(UNICODE_TEXT_MAP[ch])
        elif ord(ch) > 127:
            # Fallback for any other non-ASCII symbol (e.g., combining bar)
            if ord(ch) == 0x0304:
                continue
            out.append("")
        else:
            out.append(ch)
    return "".join(out)


def _escape_code_text(s: str) -> str:
    """Escape inline code strings for \\texttt{...}."""
    s = s.replace("p\u0304", "p_bar").replace("ν\u0304", "nu_bar").replace("t\u0304", "t_bar")
    out = []
    for ch in s:
        if ch in UNICODE_ASCII_CODE_MAP:
            mapped = UNICODE_ASCII_CODE_MAP[ch]
            for mch in mapped:
                out.append(_escape_latex_text(mch))
        elif ord(ch) > 127:
            continue
        else:
            out.append(_escape_latex_text(ch))
    return "".join(out)


def _clean_math_content(m: str) -> str:
    """Sanitize Unicode characters inside math mode."""
    out = []
    for ch in m:
        if ch in UNICODE_MATH_MAP:
            out.append(UNICODE_MATH_MAP[ch])
        elif ord(ch) > 127:
            if ord(ch) == 0x0304:
                continue
            out.append("")
        else:
            out.append(ch)
    cleaned = "".join(out)
    # Escape unescaped % or # inside \text{...}
    def _fix_text_arg(match: re.Match) -> str:
        inner = match.group(1)
        inner = re.sub(r"(?<!\\)%", r"\%", inner)
        inner = re.sub(r"(?<!\\)&", r"\&", inner)
        inner = re.sub(r"(?<!\\)#", r"\#", inner)
        return r"\text{" + inner + "}"

    cleaned = re.sub(r"\\text\{([^{}]*)\}", _fix_text_arg, cleaned)
    return cleaned


def _format_inline(text: str) -> str:
    """Convert inline Markdown (math, code, links, bold, italics) to LaTeX."""
    tokens: dict[str, str] = {}
    counter = 0

    def _stash(latex_str: str) -> str:
        nonlocal counter
        key = f"@@TOK{counter}@@"
        counter += 1
        tokens[key] = latex_str
        return key

    # 1. Protect inline code `...`
    def _repl_code(m: re.Match) -> str:
        code_str = m.group(1)
        return _stash(r"\texttt{" + _escape_code_text(code_str) + "}")

    text = re.sub(r"`([^`]+)`", _repl_code, text)

    # 2. Protect display math $$...$$ inside a line
    def _repl_dmath(m: re.Match) -> str:
        return _stash(r"\[" + _clean_math_content(m.group(1)) + r"\]")

    text = re.sub(r"\$\$(.+?)\$\$", _repl_dmath, text)

    # 3. Protect inline math $...$
    def _repl_imath(m: re.Match) -> str:
        return _stash("$" + _clean_math_content(m.group(1)) + "$")

    text = re.sub(r"(?<!\\)\$([^\$\n]+?)(?<!\\)\$", _repl_imath, text)

    # 4. Handle Markdown images ![alt](path)
    def _repl_img(m: re.Match) -> str:
        alt = _escape_latex_text(m.group(1))
        path = m.group(2).strip()
        if path.startswith("output/"):
            path = "../" + path
        return _stash(
            "\n\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=0.95\\textwidth]{{{path}}}\n"
            f"\\caption{{{alt}}}\n\\end{{figure}}\n"
        )

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _repl_img, text)

    # 5. Handle Markdown links [label](url) -> keep formatted label
    def _repl_link(m: re.Match) -> str:
        label = m.group(1)
        url = m.group(2).strip()
        formatted_label = _format_inline_plain(label, tokens)
        if url.startswith("http://") or url.startswith("https://"):
            safe_url = url.replace("%", r"\%").replace("#", r"\#")
            return _stash(f"\\href{{{safe_url}}}{{{formatted_label}}}")
        return _stash(formatted_label)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl_link, text)

    # 6. Escape plain text outside tokens, preserving **bold** and *italic*
    text = _format_inline_plain(text, tokens)

    # 7. Restore tokens (possibly nested)
    for _ in range(4):
        changed = False
        for k, v in tokens.items():
            if k in text:
                text = text.replace(k, v)
                changed = True
        if not changed:
            break
    return text


def _format_inline_plain(text: str, tokens: dict[str, str]) -> str:
    """Format bold/italics and escape remaining plain text outside @@TOK@@ markers."""
    # Split by @@TOK\d+@@ so we never alter token markers
    parts = re.split(r"(@@TOK\d+@@)", text)
    escaped_parts = []
    for p in parts:
        if p.startswith("@@TOK") and p.endswith("@@"):
            escaped_parts.append(p)
        else:
            escaped_parts.append(_escape_latex_text(p))
    joined = "".join(escaped_parts)

    # Bold **...**
    joined = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", joined)
    # Italic *...*
    joined = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\\textit{\1}", joined)
    return joined


def _convert_table(table_lines: list[str]) -> str:
    """Convert a Markdown table block into a wrapped LaTeX longtable."""
    rows: list[list[str]] = []
    for line in table_lines:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        # Split on unescaped pipes
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", s)]
        rows.append(cells)

    if len(rows) < 2:
        return ""

    header = rows[0]
    n_cols = len(header)
    # Check if row 1 is separator (---, :---, :---:)
    start_idx = 2 if all(re.match(r"^:?-{2,}:?$", c.replace(" ", "")) for c in rows[1] if c) else 1
    body_rows = rows[start_idx:]

    # Compute relative column widths based on average text length per column
    col_weights = []
    for c_idx in range(n_cols):
        max_len = len(header[c_idx]) if c_idx < len(header) else 10
        for r in body_rows:
            if c_idx < len(r):
                max_len = max(max_len, min(len(r[c_idx]), 90))
        col_weights.append(max(max_len, 8) ** 0.65)
    total_w = sum(col_weights) if sum(col_weights) > 0 else 1.0
    col_fracs = [max(0.08, w / total_w * 0.94) for w in col_weights]
    scale = 0.94 / sum(col_fracs)
    col_fracs = [f * scale for f in col_fracs]

    col_spec = " ".join(
        f">{{\\raggedright\\arraybackslash}}p{{{frac:.3f}\\textwidth}}" for frac in col_fracs
    )

    out = [
        r"\begingroup\small",
        r"\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}",
        f"\\begin{{longtable}}{{{col_spec}}}",
        r"\toprule",
    ]
    header_cells = [
        r"\textbf{" + _format_inline(header[i] if i < len(header) else "") + "}"
        for i in range(n_cols)
    ]
    out.append(" & ".join(header_cells) + r" \\")
    out.append(r"\midrule")
    out.append(r"\endhead")

    for r in body_rows:
        cells = [
            _format_inline(r[i] if i < len(r) else "")
            for i in range(n_cols)
        ]
        out.append(" & ".join(cells) + r" \\")
        out.append(r"\addlinespace[3pt]")

    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    out.append(r"\endgroup")
    return "\n".join(out)


def _convert_mermaid(lines: list[str]) -> str:
    """Convert a Mermaid diagram into a readable LaTeX flowchart summary box."""
    cleaned_lines = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("flowchart ") or s.startswith("graph ") or s == "end":
            continue
        s = s.replace("<br/>", " | ").replace("<br>", " | ")
        cleaned_lines.append("  " + _escape_code_text(s))
    body = "\n".join(cleaned_lines)
    return (
        "\n\\begin{quote}\\small\n"
        "\\noindent\\textbf{[Causal Flowchart / Architecture Diagram]}\n"
        "\\begin{verbatim}\n"
        f"{body}\n"
        "\\end{verbatim}\n"
        "\\end{quote}\n"
    )


def convert_markdown_to_latex(md_text: str) -> str:
    """Convert a complete CUFG Markdown manuscript into a standalone LaTeX document."""
    lines = md_text.splitlines()
    title = "Causal Uncertainty-Field Dynamics (CUFG)"
    start_line = 0
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            title = _format_inline(line[2:].strip())
            start_line = idx + 1
            break

    out_blocks: list[str] = []
    i = start_line
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            out_blocks.append("")
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}$", stripped):
            out_blocks.append(r"\bigskip\noindent\hrulefill\bigskip")
            i += 1
            continue

        # Fenced code block ```...```
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            i += 1
            block_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                block_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            if lang == "mermaid":
                out_blocks.append(_convert_mermaid(block_lines))
            else:
                safe_code = "\n".join(
                    "".join(UNICODE_ASCII_CODE_MAP.get(ch, ch if ord(ch) < 128 else "") for ch in bl)
                    for bl in block_lines
                )
                out_blocks.append(
                    "\\begin{verbatim}\n" + safe_code + "\n\\end{verbatim}"
                )
            continue

        # Multi-line display math $$...$$
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                math_inner = stripped[2:-2].strip()
                out_blocks.append("\\[\n" + _clean_math_content(math_inner) + "\n\\]")
                i += 1
                continue
            else:
                math_lines = [stripped[2:]]
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i])
                    i += 1
                if i < n:
                    math_lines.append(lines[i].strip()[:-2])
                    i += 1
                out_blocks.append(
                    "\\[\n" + _clean_math_content("\n".join(math_lines)) + "\n\\]"
                )
                continue

        # Markdown Table
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out_blocks.append(_convert_table(table_lines))
            continue

        # Blockquote / GitHub Alert (> [!NOTE], > ...)
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q = re.sub(r"^>\s?", "", lines[i].strip())
                q = re.sub(
                    r"^\[!(NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*",
                    r"**[\1]** ",
                    q,
                )
                quote_lines.append(_format_inline(q))
                i += 1
            out_blocks.append(
                "\\begin{quote}\n" + "\n\n".join(ql for ql in quote_lines if ql) + "\n\\end{quote}"
            )
            continue

        # Headings
        if stripped.startswith("#### "):
            out_blocks.append(r"\paragraph{" + _format_inline(stripped[5:].strip()) + "}")
            i += 1
            continue
        if stripped.startswith("### "):
            out_blocks.append(r"\subsection{" + _format_inline(stripped[4:].strip()) + "}")
            i += 1
            continue
        if stripped.startswith("## "):
            out_blocks.append(r"\section{" + _format_inline(stripped[3:].strip()) + "}")
            i += 1
            continue
        if stripped.startswith("# "):
            out_blocks.append(r"\section*{" + _format_inline(stripped[2:].strip()) + "}")
            i += 1
            continue

        # Lists (-, *, 1.)
        list_match = re.match(r"^(\s*)([-*\u2022]|\d+\.)\s+(.*)$", line)
        if list_match:
            list_items: list[tuple[str, str]] = []
            while i < n:
                lm = re.match(r"^(\s*)([-*\u2022]|\d+\.)\s+(.*)$", lines[i])
                if lm:
                    marker = "enum" if lm.group(2).endswith(".") else "item"
                    item_text = lm.group(3).strip()
                    i += 1
                    # Continuation lines of the same list item
                    while i < n:
                        nxt = lines[i]
                        nxt_s = nxt.strip()
                        if (
                            not nxt_s
                            or re.match(r"^(\s*)([-*\u2022]|\d+\.)\s+", nxt)
                            or nxt_s.startswith("#")
                            or nxt_s.startswith("|")
                            or nxt_s.startswith("```")
                            or nxt_s.startswith(">")
                            or nxt_s.startswith("---")
                        ):
                            break
                        if nxt_s.startswith("$$") and nxt_s.endswith("$$") and len(nxt_s) > 4:
                            item_text += "\n\\[\n" + _clean_math_content(nxt_s[2:-2]) + "\n\\]"
                        else:
                            item_text += " " + nxt_s
                        i += 1
                    list_items.append((marker, item_text))
                else:
                    break

            env = "enumerate" if list_items and list_items[0][0] == "enum" else "itemize"
            out_blocks.append(f"\\begin{{{env}}}")
            for _, it_txt in list_items:
                # Split out any embedded \[ ... \] blocks
                subparts = re.split(r"(\\\[[\s\S]*?\\\])", it_txt)
                rendered_subparts = [
                    sp if sp.startswith(r"\[") else _format_inline(sp) for sp in subparts
                ]
                out_blocks.append("  \\item " + "".join(rendered_subparts))
            out_blocks.append(f"\\end{{{env}}}")
            continue

        # Regular paragraph line
        out_blocks.append(_format_inline(stripped))
        i += 1

    preamble = rf"""% =============================================================================
% Causal Uncertainty-Field Dynamics (CUFG) — Auto-Generated LaTeX Manuscript
% Author: Jason Bales
% Repository: https://github.com/jasonbales1/Causal-Uncertainty-Field-Dynamics
% Note: Compiles with pdflatex or xelatex on Overleaf / TeX Live / MiKTeX.
%       For APS Physical Review D submission, replace \documentclass below with:
%       \documentclass[aps,prd,onecolumn,nofootinbib,superscriptaddress]{{revtex4-2}}
% =============================================================================
\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{amsmath,amssymb,amsfonts,mathtools,bm}}
\usepackage{{graphicx}}
\usepackage{{booktabs,longtable,array}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage[colorlinks=true,linkcolor=blue!70!black,citecolor=blue!70!black,urlcolor=blue!70!black]{{hyperref}}

\graphicspath{{{{../output/}}{{output/}}}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}
\allowdisplaybreaks

\title{{\textbf{{{title}}}}}
\author{{Jason Bales \\ \small\texttt{{https://github.com/jasonbales1/Causal-Uncertainty-Field-Dynamics}}}}
\date{{November 2026}}

\begin{{document}}
\maketitle
"""

    footer = "\n\\end{document}\n"
    return preamble + "\n".join(out_blocks) + footer


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    latex_dir = root_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for md_name, tex_name in MANUSCRIPTS:
        md_path = root_dir / md_name
        if not md_path.exists():
            continue
        md_text = md_path.read_text(encoding="utf-8")
        tex_content = convert_markdown_to_latex(md_text)
        tex_path = latex_dir / tex_name
        tex_path.write_text(tex_content, encoding="utf-8")
        exported.append((md_name, tex_path, len(tex_content)))

    print(f"Exported {len(exported)} LaTeX (.tex) manuscripts to {latex_dir}:")
    for md_name, tex_path, n_chars in exported:
        print(f"  - {md_name} -> latex/{tex_path.name} ({n_chars:,} chars)")


if __name__ == "__main__":
    main()
