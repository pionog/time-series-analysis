"""Generuje załącznik LaTeX z listą prac z input_csv/SLR.bib."""
from __future__ import annotations

import re
from pathlib import Path

BIB_PATH = Path("input_csv/SLR.bib")
OUT_PATH = Path("zalacznik_slr.tex")


def unescape_bibtex(s: str) -> str:
    s = s.replace(r"\&", "&").replace(r"\%", "%")
    s = re.sub(r"\{\\['\"`^~.=]([A-Za-z])\}", r"\1", s)
    s = re.sub(r"\\['\"`^~.=]\{([A-Za-z])\}", r"\1", s)
    s = s.replace(r"{\AA}", "Å").replace(r"{\aa}", "å")
    s = s.replace(r"{\O}", "Ø").replace(r"{\o}", "ø")
    s = s.replace(r"{\AE}", "Æ").replace(r"{\ae}", "æ")
    # usuń zbędne nawiasy ochronne BibTeX: {Word} -> Word (bez zagnieżdżeń złożonych)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\{([^{}]+)\}", r"\1", s)
    return s


def latex_escape(s: str) -> str:
    s = unescape_bibtex(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def format_authors(author: str) -> str:
    author = unescape_bibtex(author)
    parts = [p.strip() for p in re.split(r"\s+and\s+", author, flags=re.I) if p.strip()]
    formatted = []
    for p in parts:
        if "," in p:
            last, first = [x.strip() for x in p.split(",", 1)]
            # inicjały imion
            initials = []
            for token in re.split(r"[\s\-]+", first):
                token = token.strip()
                if not token:
                    continue
                initials.append(token[0] + ".")
            name = (" ".join(initials) + " " + last).strip()
        else:
            tokens = p.split()
            if len(tokens) == 1:
                name = tokens[0]
            else:
                initials = " ".join(t[0] + "." for t in tokens[:-1])
                name = f"{initials} {tokens[-1]}"
        formatted.append(latex_escape(name))
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def extract_balanced(s: str, start: int) -> tuple[str, int]:
    """Zakładając s[start]=='{', zwraca zawartość i indeks po zamykającym '}'."""
    assert s[start] == "{"
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
        i += 1
    return s[start + 1 :], len(s)


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(body)
    while i < n:
        m = re.match(r"\s*(\w+)\s*=\s*", body[i:])
        if not m:
            i += 1
            continue
        name = m.group(1).lower()
        i += m.end()
        if i >= n:
            break
        if body[i] == "{":
            val, i = extract_balanced(body, i)
        elif body[i] == '"':
            j = i + 1
            while j < n and body[j] != '"':
                j += 1
            val = body[i + 1 : j]
            i = j + 1
        else:
            j = i
            while j < n and body[j] not in ",\n":
                j += 1
            val = body[i:j].strip()
            i = j
        fields[name] = re.sub(r"\s+", " ", val).strip()
        # pomiń przecinek
        while i < n and body[i] in " \t\r\n,":
            i += 1
    return fields


def parse_bib(text: str) -> list[tuple[int, str, str, dict[str, str]]]:
    chunks = re.split(r"(?m)^%ID\s+(\d+)\s*$", text)
    entries: list[tuple[int, str, str, dict[str, str]]] = []
    for i in range(1, len(chunks), 2):
        sid = int(chunks[i])
        body = chunks[i + 1]
        m = re.search(r"@(\w+)\s*\{([^,]+),", body)
        if not m:
            raise ValueError(f"Brak wpisu BibTeX dla ID {sid}")
        etype = m.group(1).lower()
        key = m.group(2).strip()
        # odciąć zamykającą klamrę wpisu — bierzemy od po key do końca chunka
        rest = body[m.end() :]
        fields = parse_fields(rest)
        entries.append((sid, etype, key, fields))
    return entries


def venue(fields: dict[str, str], etype: str) -> str:
    for k in ("journal", "booktitle", "proceedings", "series", "publisher"):
        if fields.get(k):
            return latex_escape(fields[k])
    if etype == "phdthesis":
        return "PhD thesis"
    if etype == "mastersthesis":
        return "Master's thesis"
    if etype in {"techreport", "unpublished", "misc"}:
        return latex_escape(fields.get("howpublished") or fields.get("note") or "Technical report")
    return ""


def format_item(sid: int, etype: str, fields: dict[str, str]) -> str:
    authors = format_authors(fields.get("author", "Anonim"))
    title = latex_escape(fields.get("title", "[brak tytułu]"))
    year = latex_escape(fields.get("year", "n.d."))
    ven = venue(fields, etype)
    vol = fields.get("volume", "").strip()
    num = fields.get("number", "").strip()
    pages = fields.get("pages", "").strip().replace("--", "–").replace("-", "–")
    doi = fields.get("doi", "").strip()

    bits = [f"{authors}, \\textit{{{title}}}"]
    if ven:
        bits.append(ven)
    meta = []
    if vol:
        meta.append(f"vol.~{latex_escape(vol)}")
    if num:
        meta.append(f"no.~{latex_escape(num)}")
    if pages:
        meta.append(f"pp.~{latex_escape(pages)}")
    if meta:
        bits.append(", ".join(meta))
    bits.append(year)
    line = ", ".join(bits) + "."
    if doi:
        line += f" DOI: \\texttt{{{latex_escape(doi)}}}."
    # komentarz z oryginalnym ID SLR (przydatne przy śledzeniu)
    return f"    \\item % SLR ID {sid}\n    {line}"


def main() -> None:
    text = BIB_PATH.read_text(encoding="utf-8")
    entries = parse_bib(text)
    # kolejność jak w pliku (%ID)
    items = [format_item(sid, etype, fields) for sid, etype, _key, fields in entries]

    header = r"""\appendix
\chapter{Wykaz prac włączonych do przeglądu literatury (SLR)}
\label{chap:zalacznik_slr}

W niniejszym załączniku przedstawiono pełną listę prac naukowych zakwalifikowanych
do badania w ramach Systematycznego Przeglądu Literatury (SLR). Numeracja pozycji
w formacie [S\textit{n}] jest lokalna dla tego załącznika i nie zastępuje cytowań
z głównej bibliografii pracy.

\begin{enumerate}[label={[S\arabic*]}]
"""
    footer = r"""
\end{enumerate}
"""
    OUT_PATH.write_text(header + "\n".join(items) + footer, encoding="utf-8")
    print(f"Zapisano {OUT_PATH} ({len(items)} pozycji)")


if __name__ == "__main__":
    main()
