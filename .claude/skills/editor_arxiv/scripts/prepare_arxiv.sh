#!/usr/bin/env bash
# prepare_arxiv.sh — prepares latex/ source for arXiv submission
#
# Sources:
#   https://info.arxiv.org/help/submit_tex.html
#   https://trevorcampbell.me/html/arxiv.html
#
# Run from repo root:
#   bash .claude/skills/editor_arxiv/scripts/prepare_arxiv.sh

set -euo pipefail

SRC="latex"
OUT="latex/arxiv_latex"
MAIN="dpitg"

echo "=== Preparing arXiv submission from $SRC/ → $OUT/ ==="

# If output directory already has content, archive it to _legacy/<timestamp>
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null | grep -v '^_legacy$')" ]]; then
    TIMESTAMP=$(date +"%Y-%m-%d--%H-%M")
    LEGACY="$OUT/_legacy/$TIMESTAMP"
    mkdir -p "$LEGACY"
    find "$OUT" -maxdepth 1 ! -name '_legacy' ! -path "$OUT" -exec mv {} "$LEGACY/" \;
    echo "  archived previous contents → $LEGACY"
fi
mkdir -p "$OUT/images"

# ---------------------------------------------------------------------------
# Helper: strip LaTeX comments and alt text lines
#   - removes comment-only lines and inline comments (preserving \%)
#   - removes "Alt text: ..." lines (RSS requires these in the manuscript;
#     arXiv would print them as visible body text)
# ---------------------------------------------------------------------------
strip_comments() {
    perl -ne '
        if (/^\s*Alt text:/) { $in_alt = 1; next }
        if ($in_alt) {
            if (/\\label\{/) { $in_alt = 0 }
            else { next }
        }
        next if /^\s*%/;
        s/(?<!\\)%.*$//;
        print;
    ' "$1"
}

# ---------------------------------------------------------------------------
# 1. Find active \input files from the main .tex file
#    Handles both:  \input filename   and   \input{filename}
#    Skips commented-out lines (lines whose first non-whitespace char is %)
# ---------------------------------------------------------------------------
INPUT_FILES=$(grep -v '^\s*%' "$SRC/$MAIN.tex" \
    | perl -ne 'print "$1\n" while /\\input\s*\{?([^}\s]+)/g')

# ---------------------------------------------------------------------------
# 2. Copy and strip comments from each \input .tex file
# ---------------------------------------------------------------------------
for f in $INPUT_FILES; do
    [[ "$f" == *.tex ]] || f="${f}.tex"
    if [[ -f "$SRC/$f" ]]; then
        strip_comments "$SRC/$f" > "$OUT/$f"
        echo "  copied (comments stripped): $f"
    else
        echo "  WARNING: referenced file not found: $f" >&2
    fi
done

# ---------------------------------------------------------------------------
# 3. Copy and process the main .tex file
#    - strip comments
#    - replace \today with a hardcoded date (arXiv rebuilds PDFs periodically,
#      causing \today to drift — see arxiv_tex_submissions.md)
#    - append 4-pass typeout hint after \end{document}
#      (needed for autonum/cleveref references to resolve — arxiv_uploading_paper_tips.md)
# ---------------------------------------------------------------------------
CURRENT_DATE=$(date +"%B %d, %Y")

strip_comments "$SRC/$MAIN.tex" \
    | perl -pe "s/\\\\today/$CURRENT_DATE/" \
    > "$OUT/$MAIN.tex"

printf '\n\\typeout{get arXiv to do 4 passes: Label(s) may have changed. Rerun}\n' \
    >> "$OUT/$MAIN.tex"

echo "  copied (comments stripped, \\today → $CURRENT_DATE, 4-pass typeout appended): $MAIN.tex"

# ---------------------------------------------------------------------------
# 4. Copy pre-compiled bibliography (.bbl only — NOT .bib)
#    arXiv uses the pre-compiled .bbl; including .bib is unnecessary
#    (arxiv_uploading_paper_tips.md)
# ---------------------------------------------------------------------------
if [[ -f "$SRC/$MAIN.bbl" ]]; then
    cp "$SRC/$MAIN.bbl" "$OUT/$MAIN.bbl"
    echo "  copied: $MAIN.bbl"
else
    echo "  WARNING: $MAIN.bbl not found — compile $MAIN.tex first to generate it" >&2
fi

# ---------------------------------------------------------------------------
# 5. Copy referenced images (PNG only) into images/ subdirectory
#
#    We keep images in images/ so \graphicspath{ {./images/} } in the main
#    .tex file requires no changes. arXiv compiles from the submission root
#    and supports subdirectories, so this should work.
#
#    NOTE: arxiv_uploading_paper_tips.md recommends flattening all figures to
#    the root directory as the safest default. If arXiv compilation fails
#    (e.g. images not found), consider flattening:
#      1. Move all PNGs from $OUT/images/ to $OUT/
#      2. In $OUT/$MAIN.tex, change \graphicspath{ {./images/} }
#         to \graphicspath{ {./} }
# ---------------------------------------------------------------------------
REFERENCED_IMAGES=$(grep -h 'includegraphics' "$OUT"/*.tex 2>/dev/null \
    | perl -ne 'print "$1\n" while /\{([^}]+\.png)\}/g' \
    | sort -u)

for img in $REFERENCED_IMAGES; do
    if [[ -f "$SRC/images/$img" ]]; then
        cp "$SRC/images/$img" "$OUT/images/$img"
        echo "  copied image: $img"
    else
        echo "  WARNING: image not found: $SRC/images/$img" >&2
    fi
done

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Contents of $OUT/ ==="
find "$OUT" -type f | sort

echo ""
echo "=== Next steps ==="
echo "  1. Compile and verify:  cd $OUT && latexmk $MAIN.tex"
echo "  2. Review the generated PDF carefully"
echo "  3. Create tarball:       cd $OUT && tar -cvf ../arxiv_submission.tar *"
echo "  4. Upload arxiv_submission.tar at arxiv.org"
