To created pdf

```bash
latexmk -pdf dptig.tex
```


For word counts:

```bash
texcount -inc -sum dptig.tex
```

* `-inc` follows `\input` files
* `-sum` gives a total

By default it counts inline math as words and excludes display math — which matches conventional journal practice.

---

### Advanced: excluding content from the count

**Excluding a block (e.g. alt text)**

Wrap any block with `%TC:ignore` / `%TC:endignore`:

```latex
    }
    %TC:ignore
    Alt text: ...description...
    %TC:endignore
    \label{fig:...}
```

**Excluding a full section (e.g. appendices)**

Wrap the entire section:

```latex
%TC:ignore
\appendix
\section{...}
...
%TC:endignore
```

Or, if appendices live in a separate `\input` file, ignore the whole file:

```latex
%TC:ignore
\input{appendices}
%TC:endignore
```

**Breakdown by section**

Add `-v` for a per-section breakdown:

```bash
texcount -inc -sum -v dptig.tex
```

#### Conditional alt text in the PDF

To toggle alt text visibility at build time without editing source files, add this to the preamble of `dptig.tex`:

```latex
\ifdefined\noalttext
  \newcommand{\alttext}[1]{}
\else
  \newcommand{\alttext}[1]{\par\noindent Alt text: #1}
\fi
```

Then replace each alt text block with the command:

```latex
    }
    \alttext{Three vertically stacked panels...}
    \label{fig:posteriors}
```

Build without alt text:

```bash
latexmk -pdf -usepretex="\def\noalttext{}" dptig.tex
```

Build with alt text (default):

```bash
latexmk -pdf dptig.tex
```

`-usepretex` injects code before `\documentclass`, so `\noalttext` is defined by the time the preamble `\ifdefined` check runs.

---

**Excluded from the 6500-word count**

* Appendices — explicitly outside "main text"
* References — explicitly "no limit"
* Abstract — has its own separate 250-word limit
* Figure/table captions — journal encourages detailed captions and sets no limit on figures/tables
* Alt text — accessibility metadata, not narrative prose
* Displayed equations — conventional journal practice excludes numbered display-mode equations

**Counted**

* Main body prose — Introduction, Methods, Results, Discussion, Conclusion
* Inline math within sentences — a symbol like θ embedded in a sentence counts as one token