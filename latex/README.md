For word counts:

```bash
texcount -inc -sum precision_goal.tex
```

* `-inc` follows `\input` files;
* `-sum` gives a total. 

By default it counts inline math as words and excludes display math — which matches conventional journal practice. You can run it with -v for a breakdown by section to see where you stand.



**Excluded from the 6500-word count**

* Appendices — explicitly outside "main text"; the instructions specifically frame them as a place to move detail to avoid length issues
* References — explicitly "no limit"
* Abstract — has its own separate 250-word limit
* Figure/table captions — the instructions encourage detailed captions and say "no limit" on figures/tables, which would be contradictory if captions counted
* Displayed equations — conventional practice across journals is that numbered display-mode equations are not counted as words

**Counted**

* Main body prose — Introduction, Methods, Results, Discussion, Conclusion
* Inline math within sentences — a symbol like 
θ
θ embedded in a sentence is treated as one token; texcount (the standard LaTeX word-count tool) counts these by default