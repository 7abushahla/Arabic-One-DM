#!/usr/bin/env python
# test_strip_harakat.py
# -*- coding: utf-8 -*-

from data_loader.loader_ara import strip_harakat

# A handful of test words:
tests = [
    "أَبْجَد",       # ALEF‑HAMZA_ABOVE + FATHA + SUKUN
    "آثَار",        # ALEF‑MADDA_ABOVE + FATHA + SHADDA
    "إِسْلَام",     # ALEF‑HAMZA_BELOW + KASRA + SUKUN + FATHA
    "صَدَأْ",       # the word you mentioned, with FATHA + ALEF‑HAMZA_ABOVE + SUKUN
    "قُرْآنٌ",      # Quran with DAMMA + SUKUN + TANWEEN
    "مَدَّةٌ"       # with SHADDA + TANWEEN
]

print(" original    -> stripped")
print("────────────────────────")
for w in tests:
    stripped = strip_harakat(w)
    print(f"{w:<12} -> {stripped}")
