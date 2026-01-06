# LaTeX-Style Math PDF Generator

I've created **two solutions** for generating PDFs with proper mathematical formatting instead of plain text like "sqrt" or "3_st_pierwiastek".

## Solution 1: PyLaTeX (Recommended) ✨

**File:** `latex_pdf_generator.py`

This creates a real LaTeX document with professional mathematical typesetting.

### Requirements:
```bash
pip install pylatex
```

**You also need LaTeX installed on your system:**
- **Windows**: Install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/)
- **macOS**: Install MacTeX: `brew install --cask mactex`
- **Linux**: `sudo apt-get install texlive-full` (Ubuntu/Debian)

### Usage:
```bash
python latex_pdf_generator.py
```

### Output:
- Generates: `Wyjasnienia_Matematyka_LaTeX.pdf`
- Beautiful LaTeX typesetting
- Proper Polish characters (ą, ę, ć, etc.)
- Professional math symbols: √, ∛, fractions, etc.

---

## Solution 2: Matplotlib Alternative (No LaTeX Required) 🎨

**File:** `latex_pdf_alternative.py`

This uses matplotlib to render math formulas as images and embeds them in the PDF. **No LaTeX installation needed!**

### Requirements:
```bash
pip install fpdf matplotlib pillow
```

### Usage:
```bash
python latex_pdf_alternative.py
```

### Output:
- Generates: `Wyjasnienia_Matematyka_Matplotlib.pdf`
- Math rendered as high-quality images
- Works on any system without LaTeX
- Good quality but slightly larger file size

---

## Comparison

| Feature | PyLaTeX | Matplotlib |
|---------|---------|------------|
| LaTeX installation required | ✅ Yes | ❌ No |
| Output quality | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Good |
| File size | Smaller | Larger |
| Setup difficulty | Medium | Easy |
| Math rendering | Native LaTeX | Images |

---

## What's Different from Original Code?

Your original code used plain text representations like:
- `sqrt(81)` → Now shows as: **√81**
- `3_st_pierwiastek(-1)` → Now shows as: **∛(-1)**
- Fractions like `6/5` → Now shows as: **⁶⁄₅**
- All math is properly typeset!

---

## Quick Start

**If you have LaTeX installed:**
```bash
pip install pylatex
python latex_pdf_generator.py
```

**If you don't have LaTeX:**
```bash
pip install fpdf matplotlib pillow
python latex_pdf_alternative.py
```

---

## Troubleshooting

### PyLaTeX errors:
- **"pdflatex not found"**: Install LaTeX (see requirements above)
- **Polish characters not showing**: Make sure you have `babel` package installed

### Matplotlib errors:
- **Import errors**: Run `pip install fpdf matplotlib pillow`
- **Math rendering issues**: Update matplotlib: `pip install --upgrade matplotlib`

---

## Need Help?

Both scripts include all 10 tasks (Zadanie 1-10) from your original code with proper mathematical formatting. Choose the solution that works best for your environment!
