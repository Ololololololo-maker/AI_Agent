from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register DejaVu font for Unicode math symbols
pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

# Create PDF
pdf = SimpleDocTemplate("Zadania_Fizyka.pdf", pagesize=A4)
story = []

# Styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontName='DejaVu-Bold',
    fontSize=16,
    textColor='black',
    spaceAfter=30,
    alignment=TA_CENTER
)

section_style = ParagraphStyle(
    'CustomSection',
    parent=styles['Heading2'],
    fontName='DejaVu-Bold',
    fontSize=12,
    textColor='black',
    spaceAfter=10,
    spaceBefore=10,
    backColor='#DCDCDC'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontName='DejaVu',
    fontSize=11,
    leading=16,
    spaceAfter=6
)

# Title
story.append(Paragraph("Zadania z Fizyki", title_style))
story.append(Spacer(1, 0.5*cm))

# Content
content = [
    ("ZADANIE 1. \"Pułapka prędkości\" (Jednostki)", [
        "Gepard biegnie z prędkością 108 km/h. Ile to metrów na sekundę?",
        "(Pamiętaj o magicznej liczbie 3,6)."
    ]),

    ("ZADANIE 2. \"Leniwy spacer\" (Ruch jednostajny)", [
        "Żółw idzie prosto przed siebie ze stałą prędkością 0,2 m/s. Jaką drogę pokona w czasie 1 minuty?",
        "(Uwaga! Czas musisz mieć w sekundach!)"
    ]),

    ("ZADANIE 3. \"Siła spokoju\" (II Zasada Dynamiki)", [
        "Mamy wózek o masie 40 kg. Pchamy go siłą wypadkową 80 N. Z jakim przyspieszeniem (a) będzie się poruszał ten wózek?",
        "Wzór: a = F/m."
    ]),

    ("ZADANIE 4. \"Ziemia przyciąga\" (Ciężar vs Masa)", [
        "Plecak ucznia ma masę 5 kg. Jaki jest jego ciężar wyrażony w Niutonach?",
        "(Przyjmij przyspieszenie ziemskie g = 10 m/s²)."
    ]),

    ("ZADANIE 5. \"Start spod świateł\" (Ruch przyspieszony)", [
        "Motocykl rusza z miejsca z przyspieszeniem 3 m/s². Jaką drogę pokona w ciągu 4 sekund?",
        "Wzór: s = (a·t²)/2.",
        "(Podpowiedź: Najpierw podnieś czas do kwadratu, potem licz resztę)."
    ])
]

for section_title, paragraphs in content:
    story.append(Paragraph(section_title, section_style))
    for para_text in paragraphs:
        story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 0.3*cm))
    story.append(Spacer(1, 0.5*cm))

# Build PDF
pdf.build(story)

print("✓ PDF wygenerowany pomyślnie: Zadania_Fizyka.pdf")
