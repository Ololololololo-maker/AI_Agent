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
pdf = SimpleDocTemplate("Wyjasnienia_Matematyka.pdf", pagesize=A4)
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
story.append(Paragraph("Powtórzenie III - Rozwiązania i Wyjaśnienia", title_style))
story.append(Spacer(1, 0.5*cm))

# Content
content = [
    ("Zadanie 1. Oblicz", [
        "<b>a)</b> √81<br/>Pytanie: Jaka liczba pomnożona przez samą siebie da 81?<br/>Odp: 9, bo 9 × 9 = 81.",
        "<b>b)</b> ∛(-1)<br/>To pierwiastek trzeciego stopnia. Pytanie: Jaka liczba pomnożona przez siebie TRZY razy da -1?<br/>Odp: -1, bo (-1) × (-1) × (-1) = -1.",
        "<b>c)</b> √64 + ∛64<br/>√64 = 8 (bo 8×8=64).<br/>∛64 = 4 (bo 4×4×4=64).<br/>Wynik: 8 + 4 = 12.",
        "<b>d)</b> √0,16<br/>√16 = 4. W wyniku musi być połowa miejsc po przecinku (jedno miejsce).<br/>Odp: 0,4.",
        "<b>f)</b> √(1¹¹⁄₂₅)<br/>ZASADA: Najpierw zamieniamy liczbę mieszaną na ułamek niewłaściwy.<br/>1 × 25 + 11 = 36. Mamy √(36/25).<br/>Odp: Pierwiastek z góry (36) to 6, z dołu (25) to 5. Wynik: 6/5 (czyli 1,2)."
    ]),

    ("Zadanie 2. Mnożenie", [
        "Zasada: Jeśli mnożymy dwa pierwiastki tego samego stopnia, wrzucamy liczby pod jeden wspólny znak pierwiastka i mnożymy je.",
        "<b>a)</b> √3 × √12<br/>√(3 × 12) = √36 = 6.",
        "<b>c)</b> ∛2 × ∛4<br/>∛(2 × 4) = ∛8.<br/>Co razy co razy co daje 8? Odp: 2."
    ]),

    ("Zadanie 3. Dzielenie", [
        "Zasada: Dzielenie również wrzucamy pod jeden wspólny znak pierwiastka.",
        "<b>a)</b> √27 ÷ √3<br/>√(27 ÷ 3) = √9.<br/>Wynik: 3."
    ]),

    ("Zadanie 4. Wyłącz liczbę przed pierwiastek", [
        "Zasada: Rozkładamy liczbę pod pierwiastkiem na iloczyn, szukając 'pary' (kwadratu idealnego), który może wyjść przed znak.",
        "<b>a)</b> √12<br/>Rozbijamy 12 na 4 × 3 (bo z 4 da się wyciągnąć pierwiastek).<br/>√(4 × 3) = 2√3 (Dwójka wyszła przed znak, trójka została).",
        "<b>b)</b> √20<br/>20 = 4 × 5.<br/>Czwórka wychodzi jako 2. Piątka zostaje. Wynik: 2√5.",
        "<b>e)</b> ∛48<br/>Szukamy sześcianu (8, 27...). 48 = 8 × 6.<br/>∛8 = 2. Więc 2 wychodzi przed znak.<br/>Wynik: 2∛6."
    ]),

    ("Zadanie 5. Sprytne obliczenia", [
        "<b>a)</b> √3 × √5 × √15<br/>Zauważmy, że √3 × √5 = √15.<br/>Otrzymujemy więc: √15 × √15.<br/>ZASADA: Pierwiastek pomnożony przez ten sam pierwiastek daje liczbę spod pierwiastka.<br/>Wynik: 15."
    ]),

    ("Zadanie 6. Szacowanie", [
        "Zasada: Szukamy najbliższych liczb całkowitych, z których znamy pierwiastki.",
        "<b>a)</b> √7<br/>Mniejszy pierwiastek: √4 = 2.<br/>Większy pierwiastek: √9 = 3.<br/>Odp: Liczba leży między 2 a 3.",
        "<b>e)</b> ∛(-75)<br/>Szukamy sześcianów: 3³=27, 4³=64, 5³=125.<br/>-75 leży między -64 (czyli -4) a -125 (czyli -5).<br/>Odp: Między -5 a -4."
    ]),

    ("Zadanie 7. Oblicz (Różne techniki)", [
        "<b>a)</b> √(131²)<br/>Kwadrat i pierwiastek znoszą się nawzajem.<br/>Wynik: 131.",
        "<b>b)</b> √19 × 2 × √19<br/>Łączymy pary: √19 × √19 = 19.<br/>Zostaje działanie: 19 × 2 = 38.",
        "<b>c)</b> √3 × √6 × √2<br/>Wszystko pod jeden pierwiastek: √(3 × 6 × 2) = √36.<br/>Wynik: 6."
    ]),

    ("Zadanie 8. Włącz liczbę pod znak pierwiastka", [
        "Zasada: Aby włączyć liczbę pod pierwiastek, należy ją podnieść do kwadratu.",
        "<b>a)</b> 2√3<br/>2 wchodzi pod pierwiastek jako 2² = 4.<br/>Mnożymy: 4 × 3 = 12.<br/>Wynik: √12.",
        "<b>c)</b> 5√2<br/>5 wchodzi pod pierwiastek jako 25.<br/>Mnożymy: 25 × 2 = 50.<br/>Wynik: √50."
    ]),

    ("Zadanie 9. Usuń niewymierność z mianownika", [
        "Zasada: Mnożymy licznik i mianownik przez ten sam pierwiastek, który jest w mianowniku.",
        "<b>a)</b> 4/√2<br/>Mnożymy przez √2/√2.<br/>Licznik: 4√2.<br/>Mianownik: √2 × √2 = 2.<br/>Otrzymujemy: 4√2/2. Skracamy 4 i 2.<br/>Wynik: 2√2."
    ]),

    ("Zadanie 10. Geometria", [
        "Dane: Prostokąt o bokach a = √2 i b = 3√2.",
        "<b>POLE (a × b):</b><br/>√2 × 3√2.<br/>Mnożymy pierwiastki: √2 × √2 = 2.<br/>Działanie: 3 × 2 = 6.<br/>Wynik: 6 cm².",
        "<b>OBWÓD (2a + 2b):</b><br/>2(√2) + 2(3√2).<br/>2√2 + 6√2.<br/>Dodajemy wyrazy podobne: 2 + 6 = 8.<br/>Wynik: 8√2 cm."
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

print("✓ PDF wygenerowany pomyślnie: Wyjasnienia_Matematyka.pdf")
print("\nUżyte symbole matematyczne:")
print("  √ - pierwiastek kwadratowy")
print("  ∛ - pierwiastek trzeciego stopnia")
print("  × - mnożenie")
print("  ÷ - dzielenie")
print("  ² ³ - potęgi")
