from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register DejaVu font for Unicode math symbols
pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

# Create PDF
pdf = SimpleDocTemplate("Okregi_Matematyka.pdf", pagesize=A4)
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
    fontSize=13,
    textColor='black',
    spaceAfter=10,
    spaceBefore=15,
    backColor='#DCDCDC'
)

subsection_style = ParagraphStyle(
    'CustomSubsection',
    parent=styles['Heading3'],
    fontName='DejaVu-Bold',
    fontSize=11,
    textColor='#000080',
    spaceAfter=8,
    spaceBefore=8
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontName='DejaVu',
    fontSize=11,
    leading=16,
    spaceAfter=6,
    alignment=TA_JUSTIFY
)

formula_style = ParagraphStyle(
    'Formula',
    parent=styles['Normal'],
    fontName='DejaVu-Bold',
    fontSize=12,
    leading=18,
    spaceAfter=10,
    spaceBefore=10,
    alignment=TA_CENTER,
    textColor='#000080'
)

warning_style = ParagraphStyle(
    'Warning',
    parent=styles['Normal'],
    fontName='DejaVu-Bold',
    fontSize=11,
    leading=16,
    spaceAfter=8,
    textColor='#FF0000'
)

hint_style = ParagraphStyle(
    'Hint',
    parent=styles['Normal'],
    fontName='DejaVu',
    fontSize=10,
    leading=14,
    spaceAfter=6,
    textColor='#006400',
    leftIndent=20
)

# Title
story.append(Paragraph("Równanie Okręgu - Zadania Maturalne", title_style))
story.append(Spacer(1, 0.5*cm))

# CZĘŚĆ 1: TEORIA
story.append(Paragraph("CZĘŚĆ 1: TEORIA (Przeczytaj zanim zaczniesz!)", section_style))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("1. Równanie Kanoniczne Okręgu (Święty Graal)", subsection_style))
story.append(Paragraph("Każdy okrąg w układzie współrzędnych opisujemy wzorem, który wynika wprost z twierdzenia Pitagorasa:", body_style))

story.append(Paragraph("(x − a)² + (y − b)² = r²", formula_style))

story.append(Paragraph("Gdzie:", body_style))
story.append(Paragraph("• S(a, b) – to środek okręgu (Twoja \"buda\").", body_style))
story.append(Paragraph("• r – to promień (długość łańcucha).", body_style))
story.append(Paragraph("• (x, y) – to dowolny punkt leżący na okręgu.", body_style))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("🚨 UWAGA NA PUŁAPKĘ ZNAKÓW!", warning_style))
story.append(Paragraph("Wzór ma w sobie minusy. To oznacza, że liczby w nawiasach mają przeciwne znaki niż współrzędne środka.", body_style))
story.append(Paragraph("• Widzisz (x − 3)²? → a = 3 (dodatnie).", body_style))
story.append(Paragraph("• Widzisz (y + 5)²? → b = −5 (ujemne, bo y − (−5)).", body_style))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2. Odległość jako Promień", subsection_style))
story.append(Paragraph("Często nie masz podanego promienia na tacy. Musisz go obliczyć jako odległość między dwoma punktami (Środkiem S i punktem na okręgu P):", body_style))

story.append(Paragraph("r = |SP| = √[(x<sub>P</sub> − x<sub>S</sub>)² + (y<sub>P</sub> − y<sub>S</sub>)²]", formula_style))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("3. Środek na Środku", subsection_style))
story.append(Paragraph("Jeśli środek jest w początku układu O(0, 0), wzór upraszcza się do:", body_style))

story.append(Paragraph("x² + y² = r²", formula_style))
story.append(Spacer(1, 0.5*cm))

# CZĘŚĆ 2: ZADANIA
story.append(PageBreak())
story.append(Paragraph("CZĘŚĆ 2: ZADANIA PRAKTYCZNE (Poziom: Maturalny)", section_style))
story.append(Spacer(1, 0.4*cm))

# Zadania
zadania = [
    ("Zadanie 1. (Rozgrzewka - Budowanie Równania)",
     "Podaj równanie okręgu w postaci kanonicznej, wiedząc, że jego środek znajduje się w punkcie S(−6, 2), a promień wynosi r = 4√3.",
     "(Wskazówka: Pamiętaj o zmianie znaków w nawiasie i podniesieniu ułamka do kwadratu!)"),

    ("Zadanie 2. (Inżynieria Wsteczna)",
     "Dany jest okrąg o równaniu (x − √2)² + (y + 5)² = 18. Wyznacz współrzędne środka S tego okręgu oraz długość jego promienia r.",
     "(Wskazówka: Ile wynosi √18 po wyciągnięciu czynnika przed znak pierwiastka?)"),

    ("Zadanie 3. (Analiza Wykresu - Typ Zadania 4/str. 70)",
     "Spójrz na poniższy opis graficzny okręgu w układzie współrzędnych i napisz jego równanie.\n\n"
     "• Okrąg leży w całości w IV ćwiartce (oprócz punktów styczności).\n"
     "• Jest styczny jednocześnie do osi OX w punkcie x = 3 i do osi OY w punkcie y = −3.",
     "(Wskazówka: Narysuj to. Gdzie musi być środek, jeśli dotyka obu osi i ma promień 3?)"),

    ("Zadanie 4. (Środek w początku układu)",
     "Wyznacz równanie okręgu o środku w punkcie O(0, 0), wiedząc, że przechodzi on przez punkt P(−5, 3).",
     "(Wskazówka: Musisz najpierw obliczyć r². Podstaw x i y punktu P do wzoru x² + y² = r².)"),

    ("Zadanie 5. (Zadanie z \"Psem i Budą\")",
     "Wyznacz równanie okręgu, którego środek znajduje się w punkcie S(4, 1), a punkt P(1, −3) leży na tym okręgu.",
     "(Wskazówka: Oblicz długość odcinka |SP|, to będzie Twój promień r.)"),

    ("Zadanie 6. (Średnica)",
     "Odcinek o końcach A(−1, −2) i B(3, 6) jest średnicą pewnego okręgu. Wyznacz równanie tego okręgu.",
     "(Wskazówka: To zadanie dwuetapowe. 1. Środek okręgu to środek odcinka AB. 2. Promień to połowa długości AB lub odległość środka od punktu A.)"),

    ("Zadanie 7. (Detektyw Punktów)",
     "Masz okrąg o równaniu x² + y² = 25. Sprawdź rachunkowo, czy punkt A(−3, −4) należy do tego okręgu. Czy punkt B(5, 5) znajduje się wewnątrz, na zewnątrz, czy na okręgu?",
     ""),

    ("Zadanie 8. (Pierścień Kołowy)",
     "W jednym układzie współrzędnych narysowano dwa okręgi o wspólnym środku O(0, 0) (współśrodkowe). Pierwszy ma równanie x² + y² = 5, a drugi x² + y² = 10. Oblicz pole pierścienia kołowego, czyli obszaru znajdującego się \"pomiędzy\" tymi okręgami.",
     "(Wskazówka: Pole pierścienia to P<sub>duże</sub> − P<sub>małe</sub>. Pamiętaj, że we wzorze masz podane r², a pole koła to πr².)"),

    ("Zadanie 9. (Postać Ogólna na Kanoniczną)",
     "Uzasadnij, że równanie x² + y² − 4x + 12y + 4 = 0 opisuje okrąg. Wyznacz jego środek i promień.",
     "(Wskazówka: Zastosuj wzory skróconego mnożenia \"w drugą stronę\". (x² − 4x + ...) + (y² + 12y + ...) = −4 + ...)"),

    ("Zadanie 10. (Trójkąt Prostokątny - Level Hard)",
     "Wyznacz równanie okręgu opisanego na trójkącie prostokątnym o wierzchołkach A(−2, −4), B(6, 0) i C(0, 2).",
     "(Wskazówka: W trójkącie prostokątnym środek okręgu opisanego leży DOKŁADNIE w połowie przeciwprostokątnej. Musisz najpierw ustalić, który bok jest najdłuższy/przeciwprostokątną, licząc długości boków lub rysując to w układzie).")
]

for i, (tytul, tresc, wskazowka) in enumerate(zadania, 1):
    story.append(Paragraph(tytul, subsection_style))
    story.append(Paragraph(tresc, body_style))
    if wskazowka:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(wskazowka, hint_style))
    story.append(Spacer(1, 0.5*cm))

# Build PDF
pdf.build(story)

print("✓ PDF wygenerowany pomyślnie: Okregi_Matematyka.pdf")
