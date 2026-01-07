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
pdf = SimpleDocTemplate("Stacje_Fizyka.pdf", pagesize=A4)
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

instruction_style = ParagraphStyle(
    'Instruction',
    parent=styles['Normal'],
    fontName='DejaVu-Bold',
    fontSize=11,
    leading=16,
    spaceAfter=6,
    textColor='#000080'
)

# Title
story.append(Paragraph("Stacje Fizyczne - Siły i Dynamika", title_style))
story.append(Spacer(1, 0.3*cm))

# Instructions
story.append(Paragraph("Instrukcja:", instruction_style))
story.append(Paragraph("• Pisz obliczenia, nie tylko wynik.", body_style))
story.append(Paragraph("• Pamiętaj o jednostkach w klamrach, np. [N], [m/s²].", body_style))
story.append(Paragraph("• Korzystaj ze ściągi: g ≈ 10 m/s².", body_style))
story.append(Spacer(1, 0.7*cm))

# Content
content = [
    ("Stacja 1: Architekt Strzałek (Wektory)", [
        "Narysuj w zeszycie piłkę leżącą na trawie. Masz zaznaczyć siłę, z jaką piłka naciska na ziemię (F<sub>n</sub> = 4 N).",
        "<b>Wytyczne:</b>",
        "• Przyjmij skalę: 1 cm = 2 N.",
        "• Zaznacz <b>Punkt przyłożenia</b> (Gdzie ma być kropka? Na piłce czy na ziemi? Przypomnij sobie filmik!).",
        "• Zadbaj o poprawny <b>Kierunek</b> (poziomy czy pionowy?) i <b>Zwrot</b>."
    ]),

    ("Stacja 2: Przeciąganie Liny (Siła Wypadkowa)", [
        "Dwie drużyny ciągną linę.",
        "• Drużyna A (lewa): Ciągnie z siłą 350 N.",
        "• Drużyna B (prawa): Ciągnie z siłą 400 N.",
        "<b>Zadanie:</b>",
        "a) Oblicz wartość siły wypadkowej (F<sub>w</sub>).",
        "b) W którą stronę \"poleci\" lina? (Podaj zwrot).",
        "c) Narysuj to schematycznie (wektory o różnych długościach)."
    ]),

    ("Stacja 3: Pułapka \"Warzywniaka\" (Masa vs Ciężar)", [
        "Twoja nowa konsola do gier ma masę 3500 g.",
        "<b>Zadanie:</b>",
        "a) Zamień masę na jednostkę podstawową układu SI (kilogramy).",
        "b) Oblicz, z jaką siłą Ziemia przyciąga tę konsolę (czyli oblicz jej ciężar F<sub>g</sub>)."
    ]),

    ("Stacja 4: Start spod świateł (II Zasada Dynamiki)", [
        "Motocykl wraz z kierowcą ma masę 200 kg. Silnik działa na niego siłą wypadkową 600 N.",
        "<b>Zadanie:</b>",
        "Oblicz, z jakim przyspieszeniem (a) ruszy motocykl.",
        "Wzór: a = F/m"
    ]),

    ("Stacja 5: Hamowanie (Przekształcanie)", [
        "Pociąg towarowy o masie 100 000 kg musi zahamować. Aby uzyskać opóźnienie (przyspieszenie hamowania) o wartości 0,5 m/s², potrzebna jest potężna siła hamująca.",
        "<b>Zadanie:</b>",
        "Oblicz wartość tej siły (F).",
        "Wskazówka: Przekształć wzór a = F/m tak, żeby F zostało samo."
    ]),

    ("Stacja 6: Armata i Kula (III Zasada Dynamiki)", [
        "Wystrzelono kulę armatnią.",
        "• <b>Fakt:</b> Siła działająca na kulę jest TAKA SAMA jak siła odrzutu działająca na armatę (tylko zwroty są przeciwne).",
        "• <b>Pytanie:</b> Skoro siły są takie same, to dlaczego kula leci 2 kilometry, a armata cofa się tylko o pół metra?",
        "Wyjaśnij to, używając pojęcia MASY i II Zasady Dynamiki (a = F/m)."
    ]),

    ("Stacja 7: BOSS LEVEL (Zadanie łączone)", [
        "Przesuwasz ciężką szafę o masie 50 kg po podłodze. Pchasz ją siłą mięśni 200 N, ale podłoga stawia opór (tarcie) wynoszący 50 N.",
        "<b>Kroki do wykonania:</b>",
        "1. Oblicz <b>Ciężar szafy</b> (F<sub>g</sub>).",
        "2. Oblicz <b>Siłę Wypadkową</b> (F<sub>w</sub>) działającą w poziomie (Siła Twoja minus Tarcie).",
        "3. Oblicz <b>Przyspieszenie</b> (a), z jakim szafa zacznie się przesuwać (użyj Siły Wypadkowej z punktu 2!)."
    ])
]

for section_title, paragraphs in content:
    story.append(Paragraph(section_title, section_style))
    for para_text in paragraphs:
        story.append(Paragraph(para_text, body_style))
        story.append(Spacer(1, 0.2*cm))
    story.append(Spacer(1, 0.5*cm))

# Build PDF
pdf.build(story)

print("✓ PDF wygenerowany pomyślnie: Stacje_Fizyka.pdf")
