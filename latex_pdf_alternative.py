"""
Alternative PDF generator using matplotlib for math rendering
This works without needing LaTeX installed on your system
"""
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib import mathtext
import io
from PIL import Image
import os
import tempfile

class MathPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.temp_images = []

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Powtorzenie III - Rozwiazania i Wyjasnienia', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Strona ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def add_text(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, text)

    def add_math_inline(self, latex_code, font_size=14):
        """Render LaTeX math and insert as inline image"""
        try:
            # Create figure with math text
            fig = plt.figure(figsize=(6, 0.5))
            fig.patch.set_alpha(0)

            # Render the math
            plt.text(0.5, 0.5, f'${latex_code}$',
                    fontsize=font_size,
                    ha='center', va='center',
                    transform=fig.transFigure)
            plt.axis('off')

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.savefig(temp_file.name, dpi=150, bbox_inches='tight',
                       pad_inches=0.1, transparent=True)
            plt.close()

            self.temp_images.append(temp_file.name)

            # Add to PDF
            self.image(temp_file.name, h=7)

        except Exception as e:
            # Fallback to text if rendering fails
            self.set_font('Arial', '', 11)
            self.cell(0, 7, latex_code)
            print(f"Warning: Could not render math: {e}")

    def cleanup(self):
        """Remove temporary image files"""
        for img_path in self.temp_images:
            try:
                os.unlink(img_path)
            except:
                pass

def create_math_pdf():
    pdf = MathPDF()
    pdf.add_page()

    # Zadanie 1
    pdf.chapter_title("Zadanie 1. Oblicz")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{81}')
    pdf.ln(3)

    pdf.add_text("Pytanie: Jaka liczba pomnozona przez sama siebie da 81?\n")
    pdf.add_text("Odp: 9, bo 9 × 9 = 81.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "b) ", ln=0)
    pdf.add_math_inline(r'\sqrt[3]{-1}')
    pdf.ln(3)

    pdf.add_text("To pierwiastek trzeciego stopnia. Pytanie: Jaka liczba pomnozona przez siebie TRZY razy da -1?\n")
    pdf.add_text("Odp: -1, bo (-1) × (-1) × (-1) = -1.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "c) ", ln=0)
    pdf.add_math_inline(r'\sqrt{64} + \sqrt[3]{64}')
    pdf.ln(3)

    pdf.add_math_inline(r'\sqrt{64} = 8 \text{ (bo } 8 \times 8 = 64\text{)}')
    pdf.ln(2)
    pdf.add_math_inline(r'\sqrt[3]{64} = 4 \text{ (bo } 4 \times 4 \times 4 = 64\text{)}')
    pdf.ln(2)
    pdf.add_text("Wynik: 8 + 4 = 12.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "d) ", ln=0)
    pdf.add_math_inline(r'\sqrt{0{,}16}')
    pdf.ln(3)

    pdf.add_text("Odp: 0,4\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "f) ", ln=0)
    pdf.add_math_inline(r'\sqrt{1\frac{11}{25}}')
    pdf.ln(3)

    pdf.add_text("ZASADA: Najpierw zamieniamy liczbe mieszana na ulamek niewlasciwy.\n")
    pdf.add_math_inline(r'\sqrt{\frac{36}{25}} = \frac{6}{5} = 1{,}2')
    pdf.ln(5)

    # Zadanie 2
    pdf.chapter_title("Zadanie 2. Mnozenie")
    pdf.add_text("Zasada: Jesli mnozymy dwa pierwiastki tego samego stopnia, wrzucamy liczby pod jeden wspolny znak pierwiastka i mnozymy je.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{3} \times \sqrt{12} = \sqrt{36} = 6')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "c) ", ln=0)
    pdf.add_math_inline(r'\sqrt[3]{2} \times \sqrt[3]{4} = \sqrt[3]{8} = 2')
    pdf.ln(5)

    # Zadanie 3
    pdf.chapter_title("Zadanie 3. Dzielenie")
    pdf.add_text("Zasada: Dzielenie rowniez wrzucamy pod jeden wspolny znak pierwiastka.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{27} \div \sqrt{3} = \sqrt{\frac{27}{3}} = \sqrt{9} = 3')
    pdf.ln(5)

    # Zadanie 4
    pdf.chapter_title("Zadanie 4. Wylacz liczbe przed pierwiastek")
    pdf.add_text("Zasada: Rozkladamy liczbe pod pierwiastkiem na iloczyn, szukajac 'pary' (kwadratu idealnego).\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{12} = \sqrt{4 \times 3} = 2\sqrt{3}')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "b) ", ln=0)
    pdf.add_math_inline(r'\sqrt{20} = \sqrt{4 \times 5} = 2\sqrt{5}')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "e) ", ln=0)
    pdf.add_math_inline(r'\sqrt[3]{48} = \sqrt[3]{8 \times 6} = 2\sqrt[3]{6}')
    pdf.ln(5)

    # Zadanie 5
    pdf.chapter_title("Zadanie 5. Sprytne obliczenia")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{3} \times \sqrt{5} \times \sqrt{15}')
    pdf.ln(3)
    pdf.add_text("Zauwazmy, ze pierwsze dwa pierwiastki daja sqrt(15).\n")
    pdf.add_math_inline(r'\sqrt{15} \times \sqrt{15} = 15')
    pdf.ln(5)

    # Zadanie 6
    pdf.chapter_title("Zadanie 6. Szacowanie")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{7}')
    pdf.ln(3)
    pdf.add_math_inline(r'\sqrt{4} = 2 < \sqrt{7} < 3 = \sqrt{9}')
    pdf.ln(3)
    pdf.add_text("Odp: Liczba lezy miedzy 2 a 3.\n")

    # Zadanie 7
    pdf.chapter_title("Zadanie 7. Oblicz (Rozne techniki)")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\sqrt{131^2} = 131')
    pdf.ln(3)
    pdf.add_text("Kwadrat i pierwiastek znosza sie nawzajem.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "b) ", ln=0)
    pdf.add_math_inline(r'\sqrt{19} \times 2 \times \sqrt{19} = 19 \times 2 = 38')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "c) ", ln=0)
    pdf.add_math_inline(r'\sqrt{3} \times \sqrt{6} \times \sqrt{2} = \sqrt{36} = 6')
    pdf.ln(5)

    # Zadanie 8
    pdf.chapter_title("Zadanie 8. Wlacz liczbe pod znak pierwiastka")
    pdf.add_text("Zasada: Aby wlaczyc liczbe pod pierwiastek, nalezy ja podniesc do kwadratu.\n")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'2\sqrt{3} = \sqrt{4 \times 3} = \sqrt{12}')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "c) ", ln=0)
    pdf.add_math_inline(r'5\sqrt{2} = \sqrt{25 \times 2} = \sqrt{50}')
    pdf.ln(5)

    # Zadanie 9
    pdf.chapter_title("Zadanie 9. Usun niewymiernosc z mianownika")

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "a) ", ln=0)
    pdf.add_math_inline(r'\frac{4}{\sqrt{2}} = \frac{4\sqrt{2}}{2} = 2\sqrt{2}')
    pdf.ln(5)

    # Zadanie 10
    pdf.chapter_title("Zadanie 10. Geometria")
    pdf.add_text("Dane: Prostokat o bokach:\n")
    pdf.add_math_inline(r'a = \sqrt{2}, \quad b = 3\sqrt{2}')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "POLE:", ln=1)
    pdf.add_math_inline(r'P = a \times b = \sqrt{2} \times 3\sqrt{2} = 3 \times 2 = 6 \text{ cm}^2')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, "OBWOD:", ln=1)
    pdf.add_math_inline(r'O = 2a + 2b = 2\sqrt{2} + 6\sqrt{2} = 8\sqrt{2} \text{ cm}')
    pdf.ln(5)

    # Save PDF
    pdf.output("Wyjasnienia_Matematyka_Matplotlib.pdf")
    print("PDF wygenerowany pomyslnie z matematyka renderowana przez matplotlib!")

    # Cleanup temporary files
    pdf.cleanup()

if __name__ == '__main__':
    create_math_pdf()
