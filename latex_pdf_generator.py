from pylatex import Document, Section, Subsection, Command, Package, NoEscape
from pylatex.utils import bold

def create_math_pdf():
    # Create a new document
    geometry_options = {"margin": "2.5cm"}
    doc = Document(geometry_options=geometry_options)

    # Add packages for Polish characters and better math support
    doc.preamble.append(Command('usepackage', 'amsmath'))
    doc.preamble.append(Command('usepackage', 'amssymb'))
    doc.preamble.append(Command('usepackage', NoEscape('[utf8]{inputenc}')))
    doc.preamble.append(Command('usepackage', NoEscape('[polish]{babel}')))
    doc.preamble.append(Command('title', 'Powtórzenie III - Rozwiązania i Wyjaśnienia'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))

    doc.append(NoEscape(r'\maketitle'))

    # Zadanie 1
    with doc.create(Section('Zadanie 1. Oblicz')):
        doc.append(NoEscape(r'\textbf{a)} $\sqrt{81}$'))
        doc.append('\n\n')
        doc.append('Pytanie: Jaka liczba pomnożona przez samą siebie da 81?')
        doc.append('\n\n')
        doc.append(NoEscape(r'Odp: 9, bo $9 \times 9 = 81$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{b)} $\sqrt[3]{-1}$'))
        doc.append('\n\n')
        doc.append('To pierwiastek trzeciego stopnia. Pytanie: Jaka liczba pomnożona przez siebie TRZY razy da -1?')
        doc.append('\n\n')
        doc.append(NoEscape(r'Odp: -1, bo $(-1) \times (-1) \times (-1) = -1$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{c)} $\sqrt{64} + \sqrt[3]{64}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{64} = 8$ (bo $8 \times 8 = 64$).'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt[3]{64} = 4$ (bo $4 \times 4 \times 4 = 64$).'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $8 + 4 = 12$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{d)} $\sqrt{0{,}16}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{16} = 4$. W wyniku musi być połowa miejsc po przecinku (jedno miejsce).'))
        doc.append('\n\n')
        doc.append('Odp: 0,4.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{f)} $\sqrt{1\frac{11}{25}}$'))
        doc.append('\n\n')
        doc.append('ZASADA: Najpierw zamieniamy liczbę mieszaną na ułamek niewłaściwy.')
        doc.append('\n\n')
        doc.append(NoEscape(r'$1 \times 25 + 11 = 36$. Mamy $\sqrt{\frac{36}{25}}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Odp: Pierwiastek z góry (36) to 6, z dołu (25) to 5. Wynik: $\frac{6}{5}$ (czyli 1,2).'))

    # Zadanie 2
    with doc.create(Section('Zadanie 2. Mnożenie')):
        doc.append('Zasada: Jeśli mnożymy dwa pierwiastki tego samego stopnia, wrzucamy liczby pod jeden wspólny znak pierwiastka i mnożymy je.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $\sqrt{3} \times \sqrt{12}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{3 \times 12} = \sqrt{36} = 6$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{c)} $\sqrt[3]{2} \times \sqrt[3]{4}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt[3]{2 \times 4} = \sqrt[3]{8}$.'))
        doc.append('\n\n')
        doc.append('Co razy co razy co daje 8? Odp: 2.')

    # Zadanie 3
    with doc.create(Section('Zadanie 3. Dzielenie')):
        doc.append('Zasada: Dzielenie również wrzucamy pod jeden wspólny znak pierwiastka.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $\sqrt{27} \div \sqrt{3}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{\frac{27}{3}} = \sqrt{9}$.'))
        doc.append('\n\n')
        doc.append('Wynik: 3.')

    # Zadanie 4
    with doc.create(Section('Zadanie 4. Wyłącz liczbę przed pierwiastek')):
        doc.append('Zasada: Rozkładamy liczbę pod pierwiastkiem na iloczyn, szukając "pary" (kwadratu idealnego), który może wyjść przed znak.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $\sqrt{12}$'))
        doc.append('\n\n')
        doc.append('Rozbijamy 12 na 4 × 3 (bo z 4 da się wyciągnąć pierwiastek).')
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{4 \times 3} = 2\sqrt{3}$ (Dwójka wyszła przed znak, trójka została).'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{b)} $\sqrt{20}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$20 = 4 \times 5$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Czwórka wychodzi jako 2. Piątka zostaje. Wynik: $2\sqrt{5}$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{e)} $\sqrt[3]{48}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Szukamy sześcianu (8, 27...). $48 = 8 \times 6$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt[3]{8} = 2$. Więc 2 wychodzi przed znak.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $2\sqrt[3]{6}$.'))

    # Zadanie 5
    with doc.create(Section('Zadanie 5. Sprytne obliczenia')):
        doc.append(NoEscape(r'\textbf{a)} $\sqrt{3} \times \sqrt{5} \times \sqrt{15}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Zauważmy, że $\sqrt{3} \times \sqrt{5} = \sqrt{15}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Otrzymujemy więc: $\sqrt{15} \times \sqrt{15}$.'))
        doc.append('\n\n')
        doc.append('ZASADA: Pierwiastek pomnożony przez ten sam pierwiastek daje liczbę spod pierwiastka.')
        doc.append('\n\n')
        doc.append('Wynik: 15.')

    # Zadanie 6
    with doc.create(Section('Zadanie 6. Szacowanie')):
        doc.append('Zasada: Szukamy najbliższych liczb całkowitych, z których znamy pierwiastki.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $\sqrt{7}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Mniejszy pierwiastek: $\sqrt{4} = 2$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Większy pierwiastek: $\sqrt{9} = 3$.'))
        doc.append('\n\n')
        doc.append('Odp: Liczba leży między 2 a 3.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{e)} $\sqrt[3]{-75}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Szukamy sześcianów: $3^3=27$, $4^3=64$, $5^3=125$.'))
        doc.append('\n\n')
        doc.append('-75 leży między -64 (czyli -4) a -125 (czyli -5).')
        doc.append('\n\n')
        doc.append('Odp: Między -5 a -4.')

    # Zadanie 7
    with doc.create(Section('Zadanie 7. Oblicz (Różne techniki)')):
        doc.append(NoEscape(r'\textbf{a)} $\sqrt{131^2}$'))
        doc.append('\n\n')
        doc.append('Kwadrat i pierwiastek znoszą się nawzajem.')
        doc.append('\n\n')
        doc.append('Wynik: 131.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{b)} $\sqrt{19} \times 2 \times \sqrt{19}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Łączymy pary: $\sqrt{19} \times \sqrt{19} = 19$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Zostaje działanie: $19 \times 2 = 38$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{c)} $\sqrt{3} \times \sqrt{6} \times \sqrt{2}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wszystko pod jeden pierwiastek: $\sqrt{3 \times 6 \times 2} = \sqrt{36}$.'))
        doc.append('\n\n')
        doc.append('Wynik: 6.')

    # Zadanie 8
    with doc.create(Section('Zadanie 8. Włącz liczbę pod znak pierwiastka')):
        doc.append('Zasada: Aby włączyć liczbę pod pierwiastek, należy ją podnieść do kwadratu.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $2\sqrt{3}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'2 wchodzi pod pierwiastek jako $2^2 = 4$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Mnożymy: $4 \times 3 = 12$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $\sqrt{12}$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{c)} $5\sqrt{2}$'))
        doc.append('\n\n')
        doc.append('5 wchodzi pod pierwiastek jako 25.')
        doc.append('\n\n')
        doc.append(NoEscape(r'Mnożymy: $25 \times 2 = 50$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $\sqrt{50}$.'))

    # Zadanie 9
    with doc.create(Section('Zadanie 9. Usuń niewymierność z mianownika')):
        doc.append('Zasada: Mnożymy licznik i mianownik przez ten sam pierwiastek, który jest w mianowniku.')
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{a)} $\frac{4}{\sqrt{2}}$'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Mnożymy przez $\frac{\sqrt{2}}{\sqrt{2}}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Licznik: $4\sqrt{2}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Mianownik: $\sqrt{2} \times \sqrt{2} = 2$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Otrzymujemy: $\frac{4\sqrt{2}}{2}$. Skracamy 4 i 2.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $2\sqrt{2}$.'))

    # Zadanie 10
    with doc.create(Section('Zadanie 10. Geometria')):
        doc.append(NoEscape(r'Dane: Prostokąt o bokach $a = \sqrt{2}$ i $b = 3\sqrt{2}$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{POLE} $(a \times b)$:'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$\sqrt{2} \times 3\sqrt{2}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Mnożymy pierwiastki: $\sqrt{2} \times \sqrt{2} = 2$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Działanie: $3 \times 2 = 6$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $6$ cm$^2$.'))
        doc.append('\n\n')

        doc.append(NoEscape(r'\textbf{OBWÓD} $(2a + 2b)$:'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$2\sqrt{2} + 2 \times 3\sqrt{2}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'$2\sqrt{2} + 6\sqrt{2}$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Dodajemy wyrazy podobne: $2 + 6 = 8$.'))
        doc.append('\n\n')
        doc.append(NoEscape(r'Wynik: $8\sqrt{2}$ cm.'))

    # Generate PDF
    doc.generate_pdf('Wyjasnienia_Matematyka_LaTeX', clean_tex=False, compiler='pdflatex')
    print("PDF wygenerowany pomyślnie z prawdziwym formatowaniem LaTeX!")

if __name__ == '__main__':
    create_math_pdf()
