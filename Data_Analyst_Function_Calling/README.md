# 📊 Intelligent Data Analyst Agent (Function Calling)

Ten projekt to implementacja agenta AI pełniącego rolę **Analityka Danych** dla sklepu "Zielony Doom".

Agent wykorzystuje mechanizm **OpenAI Function Calling**, aby dynamicznie tłumaczyć pytania w języku naturalnym na operacje na danych. Wyróżnia się **hybrydowym podejściem**: potrafi wybierać między szybkimi operacjami na DataFrame (Pandas) a złożonymi zapytaniami SQL, w zależności od kontekstu.

## 🚀 Kluczowe Funkcjonalności

1.  **Inteligentny Routing Narzędzi**: Agent sam decyduje, którego narzędzia użyć:
    *   `query_dataframe` – do filtrowania, sortowania i prostych agregacji.
    *   `query_sql` – do skomplikowanych zapytań wymagających np. złączeń (JOIN) tabel.
    *   `get_schema_info` – do sprawdzenia struktury danych przed napisaniem zapytania.
2.  **Structured Outputs (Pydantic)**: Wykorzystanie biblioteki `Pydantic` do definiowania ścisłych schematów argumentów funkcji, co eliminuje błędy składniowe w generowanym kodzie.
3.  **Bezpieczeństwo SQL**: Implementacja guardrails (zabezpieczeń) pozwalających wyłącznie na operacje `SELECT` (Read-Only).
4.  **In-Memory Database**: Automatyczna konwersja DataFrame'ów do bazy SQLite w pamięci RAM.

## 🛠️ Technologie

*   **OpenAI API** (Model `gpt-4o` z obsługą `tool_calls`)
*   **Pandas** (Manipulacja danymi)
*   **SQLite3** (Relacyjna baza danych)
*   **Pydantic** (Walidacja danych i definicja schematów)
*   **IPyWidgets** (Interfejs czatu w notebooku)

## 🧠 Analiza Architektury (Concept Analysis)

### Function Calling Flow
Zamiast generować tekst odpowiedzi bezpośrednio, model LLM działa w pętli decyzyjnej:

1.  **Analiza Intencji**: Użytkownik zadaje pytanie (np. *"Ile mamy monster?"*).
2.  **Wybór Narzędzia**: Model generuje JSON z nazwą funkcji i parametrami, np.:
    ```json
    {
      "name": "query_dataframe",
      "arguments": {
        "table": "products",
        "operation": "aggregate",
        "aggregation": "sum",
        "filter_condition": "name == 'Monstera Deliciosa'"
      }
    }
    ```
3.  **Egzekucja**: Python wykonuje funkcję i zwraca wynik (np. `{"count": 10}`).
4.  **Synteza**: Model otrzymuje wynik surowy i generuje odpowiedź końcową dla człowieka.

### Hybrid Data Strategy
System rozwiązuje odwieczny dylemat "Pandas czy SQL?":
*   **Pandas** jest używany do szybkich operacji "jednotabelowych" i statystyki opisowej.
*   **SQL** jest rezerwowany dla relacji między tabelami (np. *"Kto zamówił produkt, którego jest mało w magazynie?"*).

## 💻 Jak uruchomić

1. Sklonuj repozytorium i przejdź do folderu projektu.
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install openai pandas pydantic ipywidgets matplotlib
3. Ustaw swój klucz API w kodzie lub zmiennych środowiskowych.
4. Kod jest do zastsowania głównie w Jupyter Notebook.
