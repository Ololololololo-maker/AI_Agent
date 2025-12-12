#!pip install openai matplotlib numpy

# OpenAI Settings
API_KEY = "wpisz_swoj_klucz_API_tutaj"
import openai
openai.api_key = APIKey

"""
Asystent z Function Calling do:
1. Zapytań DataFrame (pandas)
2. Zapytań SQL (sqlite)
"""

from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd
import sqlite3
import json

# Przykładowy DataFrame (na bazie moich poprzednich projektów)

def create_sample_data():
    
    # produkty
    
    products = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Monestra Deliciosa', 'Fikus Benjamina', 'Sansewieria', 'Aloes',
                 'Doniczka ceramiczne', 'Nawóz uniwersalny', 'konewka', 'ziemia do roślin'
                 ],
        'category': ['Roślina', 'Roślina', 'Roślina', 'Roślina',
                     'Akcesoria', 'Akcesoria', 'Akcesoria', 'Akcesoria'
                     ],
        'price': [89.99, 65.00, 45.50, 30.00, 25.00, 15.00, 40.00, 20.00],
        'stock': [10, 15, 20, 25, 30, 50, 100, 80],
        'difficulty': ['medium', 'hard', 'easy', 'easy', None, None, None, None],
    })
    
    # zamówienia
    
    orders = pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5],
        'customer_name': ['Jan Kowalski', 'Anna Nowak', 'Piotr Wiśniewski', 'Maria Zielińska', 'Jan Kowalski'],
        'product_id': [1, 3, 2, 5, 6],
        'quantity': [1, 2, 1, 3, 5],
        'order_date': ['2024-12-01', '2024-12-03', '2024-12-05', '2024-12-07', '2024-12-10'],
        'status': ['delivered', 'shipped', 'delivered', 'processing', 'processing']
    })
    
    return products, orders

def create_sqlite_db(products: pd.DataFrame, orders: pd.DataFrame) -> sqlite3.Connection:
    """Tworzy bazę SQLite z podanymi DataFrame'ami."""
    conn = sqlite3.connect(':memory:')
    products.to_sql('products', conn, index=False, if_exists='replace')
    orders.to_sql('orders', conn, index=False, if_exists='replace')
    return conn

# inicjalizacja danych

products_df, orders_df = create_sample_data()
db_connection = create_sqlite_db(products_df, orders_df)

print("Załadowane tabele:")
print(f"      - products: {products_df.shape[0]} wierszy")
print(f"      - orders: {orders_df.shape[0]} wierszy")

# Structured output z Pydantic

class DataFrameQuery(BaseModel):
    """Schema do zapytania do pandas DataFrame."""
    table: str = Field(description="Nazwa tabeli do zapytania: 'products' lub 'orders'")
    operation: str = Field(description="Operacja do wykonania: 'select', 'filter', 'aggregate', 'sort'")
    columns: list[str] | None = Field(default=None, description="Lista kolumn do wybrania lub operacji")
    filter_condition: str | None = Field(default=None, description="Warunek filtrowania (jeśli dotyczy)")
    sort_by: str | None = Field(default=None, description="Kolumna do sortowania (jeśli dotyczy)")
    sort_ascending: bool = Field(default=True, description="Czy sortować rosnąco (jeśli dotyczy)")
    aggregation: str | None = Field(default=None, description="Typ agregacji: 'sum', 'mean', 'count' (jeśli dotyczy)")
    group_by: str| None = Field(default=None, description="Kolumna do grupowania (jeśli dotyczy)")
    limit: int | None = Field(default=None, description="Limit liczby zwróconych wierszy (jeśli dotyczy)")
    
class SQLQuery(BaseModel):
    """Schema do zapytania SQL."""
    query: str = Field(description="Zapytanie SQL do wykonania na bazie danych.")
    explanation: str = Field(description="Wyjaśnienie zapytania SQL.")
        
# funkcje wykonujące zapytania

def execute_dataframe_query(query: DataFrameQuery) -> dict:
    """Wykonuje zapytanie na DataFrame."""
    
    # wybierz tabelę
    if query.table == 'products':
        df = products_df.copy()
    elif query.table == 'orders':
        df = orders_df.copy()
    else:
        return {"error": f"Nieznana tabela: {query.table}"}
    
    try:
        # filtrowanie (przed grupowaniem)
        if query.filter_condition:
            df = df.query(query.filter_condition)
        
        # grupowanie i agregacja (przed wyborem kolumn!)
        if query.group_by and query.aggregation:
            if query.aggregation == 'count':
                # count nie wymaga kolumny numerycznej
                df = df.groupby(query.group_by).size().reset_index(name='count')
            else:
                # dla sum, mean, min, max - wybierz kolumnę numeryczną
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    agg_col = numeric_cols[0]
                    df = df.groupby(query.group_by)[agg_col].agg(query.aggregation).reset_index()
        
        # wybór kolumn (po grupowaniu)
        elif query.columns:
            df = df[query.columns]
        
        # sortowanie
        if query.sort_by and query.sort_by in df.columns:
            df = df.sort_values(query.sort_by, ascending=query.sort_ascending)
        
        # limit
        if query.limit:
            df = df.head(query.limit)
        
        return {
            "success": True,
            "row_count": len(df),
            "data": df.to_dict(orient='records')
        }
        
    except Exception as e:
        return {"error": str(e)}
    
    
def execute_sql_query(sql_query: SQLQuery) -> dict:
    """Wykonuje zapytanie SQL na bazie danych."""
    try:
        # bezpieczeństwo - tylko SELECT
        if not sql_query.query.strip().upper().startswith('SELECT'):
            return {"error": "Tylko zapytania SELECT są dozwolone."}
        
        df = pd.read_sql_query(sql_query.query, db_connection)
        
        return {
            "success": True,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        return {"error": str(e)}
    
# definicja narzędzi dla OpenAI

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": "Wykonuje zapytanie na DataFrame (pandas). Użyj do prostych operacji.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Nazwa tabeli: 'products' lub 'orders'"},
                    "operation": {"type": "string", "description": "Operacja: 'select', 'filter', 'aggregate', 'sort'"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Kolumny do wybrania"},
                    "filter_condition": {"type": "string", "description": "Warunek filtrowania, np. 'price > 50'"},
                    "sort_by": {"type": "string", "description": "Kolumna do sortowania"},
                    "sort_ascending": {"type": "boolean", "description": "Sortowanie rosnące"},
                    "aggregation": {"type": "string", "description": "Funkcja: 'sum', 'mean', 'count', 'min', 'max'"},
                    "group_by": {"type": "string", "description": "Kolumna do grupowania"},
                    "limit": {"type": "integer", "description": "Limit wyników"}
                },
                "required": ["table", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": "Wykonuje zapytanie SQL. Użyj do złożonych zapytań z JOIN.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pełne zapytanie SQL SELECT"},
                    "explanation": {"type": "string", "description": "Krótkie wyjaśnienie co robi zapytanie"}
                },
                "required": ["query", "explanation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema_info",
            "description": "Zwraca informacje o strukturze tabel. Użyj przed pisaniem zapytań.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Nazwa tabeli: 'products', 'orders' lub 'all'"}
                },
                "required": ["table"]
            }
        }
    }
]

def get_schema_info(table: str) -> dict:
    """Zwraca informacje o strukturze tabel."""
    schemas = {
        "products": {
            "columns": list(products_df.columns),
            "dtypes": products_df.dtypes.astype(str).to_dict(),
            "sample": products_df.head(3).to_dict(orient='records'),
            "row_count": len(products_df)
        },
        "orders": {
            "columns": list(orders_df.columns),
            "dtypes": orders_df.dtypes.astype(str).to_dict(),
            "sample": orders_df.head(3).to_dict(orient='records'),
            "row_count": len(orders_df)
        }
    }
    
    if table == 'all':
        return schemas
    elif table in schemas:
        return {table: schemas[table]}
    else:
        return {"error": f"Nieznana tabela: {table}"}
    
    
# mapowanie nazw funkcji

tool_functions = {
    "query_dataframe": lambda args: execute_dataframe_query(DataFrameQuery(**args)),
    "query_sql": lambda args: execute_sql_query(SQLQuery(**args)),
    "get_schema_info": lambda args: get_schema_info(args['table'])
}


# Asystent z function calling

class DatabaseAssistant:
    """Asystent do odpytywania bazy danych z użyciem function calling OpenAI."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=APIKey)
        self.conversation_history = []
        
        # system prompt z informacjami o tabelach
        self.system_prompt = """Jesteś asystentem bazy danych sklepu "Zielony Doom".
        
        Dostępne tabele:
        1. products (produkty):
        - product_id, name, category, price, stock, difficulty
        - Kategorie: rośliny, akcesoria, pielęgnacja
        2. orders (zamówienia):
        - order_id, customer_name, product_id, quantity, order_date, status
        - Statusy: processing, shipped, delivered
        
        Zasady:
        1. Najpierw użyj get_schema_info, jeśli nie znasz struktury tabel.
        2. Dla prostych zapytań użyj query_dataframe.
        3. Dla złożonych zapytań użyj query_sql.
        4. Zawsze zwracaj wyniki w formacie JSON.
        """
        
    def ask(self, question: str) -> str:
        """Zadaj pytanie asystentowi i otrzymaj odpowiedź."""
        
        self.conversation_history.append({"role": "user", "content": question})
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        # pętla function calling
        
        while True:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # sprawdza czy są tool calls
            
            if assistant_message.tool_calls:
                # dodanie odpowiedzi asystenta
                messages.append(assistant_message)
                
                # wykonaj każde narzędzie
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Wywołuję: {function_name} z argumentami {function_args}")
                    
                    # wykonanie funkcji
                    result = tool_functions[function_name](function_args)
                    
                    print(f" Wynik: {len(result.get('data', []))} rekordów" if 'data' in result else f"Wynik: {result}")
                    
                    # dodanie wyników do wiadomości
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str)
                    })
                    
            else: # brak tool calls, jest odpowiedź
                answer = assistant_message.content
                self.conversation_history.append({"role": "assistant", "content": answer})
                return answer


API_KEY = APIKey
assistant = DatabaseAssistant(API_KEY)


import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
from ipywidgets import VBox, HBox, Layout         
            
    # UI (ipywidgets)

input_box = widgets.Text(
    placeholder='Napisz pytanie, np. "Jak często podlewać monsterę?"',
    description='Ty:',
    layout=Layout(width='70%')
)
send_button = widgets.Button(
    description='Wyślij',
    button_style='success',
    layout=Layout(width='14%')
)
reset_button = widgets.Button(
    description='Nowa rozmowa',
    button_style='warning',
    layout=Layout(width='14%')
)
chat_output = widgets.Output(
    layout={'border': '1px solid gray', 'height': '360px', 'overflow_y': 'auto', 'padding': '6px'}
    )

def on_send_clicked(_):
    with chat_output:
        user_message = input_box.value.strip()
        if not user_message:
            return
        display(Markdown(f"**👤 Ty:** {user_message}"))
        answer = assistant.ask(user_message)
        display(Markdown(f"**🤖 Asystent:** {answer}"))
        input_box.value = ""

def on_reset_clicked(_):
    global assistant
    assistant = DatabaseAssistant(api_key=APIKey)
    with chat_output:
        clear_output()
        display(Markdown("🆕 **Rozpoczęto nową rozmowę z asystentem _Zielony Doom_.**"))

send_button.on_click(on_send_clicked)
reset_button.on_click(on_reset_clicked)

display(VBox([
    chat_output,
    HBox([input_box, send_button, reset_button])
]))
            
            
            


