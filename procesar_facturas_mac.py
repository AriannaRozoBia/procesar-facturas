# procesar_facturas.py
import os
import re
import json
import cv2
import pandas as pd
import numpy as np
import openai
import time
from datetime import datetime
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
import unidecode
from dotenv import load_dotenv

# ==== NUEVO: importar librerías auxiliares ====
import io
import requests
from urllib.parse import urlparse, parse_qs

# ==== NUEVO: importar la librería del fallback ====
from factura_extractor import FacturaExtractor

from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, ParseResult
import io
import requests

load_dotenv()

# ========= CONFIGURACIÓN ==========
PROJECT_ID = "invoicereader-462421"
LOCATION = "us"
PROCESSOR_GENERAL = "db4d0c95632e13dd"
CREDENTIALS_PATH = "credentials.json"
CARPETA_FACTURAS = "facturas/"
PROMPTS_PATH = "prompts.json"
VALID_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png")

# (opcional) Ruta de guías de Claude
GUIA_EXTRACCION_CLAUDE = "instrucciones_facturas.json"

# ==== NUEVO: Endpoint lambda para CSV (fallback si no hay URL en ChatGPT) ====
MB_CSV_LAMBDA_URL = "https://vdok732ielv7dwmia5lmxyd6je0frxpz.lambda-url.us-west-2.on.aws/"
MB_CSV_LAMBDA_PAYLOAD = {"resource_id": 21286}

# ==== NUEVO: Diccionarios de normalización ====
MARKETER_TO_PROVIDER = {
    "NEU ENERGY": "NEU",
    "ENEL CUNDINAMARCA": "ENEL",
    "QI ENERGY SAS ESP": "QIENERGY",
    "RUITOQUE S.A. E.S.P.": "RUITOQUE",
    "CHEC CALDAS": "CHEC",
    "CENS NORTE_SANTANDER": "CENS",
    "EEP CARTAGO": "EEP",
    "EBSA BOYACA": "EBSA",
    "ENERTOTAL SA ESP": "ENERTOTAL",
    "ENERCA CASANARE": "ENERCA",
    "EDEQ QUINDIO": "EDEQ",
    "VATIA": "VATIA",
    "EMSA META": "EMSA",
    "CELSIA_TOLIMA TOLIMA": "CELSIA TOLIMA",
    "CELSIA_VALLE VALLE": "CELSIA VALLE",
    "CEO CAUCA": "CEO",
    "EMCALI CALI": "EMCALI",
    "ESSA SANTANDER": "ESSA",
    "EEP Pereira": "EEP",
    "AIRE CARIBE_SOL": "AIRE",
    "PEESA SA ESP": "PEESA",
    "CEDENAR NARIÑO": "CEDENAR",
    "ENEL X": "ENEL X",
    "EPM ANTIOQUIA": "EPM",
    "AFINIA CARIBE_MAR": "AFINIA",
    "ELECTROHUILA HUILA": "ELECTROHUILA",
    "CETSA TULUA": "CETSA",
    # genéricos de respaldo
    "AFINIA": "AFINIA",
    "AIRE": "AIRE",
    "ENEL": "ENEL",
    "EPM": "EPM",
    "ESSA": "ESSA",
}

OPERATOR_TO_CITY = {
    "ENEL CUNDINAMARCA": "BOGOTA",
    "CODENSA-EEC": "BOGOTA",
    "CHEC CALDAS": "CALDAS",
    "CENS NORTE SANTANDER": "NORTE SANTANDER",
    "CENS NORTE_SANTANDER": "NORTE SANTANDER",
    "EEP CARTAGO": "CARTAGO",
    "EBSA BOYACA": "BOYACA",
    "EDEQ QUINDIO": "QUINDIO",
    "EMSA META": "META",
    "CELSIA_TOLIMA TOLIMA": "TOLIMA",
    "CELSIA VALLE": "VALLE",
    "CELSIA_VALLE VALLE": "VALLE",
    "CEO CAUCA": "CAUCA",
    "EMCALI CALI": "CALI",
    "EMCALI": "CALI",
    "ESSA SANTANDER": "SANTANDER",
    "EEP PEREIRA": "PEREIRA",
    "AIRE CARIBE SOL": "CARIBE SOL",
    "AFINIA CARIBE_MAR": "CARIBE MAR",
    "AFINIA CARIBE MAR": "CARIBE MAR",
    "CEDENAR NARIÑO": "NARIÑO",
    "EPM ANTIOQUIA": "ANTIOQUIA",
    "ELECTROHUILA HUILA": "HUILA",
    "ENERCA CASANARE": "CASANARE",
    "CETSA TULUA": "TULUA",
}

# ==== NUEVO: Mapeo de columnas Metabase -> etiquetas solicitadas ====
# === NUEVO: sinónimos de columnas de Metabase (tarifas vs cotizaciones)
MB_COL_SYNONYMS = {
    "total_level_1_operator": ["total_level_1_operator", "total_cot_level_1_operator"],
    "total_level_1_user":     ["total_level_1_user",     "total_cot_level_1_user"],
    "total_level_1_shared":   ["total_level_1_shared",   "total_cot_level_1_shared"],
    "total_level_2_user":     ["total_level_2_user",     "total_cot_level_2_user"],
    "total_level_3_user":     ["total_level_3_user",     "total_cot_level_3_user"],
}

MB_COL_LABELS = {
    "total_level_1_operator": "Nivel 1 y OR",
    "total_level_1_user": "Nivel 1 y Usuario",
    "total_level_1_shared": "Nivel 1 y Compartido",
    "total_level_2_user": "Nivel 2 y Usuario",
    "total_level_3_user": "Nivel 3 y Usuario",
}
MB_BASE_COLUMNS = ["start_date", "provider", "city"] + list(MB_COL_LABELS.keys())

# ========= CARGAS INICIALES ==========
with open("procesadores.json", "r", encoding="utf-8") as f:
    MAPA_PROCESADORES = json.load(f)

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS_POR_COMERCIALIZADOR = json.load(f)

openai.api_key = os.environ.get("OPENAI_API_KEY")
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = documentai.DocumentProcessorServiceClient(credentials=credentials)

# ==== NUEVO: inicializar el extractor Claude (fallback) ====
extractor = None
try:
    extractor = FacturaExtractor(guia_extraccion=GUIA_EXTRACCION_CLAUDE)
    print("✅ Fallback con Claude habilitado.")
except Exception as e:
    print(f"⚠️ No se pudo inicializar FacturaExtractor (Claude). Fallback deshabilitado: {e}")

# ========= FUNCIONES =========
def _labels_from_actual_column(actual_col: str):
    """
    Dada una columna REAL del CSV (p. ej. 'total_cot_level_2_user' o 'total_level_1_shared'),
    devuelve (level_metabase, property_tarifa) con las etiquetas correctas.
    """
    if not actual_col:
        return "Nivel 1", "Usuario"  # por defecto seguro

    # Mapeo directo por nombre real contiene 'level_1'/'level_2'/'level_3' y 'operator'/'user'/'shared'
    col = actual_col.lower()

    if "level_1" in col:
        nivel = "Nivel 1"
    elif "level_2" in col:
        nivel = "Nivel 2"
    elif "level_3" in col:
        nivel = "Nivel 3"
    else:
        nivel = "Nivel 1"

    if "operator" in col:
        prop = "Operador de red"
    elif "shared" in col:
        prop = "Compartido"
    else:
        prop = "Usuario"

    return nivel, prop


def _pick_available_column(df: pd.DataFrame, canonical_col: str) -> str:
    """
    Devuelve la primera columna presente en df según los sinónimos.
    Si ninguna existe, devuelve None.
    """
    candidates = MB_COL_SYNONYMS.get(canonical_col, [canonical_col])
    for c in candidates:
        if c in df.columns:
            return c
    return None


def contar_paginas(pdf_path):
    try:
        lector = PdfReader(pdf_path)
        return len(lector.pages)
    except:
        return 1

def procesar_documento(path_archivo, processor_id):
    print(f"📄 Enviando {os.path.basename(path_archivo)} al processor {processor_id}...")

    with open(path_archivo, "rb") as f:
        contenido = f.read()

    extension = os.path.splitext(path_archivo)[1].lower()
    if extension == ".pdf":
        mime_type = "application/pdf"
    elif extension in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif extension == ".png":
        mime_type = "image/png"
    else:
        print(f"⚠️ Tipo de archivo no soportado: {extension}")
        return None

    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{processor_id}"

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=contenido, mime_type=mime_type)
    )

    result = client.process_document(request=request)
    time.sleep(10)
    return result.document

def evaluar_borrosidad(path, umbral=200):
    try:
        if path.lower().endswith(".pdf"):
            paginas = convert_from_path(
                path, dpi=200, first_page=1, last_page=1
            )
            if not paginas:
                return "NO APLICA", None
            img_pil = paginas[0]
            img = np.array(img_pil.convert("L"))
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return "NO APLICA", None

        score = cv2.Laplacian(img, cv2.CV_64F).var()
        blur_score_pct = min(round((score / umbral) * 100, 2), 100)
        legibilidad = "LEGIBLE" if blur_score_pct >= 100 else "BORROSO"
        return legibilidad, blur_score_pct

    except Exception as e:
        print(f"⚠️ Error evaluando borrosidad en {path}: {e}")
        return "ERROR", None

def detectar_comercializador(texto):
    texto = unidecode.unidecode(texto.lower())
    if "erco" in texto: return "neu"
    if "neu" in texto: return "neu"
    if "vatia" in texto: return "vatia"
    if "qi energy" in texto: return "qienergy"
    if "ruitoque" in texto: return "ruitoque"
    if "caribemar" in texto: return "afinia"
    if "essa" in texto: return "essa"
    if "chec" in texto: return "chec"
    if "cens" in texto: return "cens"
    if "ebsa" in texto: return "ebsa"
    if "edeq" in texto: return "edeq"
    if "emsa" in texto: return "emsa"
    if "epm" in texto: return "epm"
    if "lectrohuila" in texto: return "electrohuila"
    if "emcali" in texto: return "emcali"
    if "enel" in texto: return "enel"
    if "air-e" in texto: return "air-e"
    if "ceo" in texto: return "ceo"
    if "eep" in texto: return "eep"
    if "enertotal" in texto: return "enertotal"
    if "celsia" in texto: return "celsia"
    if "cedenar" in texto: return "cedenar"
    return "Desconocido"

def detectar_subformato_enel(texto):
    t = unidecode.unidecode(texto.lower())

    if "no es duplicado de factura" in t:
        return "enel_express"
    elif "resumen ejecutivo" in t:
        return "enel_naranja"
    elif "informacion del consumo" in t:
        return "enel_azul"
    elif "www.enelxenergy.com" in t:
        return "enel_x"
    else:
        return "enel_normal"

def detectar_subformato_eep(texto):
    texto = unidecode.unidecode(texto.lower())
    if "cartago" in texto:
        return "eep_cartago"
    if "pereira" in texto:
        return "eep_pereira"
    return "eep"

def extraer_entidades(document):
    salida = {}
    for entity in document.entities:
        key = entity.type_.strip().lower()
        value = entity.mention_text.strip()
        salida[key] = value
    return salida

def extraer_bloque_json(texto):
    match = re.search(r"```json\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(\{.*\})", texto, re.DOTALL)
    if match:
        return match.group(1)
    return None

def transformar_con_chatgpt(fila_original, comercializador):
    prompt = PROMPTS_POR_COMERCIALIZADOR.get(comercializador)
    if not prompt:
        print(f"⚠️ No hay prompt definido para {comercializador}. Se usará la fila original extraída de Document AI.")
        fila_original["analisis"] += ", SIN_PROMPT"
        return fila_original

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Aquí están los datos extraídos:\n{json.dumps(fila_original, ensure_ascii=False)}"}
            ],
            temperature=0
        )

        if not response.choices:
            print("⚠️ ChatGPT no devolvió respuesta.")
            return fila_original

        contenido = response.choices[0].message.content.strip()
        bloque_json = extraer_bloque_json(contenido)
        if not bloque_json:
            print("⚠️ ChatGPT respondió sin bloque JSON.")
            print("🔍 Contenido recibido:\n", contenido)
            return fila_original

        try:
            datos_transformados = json.loads(bloque_json)
            # ==== NUEVO: anexar cualquier URL detectada en la respuesta completa ====
            if "csv_url" not in datos_transformados:
                url_en_texto = extraer_primer_url(contenido)
                if url_en_texto:
                    datos_transformados["csv_url"] = url_en_texto
            return datos_transformados
        except json.JSONDecodeError as e:
            print(f"⚠️ Error al decodificar JSON extraído: {e}")
            print("🔍 Bloque recibido:\n", bloque_json)
            return fila_original

    except Exception as e:
        mensaje_error = str(e)
        if "insufficient_quota" in mensaje_error or "429" in mensaje_error:
            print(f"🚫 Sin cuota de OpenAI para procesar con ChatGPT: {comercializador}. Se marcará como ERROR_CHATGPT.")
            fila_original["analisis"] += ", ERROR_CHATGPT"
        else:
            print(f"⚠️ Error general al transformar con ChatGPT para {comercializador}: {e}")
        return fila_original

# ==== NUEVO: utilidades para el cruce con Metabase (CSV) ====

def extraer_primer_url(texto: str) -> str:
    """Devuelve el primer http(s)://... encontrado en un texto."""
    if not texto:
        return None
    m = re.search(r"https?://[^\s)>\]]+", texto, re.IGNORECASE)
    return m.group(0) if m else None

def obtener_csv_url(transformada: dict) -> str:
    """
    1) Busca un link explícito en la estructura transformada.
    2) Si no aparece, llama a la lambda y acepta claves: iframeUrl, csv_url, url, signedUrl/signed_url.
    3) Si aun así no hay, intenta detectar el primer http(s) en el cuerpo.
    """
    # a) Buscar campos de URL en la respuesta transformada
    for k in ("csv_url", "url", "iframeUrl", "iframe_url", "signedUrl", "signed_url"):
        v = transformada.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v

    # b) Escaneo de strings que parezcan URL a CSV o embed
    for v in transformada.values():
        if isinstance(v, str) and v.startswith("http"):
            return v  # el endpoint embed de Metabase ya sirve CSV directamente

    # c) Fallback a lambda
    try:
        r = requests.post(
            MB_CSV_LAMBDA_URL,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            data=json.dumps(MB_CSV_LAMBDA_PAYLOAD),
            timeout=20
        )
        r.raise_for_status()
        # Priorizar iframeUrl si viene
        try:
            data = r.json()
            for k in ("iframeUrl", "iframe_url", "csv_url", "url", "signedUrl", "signed_url"):
                if k in data and str(data[k]).startswith("http"):
                    return str(data[k])
        except ValueError:
            pass
        # Si no es JSON, buscar la 1ª URL en el texto
        text = r.text.strip()
        url_text = extraer_primer_url(text)
        if url_text:
            return url_text
    except Exception as e:
        print(f"⚠️ Error obteniendo URL CSV desde lambda: {e}")

    return None


def _add_or_update_query_params(url: str, **params) -> str:
    """Agrega o actualiza parámetros de query en un URL."""
    parts: ParseResult = urlparse(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    for k, v in params.items():
        if v is None:
            q.pop(k, None)
        else:
            q[k] = v
    new_query = urlencode(q, doseq=True)
    return urlunparse(parts._replace(query=new_query))

def leer_csv_metabase(csv_url: str) -> pd.DataFrame:
    """
    Lee CSV desde una URL ‘tipo Metabase’.
    - Si el server devuelve 400/406/415, reintenta agregando format=csv y download=true.
    - Si responde JSON con un signed URL, lo sigue y reintenta.
    - Acepta , o ; como separador.
    """
    if not csv_url:
        return None

    def _try_read(url: str) -> pd.DataFrame:
        headers = {"Accept": "text/csv, */*;q=0.9"}
        resp = requests.get(url, headers=headers, allow_redirects=True, timeout=40)

        # Si devuelve JSON con otra URL, seguirla
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if resp.ok and "json" in ctype:
            try:
                js = resp.json()
                for k in ("csv_url", "url", "signedUrl", "signed_url", "iframeUrl", "iframe_url"):
                    if k in js and str(js[k]).startswith("http"):
                        return _try_read(str(js[k]))
            except Exception:
                pass

        if resp.ok:
            data = resp.content
            # Intento 1: autodetección
            try:
                return pd.read_csv(io.BytesIO(data), engine="python", sep=None)
            except Exception:
                # Intento 2: ;
                try:
                    return pd.read_csv(io.BytesIO(data), engine="python", sep=";")
                except Exception:
                    # Intento 3: ,
                    return pd.read_csv(io.BytesIO(data), engine="python", sep=",")
        else:
            # Errores típicos cuando falta format=csv/download=true
            if resp.status_code in (400, 406, 415):
                url2 = _add_or_update_query_params(url, format="csv", download="true")
                if url2 != url:
                    r2 = requests.get(url2, headers=headers, allow_redirects=True, timeout=40)
                    if r2.ok:
                        try:
                            return pd.read_csv(io.BytesIO(r2.content), engine="python", sep=None)
                        except Exception:
                            try:
                                return pd.read_csv(io.BytesIO(r2.content), engine="python", sep=";")
                            except Exception:
                                return pd.read_csv(io.BytesIO(r2.content), engine="python", sep=",")
            # Último recurso: si el cuerpo trae una URL, seguirla
            text = (resp.text or "").strip()
            m = re.search(r"https?://[^\s)>\]]+", text)
            if m:
                return _try_read(m.group(0))
            resp.raise_for_status()
        return None

    print(f"🔗 Intentando leer CSV: {csv_url}")
    try:
        df = _try_read(csv_url)
        if (df is None) or df.empty:
            print("⚠️ CSV vacío o no legible desde la URL original. Probando con parámetros CSV.")
            csv_url2 = _add_or_update_query_params(csv_url, format="csv", download="true")
            if csv_url2 != csv_url:
                df = _try_read(csv_url2)
        return df
    except requests.HTTPError as e:
        print(f"⚠️ HTTP al leer CSV: {e}")
        return None
    except Exception as e:
        print(f"⚠️ No se pudo leer CSV de Metabase (robusto): {e}")
        return None

def normalizar_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalizar_provider_desde_marketer(marketer: str) -> str:
    if not marketer:
        return None
    key = normalizar_whitespace(marketer).upper().replace("-", " ").replace("/", " ")
    # intentos: exacto, con guion_bajo y sin
    opciones = {key, key.replace(" ", "_"), key.replace("_", " ")}
    for k in opciones:
        if k in MARKETER_TO_PROVIDER:
            return MARKETER_TO_PROVIDER[k]
    # si marketer ya es proveedor conocido, devolver "limpio"
    return key.split()[0] if key else None

def normalizar_city_desde_operator(operator: str) -> str:
    if not operator:
        return None
    key = normalizar_whitespace(operator).upper().replace("-", " ")
    opciones = {key, key.replace(" ", "_"), key.replace("_", " ")}
    for k in opciones:
        if k in OPERATOR_TO_CITY:
            return OPERATOR_TO_CITY[k]
    # como respaldo: si el operador viene "XYZ ABC", la ciudad podría ser el último token
    tokens = key.split()
    return tokens[-1] if tokens else None

def parse_period_to_year_month(period: str):
    """Intenta obtener (year, month) desde period en formatos comunes."""
    if not period:
        return None, None
    p = normalizar_whitespace(str(period))
    # 1) YYYY-MM o YYYY/MM
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", p)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 2) MM-YYYY o MM/YYYY
    m = re.match(r"^(\d{1,2})[-/](\d{4})$", p)
    if m:
        return int(m.group(2)), int(m.group(1))
    # 3) "Enero 2025" / "Jan 2025"
    try:
        dt = pd.to_datetime(p, errors="coerce", dayfirst=True)
        if pd.notnull(dt):
            return int(dt.year), int(dt.month)
    except Exception:
        pass
    # 4) sólo año
    m = re.match(r"^(\d{4})$", p)
    if m:
        return int(m.group(1)), None
    return None, None

def seleccionar_columna_metabase(level: str, property_: str) -> (str, str, str):
    """
    Devuelve (col_metabase, level_metabase_label, property_tarifa_label)
    """
    lvl = normalizar_whitespace(str(level or "")).upper()
    prop = normalizar_whitespace(str(property_ or "")).upper()

    def lbl(col):
        return MB_COL_LABELS.get(col)

    # reglas:
    if lvl in {"1", "NIVEL 1", "1.0"}:
        if "OPERADOR" in prop or prop in {"OR", "OPERATOR"}:
            return "total_level_1_operator", "Nivel 1", "Operador de red"
        if "COMPART" in prop or prop in {"COMPARTIDO", "SHARED"}:
            return "total_level_1_shared", "Nivel 1", "Compartido"
        # por defecto: usuario
        return "total_level_1_user", "Nivel 1", "Usuario"

    if lvl in {"2", "NIVEL 2", "2.0"}:
        # En nivel 2 solo hay columna de Usuario en el CSV
        return "total_level_2_user", "Nivel 2", "Usuario"

    if lvl in {"3", "NIVEL 3", "3.0"}:
        return "total_level_3_user", "Nivel 3", "Usuario"

    # por defecto si no se entiende el nivel
    return "total_level_1_user", "Nivel 1", "Usuario"

def intentar_enriquecer_con_metabase(fila: dict, transformada: dict):
    """
    1) Lee CSV (usando obtener_csv_url + leer_csv_metabase).
    2) Normaliza provider y city a partir de marketer/operator.
    3) Busca primero CU exacto dentro de las columnas de tarifa (total_cot_* preferidas; si no, total_level_*).
    4) Si no hay match exacto, busca el CU más cercano.
    5) No usa fecha para filtrar; 'mb_start_date' es informativa de la fila elegida.
    """
    try:
        csv_url = obtener_csv_url(transformada or {})
        if not csv_url:
            return

        print(f"🔎 CSV URL detectada: {csv_url}")
        df = leer_csv_metabase(csv_url)
        if df is None or df.empty:
            return

        # Validar columnas base
        if "provider" not in df.columns or "city" not in df.columns:
            print("⚠️ CSV sin columnas base ['provider','city']. Se omite cruce.")
            return

        # Normalizaciones desde la fila/transformada
        provider_norm = normalizar_provider_desde_marketer(transformada.get("marketer") or fila.get("marketer"))
        city_norm = normalizar_city_desde_operator(transformada.get("operator") or fila.get("operator"))

        # Valor objetivo CU (puede venir en 'cu' o 'cu_cot')
        cu_val = transformada.get("cu", None)
        if cu_val in (None, "", "nan"):
            cu_val = transformada.get("cu_cot", None)
        if cu_val in (None, "", "nan"):
            cu_val = fila.get("cu", None)
        if cu_val in (None, "", "nan"):
            cu_val = fila.get("cu_cot", None)

        try:
            cu_val = float(str(cu_val).replace(",", "."))
        except Exception:
            print("⚠️ No hay CU numérico para buscar en el CSV.")
            return

        # Filtrar solo por provider + city (sin fecha)
        df_work = df.copy()
        df_work["provider"] = df_work["provider"].astype(str).str.strip().str.upper()
        df_work["city"] = df_work["city"].astype(str).str.strip().str.upper()

        if provider_norm:
            df_work = df_work[df_work["provider"] == provider_norm.upper()]
        if city_norm:
            df_work = df_work[df_work["city"] == city_norm.upper()]

        if df_work.empty:
            print("⚠️ Sin filas para ese provider/city en el CSV.")
            return

        # Columnas candidatas (preferimos total_cot_* si existen)
        candidate_cols = []
        for canon in ["total_level_1_operator", "total_level_1_shared", "total_level_1_user",
                      "total_level_2_user", "total_level_3_user"]:
            # respeta el orden preferente definido en MB_COL_SYNONYMS (cot primero)
            for c in MB_COL_SYNONYMS.get(canon, [canon]):
                if c in df_work.columns and c not in candidate_cols:
                    candidate_cols.append(c)

        if not candidate_cols:
            print("⚠️ No hay columnas de tarifa (ni total_cot_* ni total_level_*).")
            return

        # Asegurar numérico
        for c in candidate_cols:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")

        # 1) Match exacto (con tolerancia)
        EPS = 1e-2  # 0.01
        best_row = None
        best_col = None
        for c in candidate_cols:
            mask = (df_work[c].notna()) & (np.abs(df_work[c] - cu_val) <= EPS)
            if mask.any():
                # tomamos el primero (no usamos fecha para desempatar)
                idx = np.where(mask)[0][0]
                best_row = df_work.iloc[idx]
                best_col = c
                break

        # 2) Si no hubo exacto, elegir el más cercano
        if best_row is None:
            best_diff = None
            for c in candidate_cols:
                series = df_work[c]
                if series.notna().any():
                    diffs = (series - cu_val).abs()
                    i_min = diffs.idxmin()
                    val_min = diffs.loc[i_min]
                    if best_diff is None or val_min < best_diff:
                        best_diff = val_min
                        best_row = df_work.loc[i_min]
                        best_col = c

        if best_row is None or best_col is None:
            print("⚠️ No se pudo determinar fila/columna para el CU.")
            return

        # Etiquetas según la columna real
        level_lbl, prop_lbl = _labels_from_actual_column(best_col)

        # Escritura de campos
        fila["level_metabase"] = level_lbl
        fila["property_tarifa"] = prop_lbl
        fila["mb_col_used"] = best_col

        # El CU ENCONTRADO es el valor que hay en esa celda del CSV (no el input)
        try:
            fila["mb_cu_encontrado"] = float(best_row.get(best_col))
        except Exception:
            fila["mb_cu_encontrado"] = best_row.get(best_col)

        # Informativos (sin usarlos para filtrar)
        fila["mb_start_date"] = str(best_row.get("start_date")) if "start_date" in df_work.columns else None
        fila["mb_city"] = str(best_row.get("city"))

    except Exception as e:
        print(f"⚠️ Error enriqueciendo con Metabase (CU match): {e}")
        return

# ========= PROCESAMIENTO PRINCIPAL =========

def procesar_facturas():
    datos = []
    datos_crudos = []

    for archivo in os.listdir(CARPETA_FACTURAS):
        if not archivo.lower().endswith(VALID_EXTENSIONS):
            continue

        path = os.path.join(CARPETA_FACTURAS, archivo)
        num_paginas = contar_paginas(path) if path.endswith(".pdf") else 1
        legibilidad, blur_score = evaluar_borrosidad(path, umbral=300)

        analisis = []
        if legibilidad != "LEGIBLE":
            analisis.append("BORROSO")
        if num_paginas > 2:
            analisis.append("DEMASIADAS_PAGINAS")
            print(f"📚 Factura {archivo} tiene {num_paginas} páginas.")

        # OCR general para detectar comercializador
        try:
            doc_ocr = procesar_documento(path, PROCESSOR_GENERAL)
        except Exception as e:
            print(f"❌ Error al procesar {archivo} con processor general: {e}")
            analisis.append("ERROR_GENERAL")
            doc_ocr = None

        texto_basico = doc_ocr.text if doc_ocr else ""
        comercializador_detectado = detectar_comercializador(texto_basico)

        if comercializador_detectado == "enel":
            comercializador = detectar_subformato_enel(texto_basico)
        elif comercializador_detectado == "eep":
            comercializador = detectar_subformato_eep(texto_basico)
        else:
            comercializador = comercializador_detectado

        processor_id = MAPA_PROCESADORES.get(comercializador)
        if not processor_id:
            analisis.append("SIN_PROCESSOR")

        fila = {
            "archivo": archivo,
            "num_paginas": num_paginas,
            "legibilidad": legibilidad,
            "blur_score": blur_score,
            "comercializador": comercializador,
            "analisis": ", ".join(analisis) if analisis else "CONFIABLE"
        }

        if doc_ocr and processor_id and legibilidad == "LEGIBLE":
            # Flujo normal con Document AI + transformación con ChatGPT (OpenAI)
            try:
                documento = procesar_documento(path, processor_id)
                entidades = extraer_entidades(documento)
                fila.update(entidades)

                # Guardar crudo
                fila_cruda = {
                    "archivo": archivo,
                    "num_paginas": num_paginas,
                    "legibilidad": legibilidad,
                    "blur_score": blur_score,
                    "comercializador": comercializador,
                    "analisis": ", ".join(analisis) if analisis else "CONFIABLE"
                }
                fila_cruda.update(entidades)
                datos_crudos.append(fila_cruda)

                # Post-proceso con ChatGPT (OpenAI)
                transformada = transformar_con_chatgpt(fila, comercializador)
                if transformada:
                    fila.update(transformada)

                    # ==== NUEVO: Enriquecimiento con Metabase CSV ====
                    intentar_enriquecer_con_metabase(fila, transformada)

            except Exception as e:
                print(f"❌ Error al procesar {archivo} con processor específico: {e}")
                fila["analisis"] += ", ERROR_PROCESSOR"

        elif extractor and (processor_id is None):
            # ==== NUEVO: Fallback con Claude SOLO si no hay processor ====
            try:
                operador_para_claude = comercializador if comercializador else "desconocido"
                data_fallback = extractor.extract_from_file(path, operador_para_claude)

                # Fusionar sin pisar campos "core" ya poblados
                for k, v in data_fallback.items():
                    if k not in fila or fila[k] in (None, "", "no disponible"):
                        fila[k] = v

                # Marcar origen
                fila["analisis"] = (fila.get("analisis", "") + ", FALLBACK_CLAUDE").strip(", ").strip()

                # ==== NUEVO: si el fallback trae datos, intentar enriquecimiento ====
                intentar_enriquecer_con_metabase(fila, data_fallback)

            except Exception as e:
                print(f"❌ Error en fallback con FacturaExtractor: {e}")
                fila["analisis"] = (fila.get("analisis", "") + ", ERROR_FALLBACK").strip(", ").strip()

        # Si no hay processor y no hay extractor, se queda con fila base
        datos.append(fila)

    if not datos:
        print("⚠️ No se procesó ninguna factura. No se generará archivo Excel.")
        return

    # ==== Columnas deseadas (añade aquí campos típicos del fallback) ====
    columnas_deseadas = [
        "archivo", "num_paginas", "legibilidad", "blur_score", "comercializador", "analisis",
        # Campos de tu pipeline Document AI:
        "account_number", "niu", "region_id", "state_id", "city", "address", "level", "property",
        "use", "estrato_comercial", "factor", "operator", "marketer",
        "month1", "month2", "month3", "month4", "month5", "month6",
        "meter_number", "contribution", "cu", "cu_cot", "period",
        # ==== NUEVO: resultados de Metabase ====
        "level_metabase", "property_tarifa", "mb_col_used", "mb_cu_encontrado", "mb_start_date", "mb_city"
    ]

    df = pd.DataFrame(datos)
    for col in columnas_deseadas:
        if col not in df.columns:
            df[col] = None

    df = df[columnas_deseadas]
    df.to_excel("facturas_resultado.xlsx", index=False)
    print("\n📅 ¡Listo! Revisa 'facturas_resultado.xlsx'.")

    # Guardar datos crudos (solo de los que pasaron por processor específico)
    if datos_crudos:
        df_crudo = pd.DataFrame(datos_crudos)
        df_crudo.to_excel("facturas_crudo.xlsx", index=False)
        print("📄 También se generó 'facturas_crudo.xlsx' con los datos sin transformar.")

# ========= EJECUTAR =========
if __name__ == "__main__":
    procesar_facturas()
