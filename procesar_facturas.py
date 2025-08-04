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

load_dotenv()

# ========= CONFIGURACIÓN ==========
PROJECT_ID = "invoicereader-462421"
LOCATION = "us"
PROCESSOR_GENERAL = "db4d0c95632e13dd"
CREDENTIALS_PATH = "credentials.json"
CARPETA_FACTURAS = "facturas/"
PROMPTS_PATH = "prompts.json"
POPPLER_PATH = r"C:\\Users\\User\\Documents\\Herramientas del sistema\\poppler-24.08.0\\Library\\bin"
VALID_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png")

# ========= CARGAS INICIALES ==========
with open("procesadores.json", "r", encoding="utf-8") as f:
    MAPA_PROCESADORES = json.load(f)

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS_POR_COMERCIALIZADOR = json.load(f)

openai.api_key = os.environ.get("OPENAI_API_KEY")
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = documentai.DocumentProcessorServiceClient(credentials=credentials)

# ========= FUNCIONES ==========

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
    """
    Evalúa la borrosidad de una imagen o PDF:
    - Devuelve 'LEGIBLE' si el score supera el umbral.
    - Calcula blur_score como porcentaje relativo al umbral (máx. 100%).
    """
    try:
        if path.lower().endswith(".pdf"):
            paginas = convert_from_path(
                path, dpi=200, first_page=1, last_page=1, poppler_path=POPPLER_PATH
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

        # blur_score expresado como porcentaje del umbral
        blur_score_pct = min(round((score / umbral) * 100, 2), 100)

        # Alineación total: si el score < umbral → BORROSO
        legibilidad = "LEGIBLE" if blur_score_pct >= 100 else "BORROSO"

        return legibilidad, blur_score_pct

    except Exception as e:
        print(f"⚠️ Error evaluando borrosidad en {path}: {e}")
        return "ERROR", None


def detectar_comercializador(texto):
    texto = unidecode.unidecode(texto.lower())

    if "air-e" in texto:
        return "air-e"
<<<<<<< Updated upstream
    elif "vatia" in texto:
        return "vatia"
    elif "qi energy" in texto:
        return "qienergy"
=======
    elif "qi energy" in texto:
        return "qienergy"
    elif "vatia" in texto:
        return "vatia"
>>>>>>> Stashed changes
    elif "caribemar" in texto:
        return "afinia"
    elif  "celsia" in texto:
        return "celsia"
    elif "lectrohuila" in texto:
        return "electrohuila"
    elif "emcali" in texto:
        return "emcali"
    elif "enel" in texto:
        return "enel"
    elif "essa" in texto:
        return "essa"
    elif "epm" in texto:
        return "epm"
    return "Desconocido"

def detectar_subformato_enel(texto):
    texto = unidecode.unidecode(texto.lower())
    if "factura expres" in texto:
        return "enel_express"
    elif "datos tecnicos" in texto:
        return "enel_naranja"
    elif "informacion tecnica" in texto:
        return "enel_azul"
    else:
        return "enel_normal"

def extraer_entidades(document):
    salida = {}
    for entity in document.entities:
        key = entity.type_.strip().lower()
        value = entity.mention_text.strip()
        salida[key] = value
    return salida

def extraer_bloque_json(texto):
    """
    Extrae el primer bloque JSON válido encerrado en ```json ... ``` o entre llaves { ... }
    """
    # Intenta extraer bloque entre ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", texto, re.DOTALL)
    if match:
        return match.group(1)

    # Si no hay bloque markdown, intenta extraer cualquier bloque {...}
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

        # Extraer solo el bloque JSON limpio
        bloque_json = extraer_bloque_json(contenido)
        if not bloque_json:
            print("⚠️ ChatGPT respondió sin bloque JSON.")
            print("🔍 Contenido recibido:\n", contenido)
            return fila_original

        # Intentar parsear el bloque JSON
        try:
            datos_transformados = json.loads(bloque_json)
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

# ========= PROCESAMIENTO PRINCIPAL ==========

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
            try:
                documento = procesar_documento(path, processor_id)
                entidades = extraer_entidades(documento)
                fila.update(entidades)

                # Crear fila cruda antes de transformación
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

                # Aplicar transformación con ChatGPT
                transformada = transformar_con_chatgpt(fila, comercializador)
                if transformada:
                    fila.update(transformada)
            except Exception as e:
                print(f"❌ Error al procesar {archivo} con processor específico: {e}")
                fila["analisis"] += ", ERROR_PROCESSOR"

        datos.append(fila)

    if not datos:
        print("⚠️ No se procesó ninguna factura. No se generará archivo Excel.")
        return

    columnas_deseadas = [
        "archivo", "num_paginas", "legibilidad", "blur_score", "comercializador", "analisis",
        "account_number", "niu", "region_id", "state_id", "city", "address", "level", "property", "use", "estrato_comercial",
        "factor", "operator", "marketer", "month1", "month2", "month3", "month4", "month5", "month6",
        "meter_number", "contribution", "cu", "cu_cot", "period"
    ]

    df = pd.DataFrame(datos)
    for col in columnas_deseadas:
        if col not in df.columns:
            df[col] = None

    df = df[columnas_deseadas]
    df.to_excel("facturas_resultado.xlsx", index=False)
    print("\n📅 ¡Listo! Revisa 'facturas_resultado.xlsx'.")

    # Guardar datos crudos
    df_crudo = pd.DataFrame(datos_crudos)
    df_crudo.to_excel("facturas_crudo.xlsx", index=False)
    print("📄 También se generó 'facturas_crudo.xlsx' con los datos sin transformar.")




# ========= EJECUTAR ==========
if __name__ == "__main__":
    procesar_facturas()
