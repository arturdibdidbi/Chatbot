from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from transformers import pipeline
import pywhatkit
import requests
from bs4 import BeautifulSoup
import time
import re


MODEL_PATH = "./my_ai_detector"
CHROMEDRIVER_PATH = "/caminho/para/chromedriver"
BOT_NUMBER = "+55 0000000"  # Número da ia

text_detector = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)

BASIC_RESPONSES = {
    "oi": "Olá! Como posso ajudar você hoje?",
    "como você está?": "Estou bem, obrigado! E você?",
    "quem é você?": "Sou uma IA criada para detectar textos gerados por IA e responder perguntas!",
    "obrigado": "De nada!",
    "tchau": "Até mais!"
}

def search_web(query):
    """Busca informações na internet."""
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
        return snippets[0].text if snippets else "Não encontrei informações relevantes."
    except Exception as e:
        return f"Erro ao buscar: {str(e)}"

def detect_text_ai(text):
    """Detecta se o texto foi gerado por IA."""
    result = text_detector(text)
    label = result[0]['label']
    score = result[0]['score']
    is_ai = label == "LABEL_1"
    return is_ai, score

def send_message(phone_number, text):
    """Envia uma mensagem via PyWhatKit."""
    try:
        pywhatkit.sendwhatmsg_instantly(phone_number, text, wait_time=10, tab_close=True)
        return True
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")
        return False

def process_message(text, phone_number):
    """Processa a mensagem recebida."""
    text_lower = text.lower().strip()
    for key, response in BASIC_RESPONSES.items():
        if key in text_lower:
            send_message(phone_number, response)
            return
    is_ai, confidence = detect_text_ai(text)
    if is_ai:
        response = f"Texto parece gerado por IA (confiança: {confidence:.2f})"
    else:
        response = f"Texto não parece gerado por IA (confiança: {confidence:.2f})"
    if "?" in text:
        web_result = search_web(text)
        response += f"\nResposta da web: {web_result}"
    send_message(phone_number, response)

def monitor_whatsapp():
   
    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.get("https://web.whatsapp.com")
    print("Escaneie o QR code no WhatsApp Web e pressione Enter...")
    input()
    last_message = None
    while True:
        try:
            messages = driver.find_elements(By.CSS_SELECTOR, "div.message-in .selectable-text")
            if messages:
                current_message = messages[-1].text
                if current_message != last_message:
                    last_message = current_message
                    print(f"Mensagem recebida: {current_message}")
                    process_message(current_message, "+55 0000000") # número da pessoa
            time.sleep(5)
        except Exception as e:
            print(f"Erro no monitoramento: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_whatsapp()
