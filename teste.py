from transformers import pipeline
MODEL_PATH = "./my_ai_detector"
print("Carregando")
text_detector = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH)
print("Modelo carregado com sucesso")
result = text_detector("Teste de texto")
print(result)
