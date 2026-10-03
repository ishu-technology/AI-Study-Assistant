# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "google/flan-t5-small"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # from transformers import pipeline

# # generator = pipeline("text2text-generation", model= "google/flan-t5-small")

# # response = generator("Explain AI in simple words", max_length= 50)
# # print(response)

# prompt = "Explain AI in simple words"

# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens = 100)

# answer = tokenizer.decode(outputs[0])

# print(answer)
# print("Length:", len(outputs[0]))

# print("Raw output:")
# print(outputs)
# print("\nDecoded:")
# print(tokenizer.decode(outputs[0]))
# print("\nDecoded without special tokens:")
# print(tokenizer.decode(outputs[0], skip_special_tokens= True)

