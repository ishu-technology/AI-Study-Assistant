import ollama

response= ollama.chat(model="qwen2.5:1.5b", messages=[
                                               {
                                                   "role":"user",
                                                   "content":"Explain Artificial Intelligence in simple words"
                                                       }
                                    ]
                     )

print(response["message"]["content"])