# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# client = OpenAI(api_key= )
# def ask_chatgpt(prompt):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content
# user = input("Enter your question: ")
# response = ask_chatgpt(user)
# print("ChatGPT response:", response)