<<<<<<< HEAD
import os
import openai
##OPENAI_API_KEY='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'
openai.api_key=os.getenv("OPENAI_API_KEY")

response=openai.Completion.create(
    model="text-davinci-003",
    prompt="ML Tutor: I am a ML/AI language model tutor\nYou: What is a language model?\nML tutor: A language model is a statistical model that describes the probability of a word give the previous words.\nYou: What is a statistical model?",
    temperature=0.3,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["You:"]
)

print(openai.Engine.list())
print(openai.Model.list())
=======
import os
import openai
openai.api_key=os.getenv("OPENAI_API_KEY")

response=openai.Completion.create(
    model="text-davinci-003",
    prompt="ML Tutor: I am a ML/AI language model tutor\nYou: What is a language model?\nML tutor: A language model is a statistical model that describes the probability of a word give the previous words.\nYou: What is a statistical model?",
    temperature=0.3,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["You:"]
)

print(openai.Engine.list())
print(openai.Model.list())
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
