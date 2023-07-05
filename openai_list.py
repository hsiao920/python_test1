<<<<<<< HEAD
import os
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

response=openai.Image.create(
    prompt="a white siamese cat",
    n=1,
    size="1024x1024"
    )
image_url=response['data'][0]['url']

=======
import os
import openai

openai.api_key=os.getenv("OPENAI_API_KEY")

response=openai.Image.create(
    prompt="a white siamese cat",
    n=1,
    size="1024x1024"
    )
image_url=response['data'][0]['url']

>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
