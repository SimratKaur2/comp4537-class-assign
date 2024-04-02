
inputs = ["Although I'm thrilled about the promotion and the opportunities it brings, I can't help but feel sad about leaving my colleagues who have become like family."
          ]
from transformers import pipeline 
classifier = pipeline("sentiment-analysis")
for input in inputs:
  print(classifier(input)[0]['label'] + ':'+str(round(classifier(input)[0]['score'],3))+":"+input)
