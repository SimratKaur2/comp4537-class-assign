from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
#This part points to the model and the tokenizer you want to download
tokenizer_to_download = GPT2Tokenizer.from_pretrained("distilgpt2")
model_to_download = TFGPT2LMHeadModel.from_pretrained("distilgpt2")

#This part saves (downloads) the models and tokenizer in your colab (running on your host, give your own path Amir)
tokenizer_to_download.save_pretrained("C:/HuggingFaceTrainedModel/tokenizer")
model_to_download.save_pretrained("C:/HuggingFaceTrainedModel/model")

#This part assigns the model and the tokenizer from the saved files
tokenizer = GPT2Tokenizer.from_pretrained("c:/HuggingFaceTrainedModel/tokenizer")
model = TFGPT2LMHeadModel.from_pretrained("c:/HuggingFaceTrainedModel/model")
input_data = "amir woke up this morning and saw " #@param {type:"string"}

input_ids = tokenizer.encode(input_data, return_tensors="tf")

output = model.generate(
    input_ids=input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.0,
    do_sample=True,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(generated_text)