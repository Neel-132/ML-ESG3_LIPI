
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=20,
    num_beam_groups=5,
    num_return_sequences=20,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

from tqdm.notebook import tqdm
def get_paraphrase(data, p_length = 20):
  columns = data.columns
  for i in tqdm(range(len(data))):
    temp = {}
    for col in columns:
      if col != 'news_content':
        temp[col] = [data.iloc[i][col]] * p_length
      else:
        temp[col] = paraphrase(data.iloc[i][col])
    #display(temp['news_content'])
    temp_df = pd.DataFrame(temp)
    if i == 0:
      final_df = temp_df
    else:
      final_df = pd.concat([final_df, temp_df], axis = 0)

  return final_df


