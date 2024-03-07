
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm.notebook import tqdm
import pandas as pd
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
def translate_dataframe_mbart(model, tokenizer, dataframe, column_name, source_language):
    def translate_text_mbart(text, source_lang, target_lang):

        # Set source language
        tokenizer.src_lang = source_lang

        # Encode the input text
        encoded_text = tokenizer(text, return_tensors="pt")

        # Set the target language to English
        forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=forced_bos_token_id)

        # Decode the generated tokens
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    tqdm.pandas()
    dataframe[column_name + '_translated'] = dataframe[column_name].progress_apply(lambda x: translate_text_mbart(x, source_language, 'en_XX'))
    return dataframe