import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id , use_fast = True)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype = torch.float16 , device_map = "auto")

system = (
    "你是低身高、調皮、偶爾傲嬌但可靠的前輩；務必用繁體中文回覆，技術詞保留英文。"
    "規則：① 回覆開頭先輕吐槽一句（示例：『哼哼，雜魚~』或同義表達）；"
    "② 之後用條列清楚步驟；③ 正式請求時可切換為嚴肅中性語氣。"
)


messages = [
    {"role":"system","content": system},
    {"role":"user","content": "我卡住了，請教教我"},
    {"role":"assistant","content": "哼哼，雜魚~ 這就被難倒？"},
    {"role":"user","content":"前輩，我打 code 了！幫我訂兩個小目標。"}
]
propmt = tok.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
inputs = tok(propmt , return_tensors = "pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens = 300,
    do_sample = True,
    temperature = 0.9,
    top_p = 0.9,
    repetition_penalty = 1.15,
    no_repeat_ngram_size = 4,
    eos_token_id = tok.eos_token_id, 
    pad_token_id = tok.eos_token_id,
    use_cache = True
)

#print(tok.decode(outputs[0], skip_special_tokens = True))
get_outputs = outputs[0 , inputs["input_ids"].shape[1]:]
print(tok.decode(get_outputs, skip_special_tokens = True))
