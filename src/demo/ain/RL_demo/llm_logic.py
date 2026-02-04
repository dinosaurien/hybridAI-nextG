import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, LogitsProcessorList

# --- GLOBAL CACHE ---
_MODEL = None
_TOKENIZER = None

def get_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")
        _MODEL = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16).to(device)
    return _MODEL, _TOKENIZER

def warmup_model():
    get_model()

class ActionSelector(LogitsProcessor):
    """Bias logic to favor specific words based on the first token."""
    def __init__(self, tokenizer, valid_words, preferred_words, prompt_len, bias=2.0):
        self.prompt_len = prompt_len
        self.bias = bias
        self.valid_starts = set()
        self.preferred_starts = set()
        
        def get_ids(word):
            res = []
            t1 = tokenizer.encode(word, add_special_tokens=False)
            t2 = tokenizer.encode(" " + word, add_special_tokens=False)
            if t1: res.append(t1[0])
            if t2: res.append(t2[0])
            return res

        for word in valid_words:
            for tid in get_ids(word):
                self.valid_starts.add(tid)
        
        for word in preferred_words:
            for tid in get_ids(word):
                self.preferred_starts.add(tid)
                
        self.valid_starts = list(self.valid_starts)
        self.preferred_starts = list(self.preferred_starts)

    def __call__(self, input_ids, scores):
        # Only intervene on the FIRST token generation
        if input_ids.shape[1] == self.prompt_len:
            mask = torch.full_like(scores, -float('inf'))
            mask[:, self.valid_starts] = scores[:, self.valid_starts]
            mask[:, self.preferred_starts] += self.bias
            return mask
        return scores

def generate_biased_gpt2_intent(dev):
    model, tokenizer = get_model()
    
    if model.device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_used = torch.cuda.memory_allocated(0) / 1024 / 1024
        hw_info = f"{gpu_name} (VRAM: {vram_used:.0f}MB)"
    else:
        hw_info = "CPU"
    
    valid_actions = ["SCHEDULER_POLICY", "MCS_CAP", "PRB_WEIGHT", "SLICE_QOS", "TX_POWER", "POWER_CONTROL"] 
    preferred_actions = ["PRB_WEIGHT", "TX_POWER"] 
    
    prompt = (
        f"Network Alert: {dev.get('metric')} is {dev.get('value', 0):.2f}\n"
        f"Options: {', '.join(valid_actions)}\n"
        f"Selection:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    output = model.generate(
        **inputs, 
        max_new_tokens=6, 
        logits_processor=LogitsProcessorList([
            ActionSelector(tokenizer, valid_actions, preferred_actions, prompt_len, bias=3.0)
        ]), 
        do_sample=True, 
        temperature=0.9, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    suggestion_raw = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()
    clean_response = re.split(r'[,\n]', suggestion_raw)[0].strip()
    suggestion_upper = clean_response.upper()
    
    selected_action = "MCS_CAP" 

    if "PRB" in suggestion_upper or "WEIGHT" in suggestion_upper:
        selected_action_type = "PRB_WEIGHT"
    elif "SCHEDULER" in suggestion_upper or "POLICY" in suggestion_upper:
        selected_action_type = "SCHEDULER_POLICY"
    elif "SLICE" in suggestion_upper or "QOS" in suggestion_upper:
        selected_action_type = "SLICE_QOS"
    elif "TX" in suggestion_upper: # TX_POWER is more specific than POWER
        selected_action_type = "TX_POWER"
    elif "POWER" in suggestion_upper: 
        # Maps to the generic POWER_CONTROL if TX is not present
        selected_action_type = "POWER_CONTROL"
    elif "REPORT" in suggestion_upper:
        selected_action_type = "REPORTING"
    elif "MCS" in suggestion_upper: 
        # Maps to the most specific MCS action
        selected_action_type = "MCS_CAP"


    final_intent = "INCREASE_THROUGHPUT" # Default

    if selected_action_type in ("TX_POWER", "POWER_CONTROL", "SLICE_QOS"):
        final_intent = "INCREASE_RELIABILITY"
    elif selected_action_type in ("MCS_CAP", "SCHEDULER_POLICY", "PRB_WEIGHT"):
        final_intent = "INCREASE_THROUGHPUT"
    elif selected_action_type == "REPORTING":
        final_intent = "MONITORING" # Does nothing but observe

    # 4. Print & Return
    print("-" * 50)
    print(f"[GPT-2] Running on:   {hw_info}")
    print(f"[GPT-2] Raw Output:   '{clean_response}'")
    print(f"[GPT-2] Clean Action: {selected_action}")
    print("-" * 50)

    return {
        "intent": final_intent,
        "category": "performance",
        "goal": "restore_slo",
        "slo": {
            "suggestion": selected_action,
            "metric": dev.get("metric"),
            "target": dev.get("target")
        },
        "scope": dev.get("scope", {})
    }