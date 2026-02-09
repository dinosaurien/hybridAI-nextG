import torch
import json
import uuid
import numpy as np
import logging
import re
from typing import List, Dict
from enum import Enum, auto
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class IntentState(Enum):
    MONITORING = auto()
    ACTIVATING = auto() 
    THINKING = auto()   
    ASSURANCE = auto()  
    WITHDRAWAL = auto()

class OTMGenerator:
    def __init__(self):
        self.metric_map = {
            "DRB_PdcpSduDelayDl": {"name": "latency", "unit": "ms", "service": "mbb"},
            "UE_DRB_PdcpSduDelayDl_UEID": {"name": "latency", "unit": "ms", "service": "urllc"},
            "RRU_PrbUsedDl": {"name": "resource_usage", "unit": "%", "service": "mbb"},
            "thr_dl_bps": {"name": "throughput", "unit": "Mbps", "service": "mbb"}
        }

    def generate(self, metric: str, current_value: float, llm_outcome: Dict, episode_id: str) -> Dict:
        mapping = self.metric_map.get(metric, {"name": metric, "unit": "N/A", "service": "mbb"})
        is_delay = "latency" in mapping['name'] or "delay" in metric.lower()
        target_threshold = current_value * 0.7 if is_delay else current_value * 1.3
        return {
            "version": "1.0",
            "objective": {"service": mapping["service"], "kpi": mapping["name"], "maximize": not is_delay},
            "constraints": [{
                "operator": "le" if is_delay else "ge",
                "threshold": round(float(target_threshold), 2),
                "scope": "per_user" if llm_outcome.get("scope") == "UE" else "per_cell",
                "origin": "fine_tuned_LLM", "id": "C1"
            }],
            "metadata": {"episode": episode_id, "adaptation_log": llm_outcome.get("procedure", [])}
        }

class IntentParser:
    def __init__(self):
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"[INIT] Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    def devise_plan(self, metric: str, value: float, full_state: Dict) -> Dict:
        load = full_state.get('RRU_PrbUsedDl', 0)
        tpt = full_state.get('thr_dl_bps', 0) / 1e6
        ues = full_state.get('DRB_MeanActiveUeDl', 0)
        
        prompt = (
            f"ANOMALY: {metric} is {value:.2f}. STATE: Load {load}%, Throughput {tpt:.2f}Mbps, UEs {ues}.\n\n"
            "TASK: Create an Operational Procedure for the RIC. If Throughput is 0, consider this a Link Failure or severe congestion.\n"
            "Return JSON ONLY with these exact keys:\n"
            "1. 'intent_name': Operational title.\n"
            "2. 'triggering_intents': [List of 3 symbolic strings like 'ANOMALY_DETECTED(...)'].\n"
            "3. 'procedure': [Ordered list of 5 technical steps].\n"
            "4. 'scope': 'UE' or 'CELL'."
        )
        return self._generate(prompt)

    def parse_user_intent(self, text: str) -> Dict:
        prompt = f"USER REQUEST: '{text}'. Decompose into triggering_intents and an ordered procedure. JSON only."
        return self._generate(prompt)

    def _generate(self, prompt):
        msgs = [{"role": "system", "content": "You are a 5G Architect. JSON only."}, {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"\n[DEBUG] PROMPT:\n{text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.1)
        resp = self.tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"[DEBUG] RESPONSE:\n{resp}")
        try:
            match = re.search(r"(\{.*\})", resp, re.DOTALL)
            return json.loads(match.group(1)) if match else {}
        except: return {}

class HybridAIController:
    def __init__(self):
        self.parser = IntentParser()
        self.otm_gen = OTMGenerator()
        self.state = IntentState.MONITORING
        self.current_deviation = None 
        self.intent_name = ""
        self.triggering_intents = [] # Synced
        self.procedure = []          # Synced
        self.assurance_timer = 0

    def step(self, observations):
        actions_out = []
        if self.state == IntentState.ACTIVATING:
            self.state = IntentState.THINKING
            actions_out.append({"type": "LLM_REQUEST_PLAN"})
        elif self.state == IntentState.ASSURANCE:
            if self.assurance_timer == 0:
                actions_out.append({
                    "type": "INTENT_UPDATE",
                    "intent_name": self.intent_name,
                    "triggering_intents": self.triggering_intents, # Pass new keys
                    "procedure": self.procedure,                   # Pass new keys
                    "is_manual": self.current_deviation is None
                })
            self.assurance_timer += 1
            if self.assurance_timer > 5: self.state = IntentState.WITHDRAWAL
        elif self.state == IntentState.WITHDRAWAL:
            actions_out.append({"type": "SCENARIO_END", "name": self.intent_name})
            self.state = IntentState.MONITORING
            self.current_deviation = None
            self.assurance_timer = 0
        return actions_out

    def process_deviation(self, dev_data) -> bool:
        if self.state == IntentState.MONITORING:
            self.current_deviation = dev_data
            self.state = IntentState.ACTIVATING
            return True
        return False

    def apply_llm_plan(self, outcome):
        # Syncing these variable names with the Wrapper and JS
        self.intent_name = outcome.get("intent_name", "Deviation Response")
        self.triggering_intents = outcome.get("triggering_intents", [])
        self.procedure = outcome.get("procedure", [])
        
        self.state = IntentState.ASSURANCE
        # Return OTM for the actuator tab
        dev = self.current_deviation
        return self.otm_gen.generate(dev['metric'], dev['value'], outcome, f"alert_{uuid.uuid4().hex[:4]}")

    def apply_manual_intent(self, outcome, raw_text):
        if outcome and "procedure" in outcome:
            self.intent_name = outcome.get("intent_name", raw_text)
            self.triggering_intents = outcome.get("triggering_intents", ["Manual Request"])
            self.procedure = outcome.get("procedure", [])
            self.state = IntentState.ASSURANCE
            return True
        return False