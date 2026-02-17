import torch
import json
import uuid
import re
import logging
from typing import List, Dict
from enum import Enum, auto
from transformers import AutoModelForCausalLM, AutoTokenizer
from demo.ain.agents.policy_agent import OTMGenerator

logger = logging.getLogger(__name__)

""" 
This file is not used anymore, we transition to more of memBus architecture with the agents, keep this for reference on what worked 
previously in case something broke in the new logic...
"""

class IntentState(Enum):
    MONITORING = auto()
    ACTIVATING = auto() 
    THINKING = auto()   
    ASSURANCE = auto()  
    WITHDRAWAL = auto()

class IntentParser:
    def __init__(self):
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"[INIT] Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        # Stops warnings from the module when ran
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.do_sample = False

    def _get_system_prompt(self):
        return (
            "You are a 5G Network Orchestrator Engineer. Return JSON ONLY. No markdown. No explanations.\n"
            "JSON STRUCTURE: {'intent_name': 'sentence', 'procedure': ['step1', 'step2', 'step3']}\n"
            "Available knobs you can turn:\n"
            "1. TX_POWER\n"
            "2. MCS_CAP\n"
            "3. PRB_WEIGHT\n\n"
            "Rules:\n"
            "1. You must return exactly 3 steps in the 'procedure' list.\n"
            "2. For each knob, you must choose only the direction: INCREASE or DECREASE.\n"
            "3. The word 'CHANGE' or 'SET' is forbidden. Use strictly 'INCREASE' or 'DECREASE'.\n"
            "4. Format each step exactly as: '[KNOB]: [DIRECTION] it to [reason]'.\n\n"
            "OPERATING RANGES:\n"
            "- TX Power: 2.0 to 20.0 dBm (Standard: 16.0)\n"
            "- Max MCS: 0 to 28 (Standard: 28)\n"
            "Respond only with JSON."
            "The Intent_Name should include information about the anomaly and the triggers"
        )

    def devise_plan(self, metric: str, value: float, full_state: Dict) -> Dict:
        load = full_state.get('RRU_PrbUsedDl', 0)

        curr_pow = full_state.get('tx_power_dbm', 'Unknown')
        curr_mcs = full_state.get('dl_mcs_max', 'Unknown')
        curr_prb = full_state.get('prb_weight', 'Unknown')

        prompt = (
            f"ANOMALY: {metric} is {value:.2f}. STATE: Load {load}%.\n"
            f"CURRENT CONFIG: TX Power: {curr_pow}dBm, Max MCS: {curr_mcs}, PRB Weight: {curr_prb}.\n"
            "TASK: Create an Operational Procedure.\n"
            "If a knob is already very low, do not decrease it further. Return JSON ONLY."
        )
        return self._generate(prompt)

    def parse_user_intent(self, text: str) -> Dict:
        prompt = (
            f"USER REQUEST: '{text}'.\n"
            "TASK: Decompose this into a technical network procedure. Create an Operational Procedure.\n"
            "Return JSON ONLY with keys: 'intent_name', 'triggering_intents', 'procedure', 'scope'."
        )
        return self._generate(prompt)

    def _generate(self, prompt):
        msgs = [{"role": "system", "content": self._get_system_prompt() + " JSON only. No markdown."}, 
                {"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(f"\n[DEBUG] PROMPT:\n{text}")
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            ids = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                repetition_penalty=1.1
            )
        
        resp = self.tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"[DEBUG] RESPONSE:\n{resp}")
        
        try:
            # json.loads(resp) failed sometimes previously... use regex to extract JSON block only, dont know if needed anymore
            match = re.search(r"(\{.*\})", resp, re.DOTALL)
            return json.loads(match.group(1)) if match else {}
        except: return {}

class HybridAIController:
    def __init__(self):
        self.parser = IntentParser()
        self.otm_gen = OTMGenerator()
        
        self.state = IntentState.MONITORING
        self.current_deviation = None 
        
        # State Data
        self.intent_name = ""
        self.triggering_intents = []
        self.procedure = []
        self.assurance_timer = 0
        self.ASSURANCE_DURATION = 20 

    def step(self, observations):
        actions_out = []

        # TODO: We need to somehow have assurance maybe... fix this in future
        if self.state == IntentState.ACTIVATING:
            self.state = IntentState.THINKING
            actions_out.append({"type": "LLM_REQUEST_PLAN"})

        # TODO: What to do when withdrawing, do we even need it?
        elif self.state == IntentState.ASSURANCE:
            if self.assurance_timer == 0:
                actions_out.append({
                    "type": "INTENT_UPDATE",
                    "intent_name": self.intent_name,
                    "triggering_intents": self.triggering_intents,
                    "procedure": self.procedure,
                    "is_manual": self.current_deviation is None
                })
            
            self.assurance_timer += 1
            
            # After 20 seconds (ticks), go straight to MONITORING again
            if self.assurance_timer >= self.ASSURANCE_DURATION:
                logger.info(f"[CONTROLLER] Assurance timer expired. Staying at current config and returning to Monitoring.")
                
                self.state = IntentState.MONITORING
                self.current_deviation = None
                self.assurance_timer = 0
            
                actions_out.append({"type": "UI_CLEAR"})

        return actions_out
    
    def process_deviation(self, dev_data) -> bool:
        if self.state == IntentState.MONITORING:
            self.current_deviation = dev_data
            self.state = IntentState.ACTIVATING
            return True
        return False

    def process_manual_request(self, outcome, raw_text):
        self.intent_name = outcome.get("intent_name", raw_text)
        self.triggering_intents = outcome.get("triggering_intents", ["Manual"])
        self.procedure = outcome.get("procedure", [])
        self.state = IntentState.ASSURANCE
        self.assurance_timer = 0 
        return self.otm_gen.create_otm_from_outcome(outcome, f"man_{uuid.uuid4().hex[:4]}")

    def apply_llm_plan(self, outcome):
        self.intent_name = outcome.get("intent_name", "Recovery")
        self.triggering_intents = outcome.get("triggering_intents", [])
        self.procedure = outcome.get("procedure", [])
        self.state = IntentState.ASSURANCE
        self.assurance_timer = 0 
        return self.otm_gen.create_otm_from_outcome(outcome, f"auto_{uuid.uuid4().hex[:4]}")