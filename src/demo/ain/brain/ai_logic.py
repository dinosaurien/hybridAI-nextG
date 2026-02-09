import torch
import json
import uuid
import re
import logging
from typing import List, Dict
from enum import Enum, auto
from transformers import AutoModelForCausalLM, AutoTokenizer
from ain.agents.otm_logic import OTMGenerator

logger = logging.getLogger(__name__)

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
        
        # --- FIX: Clear default sampling params to stop warnings ---
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.do_sample = False

    def _get_system_prompt(self):
        return (
            "You are a 5G RIC Operator. Translate requests into technical procedures.\n"
            "VALID KNOBS: 'MODIFY_TX_POWER', 'ADJUST_MCS_CAP', 'CHANGE_PRB_WEIGHT'.\n"
        )

    def devise_plan(self, metric: str, value: float, full_state: Dict) -> Dict:
        load = full_state.get('RRU_PrbUsedDl', 0)
        prompt = (
            f"ANOMALY: {metric} is {value:.2f}. STATE: Load {load}%.\n"
            "TASK: Create an Operational Procedure.\n"
            "Return JSON ONLY with keys: 'intent_name', 'triggering_intents', 'procedure', 'scope'."
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
            # Clean call - defaults are now cleared in __init__
            ids = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                repetition_penalty=1.1
            )
        
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
        
        # State Data
        self.intent_name = ""
        self.triggering_intents = []
        self.procedure = []
        self.assurance_timer = 0
        # FIX: 20 ticks * 0.5s/tick = 10 second duration
        self.ASSURANCE_DURATION = 20 

    def step(self, observations):
        """
        Main State Machine Loop
        """
        actions_out = []

        # 1. ACTIVATING -> THINKING
        if self.state == IntentState.ACTIVATING:
            self.state = IntentState.THINKING
            actions_out.append({"type": "LLM_REQUEST_PLAN"})

        # 2. ASSURANCE -> WITHDRAWAL (After Timer)
        elif self.state == IntentState.ASSURANCE:
            # FIX: Only send the UI update ONCE at the beginning of the state.
            if self.assurance_timer == 0:
                actions_out.append({
                    "type": "INTENT_UPDATE",
                    "intent_name": self.intent_name,
                    "triggering_intents": self.triggering_intents,
                    "procedure": self.procedure,
                    "is_manual": self.current_deviation is None
                })
            
            self.assurance_timer += 1
            
            if self.assurance_timer >= self.ASSURANCE_DURATION:
                logger.info(f"[CONTROLLER] Assurance timer expired. Withdrawing OTM.")
                self.state = IntentState.WITHDRAWAL

        # 3. WITHDRAWAL -> MONITORING
        elif self.state == IntentState.WITHDRAWAL:
            cleanup_otm = self.otm_gen.create_cleanup_otm("CELL")
            actions_out.append({"type": "EXECUTE_CLEANUP", "otm": cleanup_otm})
            
            # Reset state and clear UI
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
        self.assurance_timer = 0 # Reset timer
        return self.otm_gen.create_otm_from_outcome(outcome, f"man_{uuid.uuid4().hex[:4]}")

    def apply_llm_plan(self, outcome):
        self.intent_name = outcome.get("intent_name", "Recovery")
        self.triggering_intents = outcome.get("triggering_intents", [])
        self.procedure = outcome.get("procedure", [])
        self.state = IntentState.ASSURANCE
        self.assurance_timer = 0 # Reset timer
        return self.otm_gen.create_otm_from_outcome(outcome, f"auto_{uuid.uuid4().hex[:4]}")