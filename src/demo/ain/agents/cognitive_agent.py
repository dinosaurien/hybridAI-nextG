import asyncio
import json
import logging
import uuid
import re
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from ain.bus.mem import MemBus
from ain.bus.messages import make_msg

logger = logging.getLogger(__name__)

class CognitiveAgent:
    # Listens for: 'ai.request' (Anomaly context or User text)
    # Publishes:   'ai.response' (Raw JSON from LLM)
    def __init__(self, bus: MemBus, kb):
        self.bus = bus
        self.kb = kb
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"[INIT] Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.model.generation_config.temperature = None
        self.model.generation_config.do_sample = False

    def _get_system_prompt(self):
        return (
            "You are a 5G Network Orchestrator backed by a Knowledge Graph.\n"
            "Your Goal: Analyze anomalies and select the correct Remedial Procedures from the provided context.\n\n"
            "### OUTPUT FORMAT (JSON ONLY)\n"
            "{\n"
            "  \"intent_name\": \"Short Title (e.g. Latency Mitigation)\",\n"
            "  \"reasoning\": \"Explain WHY you are choosing these steps based on the metric state.\",\n"
            "  \"selected_procedures\": [ \"FUNC_ID_1\", \"FUNC_ID_2\" ]\n"
            "}\n\n"
            "### RULES\n"
            "ONLY use Function IDs provided in the 'RECOMMENDED PROCEDURES' list.\n"
            "Do not invent new function IDs.\n"
        )

    def _generate(self, prompt: str) -> Dict:
        msgs = [{"role": "system", "content": self._get_system_prompt()}, 
                {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
        
        resp = self.tokenizer.decode(ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        try:
            match = re.search(r"(\{.*\})", resp, re.DOTALL)
            return json.loads(match.group(1)) if match else {}
        except: return {}

    async def run(self):
        q = await self.bus.sub("ai.request")
        logger.info("[COGNITIVE] Agent Online.")
        
        while True:
            msg = await q.get()
            payload = msg.payload
            
            prompt = ""
            
            if payload.get("type") == "anomaly":
                metric = payload['metric']
                value = payload['value']
                state = payload.get('state', {})

                # Retrieval
                kb_recommendations = self.kb.query_scenario(metric)
                
                # Augmenting the prompt
                if kb_recommendations:
                    rec_str = "\n".join([f"- {r}" for r in kb_recommendations])
                else:
                    rec_str = "None. Use your general knowledge to suggest 'GENERATE_OTM_LATENCY' if applicable."

                prompt = (
                    f"### SITUATION\n"
                    f"Anomaly: {metric} is currently {value:.2f}.\n"
                    f"Network State: {json.dumps(state)}\n\n"
                    f"### RECOMMENDED PROCEDURES (FROM KNOWLEDGE BASE)\n"
                    f"{rec_str}\n\n"
                    f"### TASK\n"
                    f"Select the appropriate procedures to resolve this. Return JSON."
                )

            elif payload.get("type") == "manual":
                # Manual logic remains same for now but we want retrieval augmented generation for this as well
                prompt = f"USER REQUEST: '{payload['text']}'. Map this to known procedures if possible. JSON ONLY."

            # Generation
            logger.info(f"[COGNITIVE] Thinking with Graph Context ({(kb_recommendations)})...")
            logger.info(f"[COGNITIVE] INPUT PROMPT:\n{prompt}")
            
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(self.executor, self._generate, prompt)
            
            if outcome:
                # We just pass the raw LLM output (which has "selected_procedures")
                logger.info(f"[COGNITIVE] LLM OUTPUT:\n{json.dumps(outcome, indent=2)}")
                await self.bus.pub("execution.request", make_msg("exec", "REQ", "v1", {
                    "selected_procedures": outcome.get("selected_procedures", []),
                    "cell_id": payload.get("cell_id", "unknown"),
                    "intent_name": outcome.get("intent_name", "Optimization")
                }))


