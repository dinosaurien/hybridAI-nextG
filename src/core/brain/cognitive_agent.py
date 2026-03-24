import asyncio
import json
import uuid
import datetime
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.bus.mem import MemBus
from core.bus.messages import make_msg

logger = logging.getLogger(__name__)

OTM_SCHEMA_TEMPLATE = """
{
  "version": "1.0",
  "objective": { "service": "mbb", "kpi": "<MAIN_KPI>", "aggregation": "mean", "unit": "<UNIT>", "maximize": true_or_false },
  "constraints": [
    { "service": "mbb", "kpi": "<CONSTRAINT_KPI>", "operator": "<ge_le_gt_lt>", "threshold": 0.0, "aggregation": "min", "unit": "<UNIT>", "scope": "per_cell_window", "origin": "fine_tuned_LLM", "adapted_by": "ICL_LLM", "id": "C1" }
  ],
  "metadata": { "timescale": "10s_window", "timestamp": "<TIME>", "episode": "<ID>", "adaptation_log": [] }
}
"""

class CognitiveAgent:
    def __init__(self, bus: MemBus, kb):
        self.bus = bus
        self.kb = kb
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
        logger.info(f"[INIT] Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="cuda",
            attn_implementation="sdpa"
        )
        
        device_name = self.model.device
        logger.info(f"[INIT] Cognitive LLM successfully loaded on device: {device_name}")
        
        self.warmup_llm()

    def warmup_llm(self):
        dummy_prompt = "Warmup: System is detecting a high DRB_PdcpSduDelayDl anomaly. Procedures follow..."
        
        inputs = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.model.device)
        
        # Force a real generation of at least 50 tokens
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=50,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _generate(self, prompt: str) -> dict:
        msgs = [{"role": "system", "content": "You are a 5G Network AI. Output ONLY valid JSON. No markdown, no explanations."},
                {"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            msgs, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        prompt_length = inputs["input_ids"].shape[1]
        resp = self.tokenizer.decode(ids[0][prompt_length:], skip_special_tokens=True)
        
        logger.info(f"[LLM DEBUG] Raw output: {resp[:200]}...")
        
        try:
            # Find the first balanced JSON object in the output
            start = resp.find('{')
            if start == -1:
                logger.error(f"[COGNITIVE] No JSON found! Raw output: {resp}")
                return {}
            depth = 0
            end = start
            for i, ch in enumerate(resp[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            return json.loads(resp[start:end + 1])
        except Exception as e:
            logger.error(f"[COGNITIVE] JSON Parse Error: {e}. Raw output: {resp}")
            return {}

    async def run(self):
        q = await self.bus.sub("ai.request")
        logger.info("[COGNITIVE] Unified RAG/LLM Agent Online.")
        
        while True:
            msg = await q.get()
            payload = msg.payload
            
            prompt = ""
            
            # Telemetric anomaly
            if payload.get("type") == "generate_otm":
                metric = payload.get("metric")
                val = payload.get("value")
                
                # RAG from grounding layer (mock for now)
                kb_recommendations = self.kb.query_scenario(metric)
                rec_str = json.dumps(kb_recommendations) if kb_recommendations else "None."

                prompt = (
                    f"NETWORK ANOMALY: The metric '{metric}' has degraded to a value of {val}.\n"
                    f"KNOWLEDGE BASE PROCEDURES: {rec_str}\n\n"
                    f"TASK: Generate a strict OTM JSON to fix this issue. If the procedures suggest lowering power or fixing throughput, encode those as 'constraints'.\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Manual user request
            elif payload.get("type") == "manual":
                user_text = payload.get("text")
                prompt = (
                    f"USER REQUEST: \"{user_text}\"\n\n"
                    f"TASK: Translate this human intent into network constraints. \n"
                    f"Example: If they say 'make it energy efficient', set an objective to minimize 'tx_power_dbm' and a constraint for 'tx_power_dbm <= 12.0'.\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Adaptation after failed OTM
            elif payload.get("type") == "adapt_otm":
                metric = payload.get("metric")
                prev_otm = payload.get("previous_otm")
                prompt = (
                    f"ADAPTATION: Metric '{metric}' is still failing (current value: {payload.get('value')}).\n"
                    f"Previous OTM failed: {json.dumps(prev_otm)}\n\n"
                    f"TASK: Generate a NEW, stricter OTM JSON.\n"
                    f"IMPORTANT: You MUST document your changes! Add a string entry to the 'adaptation_log' array in the metadata explaining exactly what constraint you changed and why.\n"
                    f"Example adaptation_log:[\"Decreased dl_mcs_max threshold from 20 to 15 to prioritize stability over throughput.\"]\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            if not prompt:
                continue

            logger.info(f"[COGNITIVE] Prompting LLM.")
            
            loop = asyncio.get_running_loop()
            otm_json = await loop.run_in_executor(self.executor, self._generate, prompt)
            
            # Inject runtime variables the LLM can't know
            if otm_json and "objective" in otm_json:
                otm_json["metadata"]["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                otm_json["metadata"]["episode"] = f"alert_{msg.corr_id[:6]}"

                logger.info(f"[COGNITIVE] OTM Generated Successfully. Publishing to Optimizer. (Trace: {msg.corr_id})")

                await self.bus.pub("optimizer.target", make_msg("opt", "NEW_TARGET", "v1", otm_json, corr_id=msg.corr_id))
            else:
                logger.error("[COGNITIVE] LLM failed to generate a valid OTM. Notifying Orchestrator.")
                await self.bus.pub("optimizer.target", make_msg("opt", "LLM_FAILURE", "v1", {
                    "error": "LLM did not produce a valid OTM",
                    "request_type": payload.get("type"),
                }, corr_id=msg.corr_id))