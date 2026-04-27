import asyncio
import json
import re
import uuid
import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from core.bus.mem import MemBus
from core.bus.messages import make_msg

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Intent-feasibility vocabulary: only accept manual intents whose text hits
# at least one scenario keyword. Derived from the RDF scenarios in
# knowledge_base.py. Follows TMF IG1253 / 3GPP TR 28.312 intent validation
# pattern — reject out-of-vocabulary intents instead of scenario-guessing.
MANUAL_INTENT_VOCAB = {
    "latency", "delay", "congestion", "lag", "slow",
    "throughput", "bandwidth", "data rate", "speed",
    "bler", "block error", "error rate", "errors",
    "energy", "power", "tx power", "txpower", "green", "save",
    "upgrade", "maintenance", "drain",
    "mcs", "modulation",
    "drb_pdcpsdudelaydl", "ue_drb_pdcpsdudelaydl_ueid",
    "ue_drb_uethpdl_ueid", "ue_drb_blerdl_ueid",
    "tx_power_dbm", "dl_mcs_max", "bler_dl",
}


def _manual_intent_in_vocabulary(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kw in MANUAL_INTENT_VOCAB)


# Canonical constraint identities — the ONLY ids the LLM is allowed to
# adapt. Constraint identities flow from KB procedure catalog → orchestrator.
# The LLM's output is gated through _sanitize_llm_constraints: it may update
# thresholds of existing canonical ids (or add missing ones during merge),
# but any hallucinated id, old {parameter, value} shape, or non-actuatable
# kpi is discarded at the boundary.
CANONICAL_CONSTRAINT_IDS = {"PROC_MCS", "PROC_TXPOW"}
_ID_TO_KPI = {"PROC_MCS": "dl_mcs_max", "PROC_TXPOW": "tx_power_dbm"}
_ID_TO_OPERATOR = {"PROC_MCS": "le", "PROC_TXPOW": "ge"}
_ID_TO_UNIT = {"PROC_MCS": "", "PROC_TXPOW": "dBm"}
_KPI_TO_ID = {"dl_mcs_max": "PROC_MCS", "tx_power_dbm": "PROC_TXPOW"}


def _sanitize_llm_constraints(base_constraints: list, llm_constraints: list) -> list:
    """Rebuild the canonical constraint array from a trusted base, applying
    LLM-emitted threshold deltas only for recognized canonical ids.

    - Start from base_constraints (all canonical ids already in the active
      or previous OTM). Preserves shape including non-threshold fields.
    - For each LLM constraint: resolve id (from explicit 'id' or inferred
      from 'kpi'/'parameter'); resolve threshold ('threshold' or legacy
      'value'). Match by id → update threshold only. New canonical id →
      synthesize full canonical entry. Anything else → discarded.
    """
    result: list = []
    by_id: dict = {}
    for c in base_constraints or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if not cid or cid in by_id:
            continue
        entry = dict(c)
        entry.pop("applied_value", None)  # recompute per dispatch
        result.append(entry)
        by_id[cid] = entry

    for c in llm_constraints or []:
        if not isinstance(c, dict):
            continue
        raw_threshold = c.get("threshold", c.get("value"))
        if raw_threshold is None:
            continue
        try:
            new_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            continue

        cid = c.get("id")
        if not cid:
            cid = _KPI_TO_ID.get(c.get("kpi") or c.get("parameter"))
        if not cid:
            continue

        if cid in by_id:
            by_id[cid]["threshold"] = new_threshold
            by_id[cid]["adapted_by"] = "cognitive_llm"
            
            # The LLM must be able to change a floor to a ceiling
            if "operator" in c:
                by_id[cid]["operator"] = c["operator"]
                
        elif cid in CANONICAL_CONSTRAINT_IDS:
            synthesized = {
                "service": "mbb",
                "kpi": _ID_TO_KPI[cid],
                # Trust the LLM's operator, fallback to default only if missing
                "operator": c.get("operator", _ID_TO_OPERATOR[cid]),
                "threshold": new_threshold,
                "unit": _ID_TO_UNIT[cid],
                "id": cid,
                "origin": "cognitive_llm",
                "adapted_by": "cognitive_llm",
            }
            result.append(synthesized)
            by_id[cid] = synthesized
        # unknown id or non-actuatable kpi → discarded silently

    return result


OTM_SCHEMA_TEMPLATE = """
{
  "version": "1.0",
  "objective": { "service": "mbb", "kpi": "<MAIN_KPI>", "aggregation": "mean", "unit": "<UNIT>", "maximize": true_or_false },
  "constraints": [],
  "metadata": { "adaptation_log": [] }
}
"""

class CognitiveAgent:
    MAX_NEW_TOKENS = 16384

    def __init__(self, bus, kb, episode_store=None):
        self.bus = bus
        self.kb = kb
        self.episode_store = episode_store
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._model_lock = threading.Lock()  # Serialize all Llama model access
        self.model = None

        try:
            logger.info("[INIT] Downloading Qwen3.5-9B GGUF (8-bit Quantized)...")
            
            model_path = hf_hub_download(
                repo_id="unsloth/Qwen3.5-9B-GGUF", 
                filename="Qwen3.5-9B-Q8_0.gguf"
            )

            logger.info(f"[INIT] Loading 8-bit GGUF into 6800 XT VRAM via llama.cpp...")
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1, 
                n_ctx=16384,      
                verbose=False    
            )
            
        except Exception as e:
            logger.critical(f"[INIT] Failed to load LLM model: {e}")
            raise RuntimeError(f"LLM model is required but failed to load: {e}") from e

    def _generate(self, prompt: str) -> dict:
        if not self.model:
            return {}

        with self._model_lock:
            return self._generate_locked(prompt)

    def _generate_locked(self, prompt: str) -> dict:
        system_prompt = (
            "You are a 5G Network AI. Output ONLY valid JSON. No markdown, no explanations.\n"
            "ARCHITECTURAL ROLE:\n"
            "You do not set exact hardware values. You define the \"Feasible Search Space\" for a downstream mathematical optimizer.\n"
            "- To set a MAXIMUM limit (a ceiling), use the operator \"le\" (less than or equal to).\n"
            "- To set a MINIMUM limit (a floor), use the operator \"ge\" (greater than or equal to).\n"
            "The downstream optimizer will read your objective and pick a specific numerical value within the boundaries you set.\n"
            "If the optimizer picks a value that causes network failure, you must restrict its search space by lowering the ceiling (\"le\") or raising the floor (\"ge\").\n"
            "UNIT RULES (NEVER violate these):\n"
            "OBSERVABLE METRICS (use in objective or constraints):\n"
            "- Latency/Delay (DRB_PdcpSduDelayDl, UE_DRB_PdcpSduDelayDl_UEID): unit is ms, typical range 5-200. Example threshold: 50.0\n"
            "- Throughput (UE_DRB_UEThpDl_UEID): unit is kbps, typical range 1000-500000. Example threshold: 50000.0\n"
            "- BLER (UE_DRB_BlerDl_UEID): unitless ratio, range 0.0-1.0. Example threshold: 0.01\n"
            "- PRB utilization (RRU_PrbUsedDl): percentage, range 0-100. Example threshold: 80.0\n"
            "ACTUATABLE PARAMETERS (include these in constraints — they are applied directly to the network):\n"
            "- MCS cap (dl_mcs_max): integer index, range 0-28. Example threshold: 20\n"
            "- TX power (tx_power_dbm): unit is dBm, range 30-60. Example threshold: 46.0\n"
            "You MUST include at least one actuatable parameter in your constraints, otherwise no action will be taken.\n"
            "NEVER use throughput-scale numbers (millions) for latency thresholds or vice versa.\n"
            "CONSTRAINT IDENTIFIERS:\n"
            "When constraints from a previous Intent Specification are shown to you in the prompt, copy their `id` field verbatim and adjust only the threshold value. Never invent new constraint identifiers — they will be discarded by the downstream sanitiser."
        )
        
        # llama.cpp has built-in chat templating
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.MAX_NEW_TOKENS,
                temperature=0.0, # Greedy decoding for strict JSON schema adherence
            )
            
            resp_text = response["choices"][0]["message"]["content"]
            logger.info(f"[LLM DEBUG] Raw output: {resp_text}")
            
            # Find and parse the JSON from the llm output
            start = resp_text.find('{')
            if start == -1:
                return {}
            depth = 0
            end = start
            for i, ch in enumerate(resp_text[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            return json.loads(resp_text[start:end + 1])
            
        except Exception as e:
            logger.error(f"[COGNITIVE] JSON Parse or Generation Error: {e}")
            return {}

    def _generate_reflection(self, prompt: str) -> str:
        """Generate free-form text (not JSON) for self-reflection.

        Uses a different system prompt since reflections are verbal analysis,
        not structured OTM output. This is the Reflexion (Shinn et al., 2023)
        self-reflection actor: it analyzes WHY an episode succeeded or failed
        and produces actionable guidance for future episodes.
        """
        if not self.model:
            return ""

        with self._model_lock:
            return self._generate_reflection_locked(prompt)

    def _generate_reflection_locked(self, prompt: str) -> str:
        system_prompt = (
            "You are a 5G Network AI performing self-reflection on a recent network management episode. "
            "Analyze WHY the outcome occurred and WHAT should be done differently next time. "
            "Be concise (2-3 sentences). Focus on actionable operational insight, not generic advice. "
            "Output plain text only — no JSON, no markdown, no bullet points."
        )

        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.3,  # Slightly more creative than OTM generation
            )
            text = response["choices"][0]["message"]["content"].strip()
            logger.info(f"[COGNITIVE] Reflection generated: {text[:100]}...")
            return text
        except Exception as e:
            logger.error(f"[COGNITIVE] Reflection generation failed: {e}")
            return ""

    def _cache_system_prefix(self):
        # tokenize the fixed system prompt once so we can prepend it cheaply.
        sys_msg = [{"role": "system",
                     "content": "You are a 5G Network AI. Output ONLY valid JSON. No markdown, no explanations."}]
        ids = self.tokenizer.apply_chat_template(
            sys_msg, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )
        # Store on the correct device
        self._system_token_ids = ids["input_ids"].to(next(self.model.parameters()).device)
        logger.info(f"[INIT] System prefix cached: {self._system_token_ids.shape[1]} tokens")

    # not used with llama.cpp, but keeping for reference if I switch back to HuggingFace Transformers
    def _warmup_llm(self):
        # To make the first real inference faster the model is warmed up
        dummy = "Warmup: high DRB_PdcpSduDelayDl. Output a short JSON."
        for _ in range(2):
            self._generate(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info("[INIT] LLM warmup complete (2 passes)")


    @staticmethod
    def _parse_temporal(text: str, schedule_context: list = None) -> dict:
        """Deterministic temporal extraction from user text.

        Returns {"has_schedule": bool, "activate_at_utc": str|None, "deactivate_at_utc": str|None}.
        Handles patterns like:
          - "in 2 hours"  /  "in 30 minutes"
          - "at 02:30"  /  "at 14:00"
          - "tonight"  /  "this evening"
          - "tomorrow"  /  "tomorrow morning"  /  "tomorrow night"
          - "after that" / "immediately after" (relative to existing schedule)
          - "for X hours" (explicit duration override)
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        activate_at = None
        duration = None
        lower = text.lower()

        # --- Explicit duration override: "for X hours/minutes" ---
        dur_match = re.search(r'for\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)', lower)
        if dur_match:
            dur_amount = float(dur_match.group(1))
            dur_unit = dur_match.group(2)
            if dur_unit.startswith('h'):
                duration = datetime.timedelta(hours=dur_amount)
            else:
                duration = datetime.timedelta(minutes=dur_amount)

        # --- "in X hours/minutes/seconds" ---
        m = re.search(r'in\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)', lower)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
            if unit.startswith('h'):
                activate_at = now + datetime.timedelta(hours=amount)
            elif unit.startswith('m'):
                activate_at = now + datetime.timedelta(minutes=amount)
            else:
                activate_at = now + datetime.timedelta(seconds=amount)
            if duration is None:
                duration = datetime.timedelta(hours=1)

        # --- "at HH:MM" ---
        if activate_at is None:
            m = re.search(r'at\s+(\d{1,2}):(\d{2})', lower)
            if m:
                hour, minute = int(m.group(1)), int(m.group(2))
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate <= now:
                    candidate += datetime.timedelta(days=1)
                activate_at = candidate
                if duration is None:
                    duration = datetime.timedelta(hours=2)

        # --- "tonight" / "this evening" ---
        if activate_at is None and ("tonight" in lower or "this evening" in lower):
            activate_at = now.replace(hour=19, minute=0, second=0, microsecond=0)
            if activate_at <= now:
                activate_at += datetime.timedelta(days=1)
            if duration is None:
                duration = datetime.timedelta(hours=5)

        # --- "tomorrow night/evening" (must come before generic "tomorrow") ---
        if activate_at is None and re.search(r'tomorrow\s+(night|evening)', lower):
            activate_at = (now + datetime.timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
            if duration is None:
                duration = datetime.timedelta(hours=4)

        # --- "tomorrow morning" ---
        if activate_at is None and "tomorrow morning" in lower:
            activate_at = (now + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
            if duration is None:
                duration = datetime.timedelta(hours=4)

        # --- "tomorrow afternoon" ---
        if activate_at is None and "tomorrow afternoon" in lower:
            activate_at = (now + datetime.timedelta(days=1)).replace(hour=13, minute=0, second=0, microsecond=0)
            if duration is None:
                duration = datetime.timedelta(hours=4)

        # --- generic "tomorrow" ---
        if activate_at is None and "tomorrow" in lower:
            activate_at = (now + datetime.timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
            if duration is None:
                duration = datetime.timedelta(hours=8)

        # --- "after that" / "immediately after" / "then" (chain to existing schedule) ---
        if activate_at is None and schedule_context:
            if any(phrase in lower for phrase in ("after that", "immediately after", "following that")):
                # Find the latest deactivate_at from the existing schedule
                latest_end = None
                for item in schedule_context:
                    da = item.get("deactivate_at")
                    if da:
                        da_dt = datetime.datetime.fromisoformat(da)
                        if latest_end is None or da_dt > latest_end:
                            latest_end = da_dt
                if latest_end:
                    activate_at = latest_end
                    if duration is None:
                        duration = datetime.timedelta(hours=4)

        if activate_at is None:
            return {"has_schedule": False, "activate_at_utc": None, "deactivate_at_utc": None}

        deactivate_at = activate_at + duration if duration else None
        return {
            "has_schedule": True,
            "activate_at_utc": activate_at.isoformat(),
            "deactivate_at_utc": deactivate_at.isoformat() if deactivate_at else None,
        }

    async def _reflection_loop(self):
        """Separate loop for Reflexion self-reflections (Shinn et al., 2023).

        Runs on its own bus topic (ai.reflect) so reflections never block
        urgent OTM generation/adaptation on the main ai.request queue.
        Uses a dedicated ThreadPoolExecutor to avoid contending with OTM generation.
        """
        q_reflect = await self.bus.sub("ai.reflect")
        reflect_executor = ThreadPoolExecutor(max_workers=1)
        logger.info("[COGNITIVE] Reflection loop online (ai.reflect).")

        while True:
            msg = await q_reflect.get()
            payload = msg.payload
            episode = payload.get("episode", {})
            episode_id = payload.get("episode_id")
            anomaly = episode.get("anomaly", {})
            outcome = episode.get("outcome", {})
            otm = episode.get("otm_prescribed", {})

            # Canonical Reflexion: Msr is invoked when Me reports failure.
            if outcome.get("resolved"):
                logger.debug(f"[COGNITIVE] Skipping reflection for resolved episode {(episode_id or '')[:8]}.")
                
                # DEFENSIVE GUARD: Ensure we still unpause the Orchestrator even if we skip
                await self.bus.pub("ai.reflection_done", make_msg("cognitive", "DONE", "v1", {"episode_id": episode_id}))
                continue

            metric = anomaly.get("metric", "unknown")
            scope = anomaly.get("scope", {})
            val_before = anomaly.get("value", "?")
            val_after = outcome.get("metric_after", "?")

            constraints_str = (json.dumps(otm.get("constraints",[]), indent=2)
                               if otm.get("constraints") else "none")
            proc_id = otm.get("procedure_id", "none")

            # Pull prior reflections for the same task
            prior_reflections_block = ""
            if self.episode_store:
                priors = self.episode_store.get_recent_reflections(anomaly)
                if priors:
                    prior_reflections_block = (
                        "\nPrior reflections on this same task:\n"
                        + "\n".join(f"  - {r}" for r in priors)
                        + "\n"
                    )

            reflect_prompt = (
                f"You are a 5G network-management AI that improves itself by "
                f"self-reflecting on failed trials.\n\n"
                f"You were given an anomaly and your prescribed OTM FAILED to resolve it.\n\n"
                f"Task: metric={metric}, scope={scope}.\n"
                f"Trajectory:\n"
                f"  1. Observed anomaly: {metric} = {val_before}.\n"
                f"  2. Applied OTM procedure '{proc_id}' with constraints:\n"
                f"     {constraints_str}\n"
                f"  3. After execution, {metric} = {val_after} — still out of bounds.\n"
                f"{prior_reflections_block}\n"
                f"ACTION SPACE (the ONLY knobs this system exposes):\n"
                f"  - dl_mcs_max   : integer 0-28 (downlink MCS cap)\n"
                f"  - tx_power_dbm : number 30-60 (base-station transmit power, dBm)\n"
                f"Do NOT suggest scheduler changes, beamforming, handover tuning, "
                f"QoS class changes, or any other mechanism — they are not actuatable.\n\n"
                f"In 2-4 complete sentences, diagnose a plausible reason the trial failed "
                f"and propose a concrete, high-level plan phrased in terms of dl_mcs_max "
                f"and/or tx_power_dbm that should avoid the same failure next time."
            )

            loop = asyncio.get_running_loop()
            reflection_text = await loop.run_in_executor(
                reflect_executor, self._generate_reflection, reflect_prompt
            )

            if reflection_text and self.episode_store and episode_id:
                self.episode_store.add_reflection(episode_id, reflection_text)
                logger.info(f"[COGNITIVE] Reflexion stored for episode {episode_id[:8]}")
            elif not reflection_text:
                logger.warning(f"[COGNITIVE] Empty reflection for episode {episode_id[:8] if episode_id else '?'}")
            
            # Notify Orchestrator that the reflection is complete so it can un-pause and adapt
            await self.bus.pub("ai.reflection_done", make_msg("cognitive", "DONE", "v1", {"episode_id": episode_id}))

    
    async def run(self):
        q = await self.bus.sub("ai.request")

        # Launch reflection loop as independent task — never blocks OTM generation
        asyncio.create_task(self._reflection_loop())

        logger.info("Cognitive agent online.")

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

                # Reflexion tail-take: reflections are for THIS task
                # (metric, cell_id). No embedding similarity , see Shinn et
                # al. 2023 Algorithm 1 and episode_store.ReflexionMemory.
                experience_block = ""
                if self.episode_store:
                    anomaly_ctx = {
                        "metric": metric,
                        "value": val,
                        "scope": payload.get("scope") or {},
                    }
                    reflections = self.episode_store.get_recent_reflections(anomaly_ctx)
                    if reflections:
                        experience_block = (
                            f"\n{self.episode_store.format_reflections_for_prompt(reflections)}\n"
                            f"Use these prior reflections to avoid repeating past failures.\n\n"
                        )

                prompt = (
                    f"NETWORK ANOMALY: The metric '{metric}' has degraded to a value of {val}.\n"
                    f"KNOWLEDGE BASE PROCEDURES: {rec_str}\n\n"
                    f"{experience_block}"
                    f"TASK:\n"
                    f"1. Generate a strict OTM JSON to fix this issue using the procedures.\n"
                    f"2. WARNING: Latency/Delay is measured in ms (e.g., 40.0 to 80.0). Throughput is measured in kbps (e.g., 50000.0). DO NOT mix up these numbers!\n"
                    f"3. Keep the 'adaptation_log' extremely brief (max 1 sentence).\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Manual user request (also RAG from kb, and scheduling for future intents logic exists)
            elif payload.get("type") == "manual":
                user_text = payload.get("text") or ""

                if not _manual_intent_in_vocabulary(user_text):
                    logger.warning(f"[COGNITIVE] Manual intent REJECTED (out of KB vocabulary): '{user_text}'")
                    await self.bus.pub("ai.response", make_msg("opt", "LLM_FAILURE", "v1", {
                        "error": f"Manual intent out of KB vocabulary: '{user_text}'",
                        "request_type": "manual",
                        "reason": "intent_infeasible",
                    }, corr_id=msg.corr_id))
                    continue

                # Fetch the semantic knowledge from the RDF Graph
                rdf_context = self.kb.get_rdf_scenarios_for_llm()
                
                schedule_ctx = payload.get("schedule_context", [])
                schedule_block = ""
                if schedule_ctx:
                    schedule_block = (
                        f"\nCURRENT SCHEDULE QUEUE (already scheduled intents):\n"
                        f"{json.dumps(schedule_ctx, indent=2)}\n"
                        f"NOTE: 'after that'/'then' activates AFTER the most recent scheduled intent.\n\n"
                    )
                prompt = (
                    f"USER REQUEST: \"{user_text}\"\n\n"
                    f"KNOWLEDGE BASE (RDF Semantic Grounding):\n{rdf_context}\n\n"
                    f"{schedule_block}"
                    f"TASK:\n"
                    f"1. Read the Knowledge Base. Match the user's request to the most conceptually relevant SCENARIO.\n"
                    f"2. Use the 'OTM_INSTRUCTION' text from that Scenario to build the 'objective' and 'constraints' in your JSON.\n"
                    f"3. Set 'procedure_id' in the JSON metadata to the exact 'Mapped procedure_id' of that Scenario.\n"
                    f"4. Keep the 'adaptation_log' brief, motivate your reasoning. One or two sentences should suffice.\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Merge manual intents: user intent takes priority over the active OTM
            elif payload.get("type") == "merge_manual":
                active_otm = payload.get("active_otm", {})
                user_text = payload.get("text", "")

                if not _manual_intent_in_vocabulary(user_text):
                    logger.warning(f"[COGNITIVE] Manual intent REJECTED (out of KB vocabulary, active OTM preserved): '{user_text}'")
                    await self.bus.pub("ai.response", make_msg("opt", "LLM_FAILURE", "v1", {
                        "error": f"Manual intent out of KB vocabulary: '{user_text}'",
                        "request_type": "merge_manual",
                        "reason": "intent_infeasible",
                    }, corr_id=msg.corr_id))
                    continue

                clean_otm = {
                    "objective": active_otm.get("objective", {}),
                    "constraints": active_otm.get("constraints", [])
                }
                
                rdf_context = self.kb.get_rdf_scenarios_for_llm()
                
                schedule_ctx = payload.get("schedule_context", [])
                schedule_block = ""
                if schedule_ctx:
                    schedule_block = f"\nCURRENT SCHEDULE QUEUE:\n{json.dumps(schedule_ctx, indent=2)}\n\n"

                prompt = (
                    f"MERGE MANUAL REQUEST: Active OTM:\n{json.dumps(clean_otm, indent=2)}\n\n"
                    f"USER INTENT: \"{user_text}\"\n\n"
                    f"KNOWLEDGE BASE (RDF Semantic Grounding):\n{rdf_context}\n\n"
                    f"{schedule_block}"
                    f"TASK:\n"
                    f"1. Match the user's request to the most relevant SCENARIO in the Knowledge Base.\n"
                    f"2. Generate a NEW OTM JSON. The user's request takes priority. Use the OTM_INSTRUCTIONs from the chosen Scenario.\n"
                    f"3. You may preserve constraints from the Active OTM ONLY if they do not conflict with the new Scenario.\n"
                    f"4. Set 'procedure_id' in the JSON metadata to the 'Mapped procedure_id' of the chosen Scenario.\n"
                    f"5. Keep the 'adaptation_log' brief, motivate your reasoning. One or two sentences should suffice.\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Adaptation after failed OTM
            elif payload.get("type") == "adapt_otm":
                metric = payload.get("metric")
                prev_otm = payload.get("previous_otm")
                val = payload.get("value")

                # Reflexion tail-take for adaptation path (same task scoping
                # as generate_otm). The reflections were written after the
                # previous trial of this same (metric, cell_id) failed, which
                # is precisely when they are most relevant.
                experience_block = ""
                if self.episode_store:
                    anomaly_ctx = {
                        "metric": metric,
                        "value": val,
                        "scope": payload.get("scope") or {},
                    }
                    reflections = self.episode_store.get_recent_reflections(anomaly_ctx)
                    if reflections:
                        experience_block = (
                            f"\n{self.episode_store.format_reflections_for_prompt(reflections)}\n"
                            f"Apply these reflections to pick a different adjustment this time.\n\n"
                        )

                # Show APPLIED values (what actually went to ns-3), not just thresholds.
                # The constraint executor writes 'applied_value' into each constraint
                # after objective-biased parameter selection. We also include the
                # canonical id in the rendering so the LLM can copy it forward
                # rather than inventing one.
                prev_constraints = prev_otm.get("constraints", [])
                constraints_for_llm = []
                for c in prev_constraints:
                    cid = c.get("id", "UNKNOWN")
                    entry = f"id={cid}: {c.get('kpi')} {c.get('operator')} {c.get('threshold')}"
                    applied = c.get("applied_value")
                    if applied is not None and applied != c.get("threshold"):
                        entry += f" (actual applied: {applied})"
                    constraints_for_llm.append(entry)
                constraints_str = "\n".join(f"  - {e}" for e in constraints_for_llm) if constraints_for_llm else "none"

                prev_log_entries = prev_otm.get("metadata", {}).get("adaptation_log", [])
                log_context = "\n".join([f" - {entry}" for entry in prev_log_entries])
                if not log_context:
                    log_context = " - No previous adaptations."

                prompt = (
                    f"ADAPTATION REQUIRED: The network procedure paused because metric '{metric}' failed a health check or is still deviating (current value: {val}).\n"
                    f"Active OTM Constraints that failed to fix this:\n{constraints_str}\n\n"
                    f"PREVIOUS ADAPTATION HISTORY:\n{log_context}\n\n"
                    f"NOTE: 'actual applied' shows the real parameter value sent to the network after objective-aware biasing. "
                    f"Your new threshold will also be biased — set it HIGHER than your target if the bias lowers it, or LOWER if the bias raises it.\n\n"
                    f"{experience_block}"
                    f"TASK:\n"
                    f"1. Make incremental adjustments to the constraint thresholds.\n"
                    f"2. MANDATORY: At least one constraint's threshold value MUST differ from the value shown in the active OTM above. Returning the same thresholds unchanged is treated as a refusal to adapt and the system will escalate; even a small numeric shift (e.g. ±1 for MCS, ±2 dBm for TX power) is preferable to repeating the previous values.\n"
                    f"3. Generate an updated OTM JSON with adjusted constraints.\n"
                    f"4. WARNING: Latency/Delay is measured in ms (e.g., 40.0 to 80.0). Throughput is measured in kbps (e.g., 50000.0). DO NOT mix up these numbers!\n"
                    f"5. CRITICAL: Preserve the `id` field of each existing constraint exactly as shown above. Do not invent new id values; the orchestrator updates constraints in place by matching on id, so any unrecognised id will be silently dropped by the sanitiser. Modify only the threshold value, not the id, kpi, operator, or unit fields.\n"
                    f"6. Keep the 'adaptation_log' brief (MAXIMUM 2 short sentences summarizing the change).\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )
            
            # Procedure step failed 3 times in a row, pick a new scenario. 
            elif payload.get("type") == "scenario_failed":
                failed_procedure = payload.get("failed_procedure")
                metric = payload.get("metric")
                val = payload.get("value")
                rdf_context = self.kb.get_rdf_scenarios_for_llm()
                
                prompt = (
                    f"ESCALATION: The network metric '{metric}' is in a critical state (value: {val}).\n"
                    f"We attempted the '{failed_procedure}' procedure, but it repeatedly failed health checks and was aborted.\n\n"
                    f"KNOWLEDGE BASE (RDF Semantic Grounding):\n{rdf_context}\n\n"
                    f"TASK:\n"
                    f"1. You MUST select a COMPLETELY DIFFERENT scenario from the Knowledge Base to fix '{metric}'. Do not select '{failed_procedure}' again.\n"
                    f"2. Use the 'OTM_INSTRUCTION' text from the NEW Scenario to build the 'objective'. Leave constraints empty.\n"
                    f"3. Set 'procedure_id' in the metadata to the new Scenario's ID.\n"
                    f"4. In the 'adaptation_log', state that you are switching strategies because the previous procedure failed, what new scenario you chose and why. Keep it brief, max two sentences.\n\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            # Merge: anomaly arrived while a user/proactive OTM was active
            elif payload.get("type") == "merge_otm":
                active_otm = payload.get("active_otm", {})
                anomaly = payload.get("anomaly", {})
                anomaly_metric = anomaly.get("metric", "unknown")
                anomaly_value = anomaly.get("value", "unknown")

                # Fetch procedures for the new anomaly
                kb_recommendations = self.kb.query_scenario(anomaly_metric)
                rec_str = json.dumps(kb_recommendations) if kb_recommendations else "None."

                # Reflexion take on the incoming anomaly's task, same
                # semantics as generate_otm/adapt_otm. Merging is still a
                # reactive response to the anomaly so prior reflections on
                # this (metric, cell_id) are just as relevant.
                experience_block = ""
                if self.episode_store:
                    anomaly_ctx = {
                        "metric": anomaly_metric,
                        "value": anomaly_value,
                        "scope": anomaly.get("scope") or {},
                    }
                    reflections = self.episode_store.get_recent_reflections(anomaly_ctx)
                    if reflections:
                        experience_block = (
                            f"\n{self.episode_store.format_reflections_for_prompt(reflections)}\n"
                            f"Use these reflections when choosing the constraints added for the new anomaly.\n\n"
                        )

                prompt = (
                    f"MERGE REQUEST: A network anomaly has occurred on '{anomaly_metric}' "
                    f"(current value: {anomaly_value}) while the following OTM is active:\n"
                    f"{json.dumps(active_otm)}\n\n"
                    f"KNOWLEDGE BASE PROCEDURES FOR NEW ANOMALY: {rec_str}\n\n"
                    f"{experience_block}"
                    f"TASK: Generate a MERGED OTM JSON that handles both the active OTM and the new anomaly.\n"
                    f"1. PRESERVE the original objective and existing actuatable constraints from the active OTM if possible. When preserving a constraint, copy its `id` field verbatim from the active OTM shown above — do not rename it.\n"
                    f"2. You may adjust the THRESHOLD VALUE of an existing constraint to address the '{anomaly_metric}' anomaly, using the Knowledge Base procedures as a guide. Keep the id, kpi, operator, and unit fields exactly as they appear in the active OTM.\n"
                    f"3. Do NOT add observational metrics (like Latency or Throughput) to the constraints array. Only adjust parameters the system can actuate (dl_mcs_max, tx_power_dbm).\n"
                    f"4. If a constraint adjustment conflicts with the original (e.g., both could be argued for a different MCS cap), prioritize the MORE CONSERVATIVE value (lower MCS ceiling, higher TX power floor) to ensure stability.\n"
                    f"5. Keep the 'adaptation_log' brief, motivate your reasoning. One or two sentences should suffice.\n"
                    f"STRICT SCHEMA TO FOLLOW:\n{OTM_SCHEMA_TEMPLATE}"
                )

            if not prompt:
                continue

            logger.debug("\n" + "="*50)
            logger.debug(f"[LLM PROMPT - {payload.get('type')}]")
            logger.debug(prompt)
            logger.debug("="*50 + "\n")

            logger.info(f"[COGNITIVE] Prompting LLM.")
            loop = asyncio.get_running_loop()
            otm_json = await loop.run_in_executor(self.executor, self._generate, prompt)

            # Inject runtime variables about the metadata that the LLM can't know
            if otm_json and "objective" in otm_json:
                otm_json.setdefault("metadata", {})
                otm_json["metadata"]["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                otm_json["metadata"]["episode"] = f"alert_{msg.corr_id[:6]}"

                req_type = payload.get("type")
                source_tag = "cognitive_llm"

                # Constraint gateway: the LLM does NOT own constraint identity.
                # The procedure catalog (applied by orchestrator._merge_step_into_otm)
                # is the sole authority on canonical constraint shape and ids.
                # Here we only allow threshold deltas on recognized canonical ids
                # to pass through; hallucinated shapes/ids/kpis are discarded.
                if req_type in ("adapt_otm", "merge_otm", "merge_manual"):
                    base_src = payload.get("previous_otm") or payload.get("active_otm") or {}
                    base_constraints = base_src.get("constraints", []) if isinstance(base_src, dict) else []
                    llm_constraints = otm_json.get("constraints", [])
                    if not isinstance(llm_constraints, list):
                        llm_constraints = []
                    otm_json["constraints"] = _sanitize_llm_constraints(base_constraints, llm_constraints)
                elif req_type in ("generate_otm", "manual", "scenario_failed"):
                    # Orchestrator fills these from the procedure catalog via
                    # _merge_step_into_otm when activating the procedure.
                    otm_json["constraints"] = []

                # Backfill origin tag where not already set (preserved from base)
                for c in otm_json.get("constraints", []):
                    if not c.get("origin"):
                        c["origin"] = source_tag

                # Carry forward previous adaptation_log and append system entry
                prev_log = []
                if req_type == "adapt_otm":
                    prev_log = payload.get("previous_otm", {}).get("metadata", {}).get("adaptation_log", [])
                elif req_type in ("merge_otm", "merge_manual"):
                    prev_log = payload.get("active_otm", {}).get("metadata", {}).get("adaptation_log", [])

                # ensure current_log is a list before iterating
                current_log = otm_json["metadata"].get("adaptation_log", [])
                if isinstance(current_log, str):
                    current_log = [current_log]
                elif not isinstance(current_log, list):
                    current_log = []

                # same sanitization for prev_log — if LLM upstream emitted a bare string,
                # list(prev_log) would iterate chars into entries
                if isinstance(prev_log, str):
                    prev_log = [prev_log]
                elif not isinstance(prev_log, list):
                    prev_log = []

                # Write the sanitized list back to the JSON just to be extra safe
                otm_json["metadata"]["adaptation_log"] = current_log

                merged_log = []
                for entry in list(prev_log) + list(current_log):
                    if not isinstance(entry, str) or len(entry) < 3:
                        continue
                    if entry not in merged_log:
                        merged_log.append(entry)

                now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if req_type == "generate_otm":
                    merged_log.append(f"[{now_str}] Generated new OTM for {payload.get('metric')} via {source_tag}")
                elif req_type == "adapt_otm":
                    merged_log.append(f"[{now_str}] Adapted OTM for {payload.get('metric')} via {source_tag}")
                elif req_type == "merge_otm":
                    anomaly = payload.get("anomaly", {})
                    merged_log.append(
                        f"[{now_str}] Merged: added constraint for {anomaly.get('metric')} "
                        f"(value: {anomaly.get('value')}) into active OTM via {source_tag}"
                    )
                elif req_type == "merge_manual":
                    merged_log.append(
                        f"[{now_str}] Merged manual intent '{payload.get('text', '')}' into active OTM "
                        f"via {source_tag} — manual intent takes priority as primary objective"
                    )
                elif req_type == "manual":
                    merged_log.append(f"[{now_str}] Manual intent OTM generated via {source_tag}")

                otm_json["metadata"]["adaptation_log"] = merged_log

                # procedure_id is a control-plane identifier — the LLM must not own it.
                # Overwrite whatever the LLM emitted with the authoritative value for every req_type.
                if req_type == "generate_otm":
                    # SPARQL on the RDF graph is the single source of truth for
                    # metric → procedure_id. The catalog dict's trigger_metrics
                    # field is incomplete for some metrics (e.g. throughput KPIs),
                    # so match_scenario_by_metric would wrongly return None here.
                    metric = payload.get("metric", "")
                    matched = self.kb.resolve_procedure_id(metric) if metric else None
                    otm_json["metadata"]["procedure_id"] = matched  # may be None if no match
                    if matched:
                        merged_log.append(
                            f"[{now_str}] Anomaly on {metric} matched to procedure: "
                            f"{self.kb.get_scenario_name(matched)} (id={matched})"
                        )
                elif req_type in ("adapt_otm", "merge_otm"):
                    src = payload.get("previous_otm") or payload.get("active_otm") or {}
                    otm_json["metadata"]["procedure_id"] = (
                        src.get("metadata", {}).get("procedure_id") if isinstance(src, dict) else None
                    )
                elif req_type in ("manual", "merge_manual"):
                    # Manual intents may legitimately target a KB scenario.
                    # Validation order: trust LLM's id if it resolves, else fall
                    # back to deterministic keyword match on the user text
                    # (symmetric to generate_otm's SPARQL fallback).
                    llm_pid = otm_json.get("metadata", {}).get("procedure_id")
                    resolved_pid = None
                    if llm_pid and self.kb.get_procedure_steps(llm_pid):
                        resolved_pid = llm_pid
                    else:
                        kw_match = self.kb.match_scenario_by_keywords(payload.get("text", ""))
                        if kw_match and self.kb.get_procedure_steps(kw_match):
                            resolved_pid = kw_match
                    otm_json["metadata"]["procedure_id"] = resolved_pid

                # Temporal scheduling (for manual user intents), only one scheduled event at a time right now to keep it simple. 
                if payload.get("type") in ("manual", "merge_manual"):
                    user_text = payload.get("text", "")
                    schedule_ctx = payload.get("schedule_context", [])
                    otm_json["metadata"]["original_text"] = user_text
                    otm_json["metadata"]["temporal_resolved"] = self._parse_temporal(user_text, schedule_ctx)
                    temporal = otm_json["metadata"]["temporal_resolved"]
                    if temporal["has_schedule"]:
                        logger.info(f"[COGNITIVE] Temporal intent detected: activate={temporal['activate_at_utc']}, "
                                    f"deactivate={temporal['deactivate_at_utc']}")

                logger.info(f"[COGNITIVE] OTM Generated Successfully. Publishing to Optimizer. (Trace: {msg.corr_id})")

                await self.bus.pub("ai.response", make_msg("opt", "NEW_TARGET", "v1", otm_json, corr_id=msg.corr_id))
            else:
                logger.error("[COGNITIVE] LLM failed to generate a valid OTM. Notifying Orchestrator.")
                await self.bus.pub("ai.response", make_msg("opt", "LLM_FAILURE", "v1", {
                    "error": "LLM did not produce a valid OTM",
                    "request_type": payload.get("type"),
                }, corr_id=msg.corr_id))