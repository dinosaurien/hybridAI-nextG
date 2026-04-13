import asyncio
import copy
import json
import time
import logging
import uuid as _uuid
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional
from core.bus.mem import MemBus
from core.bus.messages import make_msg
from core.utils.knowledge_base import KnowledgeBase
from core.common.types import ScheduledIntent, ScheduleState, ActiveProcedure, ProcedureState, ProcedureStep

logger = logging.getLogger(__name__)

class IntentState(Enum):
    MONITORING = auto()
    THINKING = auto()
    ASSURANCE = auto()  # 30s grace period to let RL settle
    WITHDRAWAL = auto() # Evaluate whether the OTM resolved the deviation

class OrchestratorAgent:
    THINKING_TIMEOUT = 75.0   # seconds before assuming LLM failed
    WITHDRAWAL_WINDOW = 10.0  # seconds to evaluate post-assurance telemetry

    def __init__(self, bus: MemBus, kb: KnowledgeBase):
        self.bus = bus
        self.kb = kb
        self.state = IntentState.MONITORING
        self.thinking_start_time = 0.0
        self.assurance_end_time = 0.0
        self.withdrawal_end_time = 0.0
        self.active_otm = None
        self.active_anomaly_metric = None
        self.pending_manual_intents = []
        self.e2_acks_received = 0
        self.last_metric_value = None
        self.schedule_queue: List[ScheduledIntent] = []
        self.active_procedure: Optional[ActiveProcedure] = None
        self.pending_cancel_revert = None  # OTM to revert to after graceful cancel completes
        self.last_kpi_snapshot: dict = {}  # Full latest telemetry for health checks

        # Evaluation event log, append-only JSONL for later analysis of orchestration logic
        self._event_log_path = Path("models/orchestrator_events.jsonl")
        self._event_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event_type: str, **data):
        """Append a structured event to the orchestrator event log."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "state": self.state.name,
            **data,
        }
        try:
            with open(self._event_log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass  # Never crash for logging

    async def run(self):
        q_dev = await self.bus.sub("deviation.detected")
        q_otm = await self.bus.sub("ai.response")
        q_ui  = await self.bus.sub("ui.input")
        q_cmd = await self.bus.sub("command.notify")
        q_kpi = await self.bus.sub("kpi.raw")
        q_cancel = await self.bus.sub("schedule.cancel")

        logger.info("[ORCHESTRATOR] Orchestrator Online. Monitoring for Anomalies & Manual Intents.")

        while True:
            current_time = time.time()

            # Drain E2 acks (track that commands were dispatched)
            while not q_cmd.empty():
                q_cmd.get_nowait()
                self.e2_acks_received += 1

            # Track latest value of the anomaly metric from telemetry
            while not q_kpi.empty():
                kpi_msg = q_kpi.get_nowait()
                raw = kpi_msg.payload.get("kpi", kpi_msg.payload)
                self.last_kpi_snapshot = raw  # Full snapshot for health checks
                if self.active_anomaly_metric:
                    cell = raw.get("CellMetrics", {})
                    val = cell.get(self.active_anomaly_metric)
                    if val is not None:
                        self.last_metric_value = float(val)

            # Timeout: if LLM hasn't responded, unlock the loop
            if self.state == IntentState.THINKING and current_time >= self.thinking_start_time + self.THINKING_TIMEOUT:
                logger.warning("[ORCHESTRATOR] THINKING timeout reached — Cognitive Core did not respond. Returning to MONITORING.")
                self.state = IntentState.MONITORING

            # ASSURANCE -> WITHDRAWAL transition
            if self.state == IntentState.ASSURANCE and current_time >= self.assurance_end_time:
                if self.e2_acks_received == 0 and not (
                    self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE
                ):
                    logger.warning("[ORCHESTRATOR] Assurance expired but NO E2 commands were dispatched. Returning to MONITORING.")
                    self.state = IntentState.MONITORING
                else:
                    logger.info(f"[ORCHESTRATOR] Assurance expired ({self.e2_acks_received} E2 commands dispatched). Entering WITHDRAWAL to evaluate outcome.")
                    self.state = IntentState.WITHDRAWAL
                    self.withdrawal_end_time = current_time + self.WITHDRAWAL_WINDOW

            # WITHDRAWAL evaluation
            if self.state == IntentState.WITHDRAWAL and current_time >= self.withdrawal_end_time:
                resolved = self._evaluate_withdrawal()
                if resolved:
                    # If a procedure is active, advance to the next step instead of MONITORING
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE:
                        self.active_procedure.step_failures = 0
                        advanced = await self._advance_procedure()
                        if not advanced:
                            # No more steps, procedure is complete
                            await self._complete_procedure()
                            self.state = IntentState.MONITORING
                        # else: _advance_procedure already set state to ASSURANCE
                    else:
                        logger.info("[ORCHESTRATOR] WITHDRAWAL: Deviation resolved. Returning to MONITORING.")
                        self.state = IntentState.MONITORING
                else:
                    # Check procedure step failure limit
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE:
                        self.active_procedure.step_failures += 1
                        if self.active_procedure.step_failures >= 3:
                            logger.warning(f"[ORCHESTRATOR] Procedure step failed 3 times. Rolling back procedure.")
                            await self._rollback_procedure()
                            continue

                    logger.warning("[ORCHESTRATOR] WITHDRAWAL: Deviation persists. Re-engaging Cognitive Core for adaptation.")
                    self.state = IntentState.THINKING
                    self.thinking_start_time = current_time
                    self.e2_acks_received = 0
                    metric = self.active_anomaly_metric
                    await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "adapt_otm", "metric": metric,
                        "value": self.last_metric_value,
                        "previous_otm": self.active_otm
                    }))

            # Graceful cancel revert: apply when system settles back to MONITORING
            if self.state == IntentState.MONITORING and self.pending_cancel_revert is not None:
                await self._apply_cancel_revert()

            while not q_ui.empty():
                msg = q_ui.get_nowait()
                text = msg.payload.get("text", "").upper()

                if "BREAK_SIM" in text:
                    action = "trigger-traffic-spike" if "LATENCY" in text else "trigger-blockage"
                    logger.info(f"[ORCHESTRATOR] Routing manual chaos command: {text} -> {action}")
                    await self.bus.pub("sim.control", make_msg("orch", "CONTROL", "v1", {"action": action}))
                else:
                    # If a procedure is actively executing, queue the manual intent
                    # to avoid orphaning the procedure mid-step
                    if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE and \
                       self.state in (IntentState.ASSURANCE, IntentState.WITHDRAWAL):
                        logger.info(f"[ORCHESTRATOR] Procedure '{self.active_procedure.scenario_name}' active. "
                                    f"Queuing manual intent: '{text}'")
                        self.pending_manual_intents.append(text)
                    elif self.state in (IntentState.MONITORING, IntentState.ASSURANCE, IntentState.WITHDRAWAL):
                        self.state = IntentState.THINKING
                        self.thinking_start_time = current_time
                        self.e2_acks_received = 0
                        await self._route_manual_intent(text)
                    else:
                        logger.info(f"[ORCHESTRATOR] Busy ({self.state.name}). Queuing manual intent: '{text}'")
                        self.pending_manual_intents.append(text)
                
            # Process queued manual intents the moment we become idle.
            # Placed BEFORE the anomaly queue (q_dev) so human requests take priority.
            if self.state == IntentState.MONITORING and self.pending_manual_intents:
                queued_text = self.pending_manual_intents.pop(0)
                self.state = IntentState.THINKING
                self.thinking_start_time = current_time
                self.e2_acks_received = 0
                await self._route_manual_intent(queued_text)

            # Handle schedule cancellation requests from the UI
            while not q_cancel.empty():
                cancel_msg = q_cancel.get_nowait()
                cancel_id = cancel_msg.payload.get("schedule_id")
                if cancel_id:
                    await self._cancel_scheduled_intent(cancel_id)

            while not q_dev.empty():
                msg = q_dev.get_nowait()
                dev = msg.payload
                metric = dev['metric']

                # Suppress anomaly interrupts while a procedure is actively executing —
                # the procedure IS the structured response to this anomaly
                if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE and \
                   self.state in (IntentState.ASSURANCE, IntentState.WITHDRAWAL):
                    logger.debug(f"[ORCHESTRATOR] Suppressing anomaly on {metric} — procedure "
                                 f"'{self.active_procedure.scenario_name}' is handling it.")
                    continue

                if self.state == IntentState.MONITORING:
                    logger.info(f"[ORCHESTRATOR] Anomaly Detected: {metric} (state: {self.state.name}).")
                    self._log_event("anomaly_detected",
                                    metric=metric, value=dev.get('value'),
                                    severity=dev.get('severity', 'unknown'))
                    self.state = IntentState.THINKING
                    self.thinking_start_time = current_time
                    self.e2_acks_received = 0

                    await self.bus.pub("deviation.broadcast", make_msg("orch", "DEV", "v1", dev, corr_id=msg.corr_id))

                    if self.active_otm and self.active_anomaly_metric == metric:
                        # Same metric failing again: tighten existing constraints
                        logger.warning(f"[ORCHESTRATOR] Persistent failure on {metric}. Requesting ADAPTATION. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "adapt_otm", "metric": metric, "value": dev['value'], "previous_otm": self.active_otm
                        }, corr_id=msg.corr_id))
                    elif self.active_otm:
                        # Different metric or manual OTM active: merge anomaly into active OTM
                        logger.info(f"[ORCHESTRATOR] Merging anomaly on {metric} into active OTM. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "merge_otm",
                            "active_otm": self.active_otm,
                            "anomaly": {
                                "metric": metric,
                                "value": dev['value'],
                                "target": dev.get('target', 40.0),
                                "direction": dev.get('direction', 'lower_better'),
                                "severity": dev.get('severity', 'medium'),
                            }
                        }, corr_id=msg.corr_id))
                    else:
                        # No active OTM: generate fresh
                        logger.info(f"[ORCHESTRATOR] Requesting NEW OTM from Cognitive Core. (Trace: {msg.corr_id})")
                        await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "generate_otm", "metric": metric, "value": dev['value']
                        }, corr_id=msg.corr_id))
                    self.active_anomaly_metric = metric
                    break
                else:
                    logger.debug(f"[ORCHESTRATOR] Busy ({self.state.name}). Ignoring anomaly on {metric}")

            while not q_otm.empty():
                msg = q_otm.get_nowait()
                if msg.type == "LLM_FAILURE":
                    logger.warning(f"[ORCHESTRATOR] Cognitive Core failed to generate OTM "
                                   f"(type={msg.payload.get('request_type')}). Returning to MONITORING.")
                    self._log_event("llm_generation_failed",
                                    error=msg.payload.get("error"),
                                    request_type=msg.payload.get("request_type"))
                    self.state = IntentState.MONITORING

                else:
                    otm = msg.payload
                    temporal = otm.get("metadata", {}).get("temporal_resolved", {})

                    if temporal.get("has_schedule"):
                        activate_at = datetime.fromisoformat(temporal["activate_at_utc"])
                        now_utc = datetime.now(timezone.utc)

                        if activate_at > now_utc + timedelta(seconds=5):
                            # Future intent: enqueue and return to monitoring
                            await self._enqueue_scheduled_intent(otm, msg.corr_id)
                            self.state = IntentState.MONITORING
                        else:
                            # Activate immediately but track deactivation
                            if temporal.get("deactivate_at_utc"):
                                si = await self._enqueue_scheduled_intent(otm, msg.corr_id)
                                si.state = ScheduleState.ACTIVE
                                si.pre_intent_otm = self.active_otm
                            self._apply_otm(otm)
                    else:
                        self._apply_otm(otm)

            await self._tick_scheduler()
            await asyncio.sleep(0.2)

    #  Helpers

    async def _route_manual_intent(self, text: str):
        """Route a manual intent — merge with active OTM if one exists, else fresh."""
        # Clear stale anomaly metric so WITHDRAWAL evaluates the manual intent's own metric
        self.active_anomaly_metric = None
        self.last_metric_value = None
        schedule_ctx = self._get_schedule_context()
        if self.active_otm:
            logger.info(f"[ORCHESTRATOR] Merging manual intent '{text}' with active OTM (manual takes priority).")
            await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                "type": "merge_manual", "text": text, "active_otm": self.active_otm,
                "schedule_context": schedule_ctx,
            }))
        else:
            logger.info(f"[ORCHESTRATOR] Manual User Intent: '{text}' -> Sending to AI")
            await self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                "type": "manual", "text": text,
                "schedule_context": schedule_ctx,
            }))

    def _get_schedule_context(self) -> list:
        """Build a summary of PENDING/ACTIVE scheduled intents for LLM context."""
        return [
            {
                "raw_text": si.raw_text,
                "state": si.state.name,
                "activate_at": si.activate_at.isoformat(),
                "deactivate_at": si.deactivate_at.isoformat() if si.deactivate_at else None,
            }
            for si in self.schedule_queue
            if si.state in (ScheduleState.PENDING, ScheduleState.ACTIVE)
        ]

    def _apply_otm(self, otm: dict):
        """Common path: activate an OTM immediately (ASSURANCE window)."""
        self.active_otm = otm
        self.state = IntentState.ASSURANCE
        self.assurance_end_time = time.time() + 30.0
        self.e2_acks_received = 0

        # Initialize procedure queue if this OTM is the first step of a procedure
        procedure_id = otm.get("metadata", {}).get("procedure_id")
        if procedure_id:
            # If a different procedure is already active, cancel it first
            if self.active_procedure and self.active_procedure.state == ProcedureState.ACTIVE \
               and self.active_procedure.scenario_id != procedure_id:
                prev_name = self.active_procedure.scenario_name
                self.active_procedure.state = ProcedureState.COMPLETED
                now_str = datetime.now(timezone.utc).isoformat()
                cancel_msg = (f"[{now_str}] Procedure '{prev_name}' superseded by new procedure '{procedure_id}'.")
                logger.info(f"[ORCHESTRATOR] {cancel_msg}")
                self._append_to_active_log(cancel_msg)
                self.active_procedure = None

            if self.active_procedure is None:
                steps = self.kb.get_procedure_steps(procedure_id)
                if steps:
                    self.active_procedure = ActiveProcedure(
                        scenario_id=procedure_id,
                        scenario_name=self.kb.get_scenario_name(procedure_id),
                        steps=steps,
                        current_step_idx=0,
                        pre_procedure_otm=copy.deepcopy(self.active_otm),
                    )
                    
                    first_step = steps[0]
                    self._merge_step_into_otm(first_step)
                    # forward to optimizer
                    asyncio.create_task(self._publish_procedure_update("step_activated", first_step))
                    
                    logger.info(f"[ORCHESTRATOR] Procedure '{self.active_procedure.scenario_name}' activated. Entering ASSURANCE for step 0.")

        logger.info("[ORCHESTRATOR] OTM Received. Entering 30s ASSURANCE window.")
        self._log_event("otm_applied",
                        otm=otm,
                        anomaly_metric=self.active_anomaly_metric,
                        metric_value=self.last_metric_value,
                        procedure_id=otm.get("metadata", {}).get("procedure_id"))

        asyncio.create_task(self.bus.pub("intent.execute", make_msg("orch", "NEW_TARGET", "v1", self.active_otm)))

    def _evaluate_withdrawal(self) -> bool:
        """Check whether the anomaly metric has recovered.

        Uses the KnowledgeBase health thresholds as the success criteria.
        Returns True if the deviation is resolved, False if it persists.
        """
        if self.last_metric_value is None or self.active_anomaly_metric is None:
            resolved = True
        else:
            thresholds = self.kb.get_health_thresholds("CELL_001")
            metric = self.active_anomaly_metric

            if "delay" in metric.lower() or "latency" in metric.lower():
                limit = thresholds.get("latency_max_ms", 50.0)
                resolved = self.last_metric_value <= limit
            elif "thp" in metric.lower() or "throughput" in metric.lower():
                limit = thresholds.get("throughput_min_mbps", 5.0) * 1e6
                resolved = self.last_metric_value >= limit
            else:
                resolved = False

        self._log_event("withdrawal_eval",
                        resolved=resolved,
                        metric=self.active_anomaly_metric,
                        value=self.last_metric_value,
                        e2_acks=self.e2_acks_received)
        return resolved

    #  Procedure Queue
    async def _advance_procedure(self) -> bool:
        """Advance to the next executable step. Returns True if a step was activated, False if done."""
        proc = self.active_procedure
        if not proc or proc.state != ProcedureState.ACTIVE:
            return False

        now_str = datetime.now(timezone.utc).isoformat()

        while proc.current_step_idx + 1 < len(proc.steps):
            proc.current_step_idx += 1
            step = proc.steps[proc.current_step_idx]

            # Skip branch-only steps in normal linear flow
            if step.branch_only and not proc.branched_to:
                skip_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' SKIPPED: "
                            f"branch-only (not reached via conditional jump)")
                logger.info(f"[ORCHESTRATOR] {skip_msg}")
                self._append_to_active_log(skip_msg)
                await self._publish_procedure_update("step_skipped", step)
                continue

            # Clear branch flag only after consuming a non-branch-only step
            if not step.branch_only:
                proc.branched_to = False

            # Health check step: evaluate immediately against latest telemetry
            if step.health_check:
                passed, details = self._evaluate_health_check(step)
                event = "health_check_passed" if passed else "health_check_failed"
                await self._publish_procedure_update(event, step)
                if passed:
                    ok_msg = (f"[{now_str}] Health check '{step.name}' PASSED: {details}")
                    logger.info(f"[ORCHESTRATOR] {ok_msg}")
                    self._append_to_active_log(ok_msg)
                    proc.step_failures = 0  # Reset failures on success
                    continue  # Advance to next step
                else:
                    fail_msg = (f"[{now_str}] Health check '{step.name}' FAILED: {details}")
                    logger.warning(f"[ORCHESTRATOR] {fail_msg}")
                    self._append_to_active_log(fail_msg)
                    
                    # Try to branch if the user explicitly defined a mechanical fallback
                    if step.on_fail_goto:
                        target_idx = self._find_step_index_by_name(proc, step.on_fail_goto)
                        if target_idx is not None:
                            branch_msg = (f"[{now_str}] Branching to alternative step '{step.on_fail_goto}'")
                            logger.info(f"[ORCHESTRATOR] {branch_msg}")
                            self._append_to_active_log(branch_msg)
                            proc.current_step_idx = target_idx - 1  # -1 because loop increments
                            proc.branched_to = True
                            continue
                    
                    # If we have tried to adapt 3 times for this step, escalate to Cognitive Core for a scenario switch instead of just an OTM adaptation
                    proc.step_failures += 1
                    if proc.step_failures >= 3:
                        last_tried_scenario = proc.scenario_id
                        self.scenario_failure_count += 1
                        
                        failed_metric = step.health_check["checks"][0]["metric"]
                        val = self.last_kpi_snapshot.get("CellMetrics", {}).get(failed_metric, 0.0)
                        
                        # fallback and human escalation... i dont know if this is needed
                        if self.scenario_failure_count >= 2:
                            logger.critical(f"[ESCALATION] Multiple scenarios failed to resolve anomaly on '{failed_metric}'. Applying Safe Baseline and FREEZING.")
                            await self._publish_procedure_update("procedure_failed", step)
                            
                            # Wipe the broken procedure
                            self.active_procedure = None
                            self.scenario_failure_count = 0  # Reset so it can be clean when humans take over
                            
                            # Construct a safe, deterministic OTM with an Autopsy
                            baseline_otm = {
                                "version": "1.0",
                                "objective": {"service": "mbb", "kpi": "latency", "aggregation": "mean", "unit": "ms", "maximize": False},
                                "constraints": [
                                    {"service": "mbb", "kpi": "dl_mcs_max", "operator": "le", "threshold": 20.0, "unit": "", "id": "SAFE_MCS"},
                                    {"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge", "threshold": 46.0, "unit": "dBm", "id": "SAFE_PWR"}
                                ],
                                "metadata": {
                                    "timescale": "10s_window",
                                    "procedure_id": None,
                                    "is_critical_fallback": True,  # Flag for the Optimizer UI formatting
                                    "adaptation_log": [
                                        f"CRITICAL CRASH: AI exhausted all attempts to fix '{failed_metric}'.",
                                        f"Last attempted scenario '{last_tried_scenario}' aborted.",
                                        f"Fatal Metric: {failed_metric} degraded to {val:.2f}.",
                                        "Action taken: Reverted to deterministic safe mode (MCS <= 20, Tx Power >= 46).",
                                        "SYSTEM FROZEN: Autonomous control disabled. Awaiting manual human override."
                                    ]
                                }
                            }
                            
                            # Update our active OTM tracker
                            self.active_otm = baseline_otm
                            
                            # Publish the safe constraints directly to the RL engine
                            asyncio.create_task(self.bus.pub("intent.execute", make_msg("orch", "NEW_TARGET", "v1", self.active_otm)))
                            
                            # Lock the system. It will ignore new anomalies until a UI manual intent arrives.
                            self.state = IntentState.ESCALATED
                            logger.critical("[ORCHESTRATOR] System locked in ESCALATED state. Awaiting human intervention.")
                            
                            return True
                            
                        # SCENARIO SWITCH
                        logger.warning(f"[ORCHESTRATOR] Health check failed 3 times. Escalating '{failed_metric}' issue to Cognitive Core for Scenario Switch.")
                        await self._publish_procedure_update("procedure_failed", step)
                        await self._rollback_procedure()
                        
                        self.state = IntentState.THINKING
                        self.thinking_start_time = time.time()
                        
                        asyncio.create_task(self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                            "type": "scenario_failed",
                            "failed_procedure": last_tried_scenario,  # Tell the LLM what not to use again
                            "metric": failed_metric,
                            "value": val
                        })))
                        return True
                        
                    logger.warning(f"[ORCHESTRATOR] Routing health check failure to AI Adaptation (attempt {proc.step_failures}/3).")
                    self.state = IntentState.THINKING
                    self.thinking_start_time = time.time()
                    self.e2_acks_received = 0
                    
                    # Step back so we re-evaluate this exact health check next loop
                    proc.current_step_idx -= 1
                    
                    # Extract the failing metric to give the LLM context
                    failed_metric = step.health_check["checks"][0]["metric"]
                    val = self.last_kpi_snapshot.get("CellMetrics", {}).get(failed_metric, 0.0)
                    
                    # Let LLM adapt the OTM if the health check failed.
                    asyncio.create_task(self.bus.pub("ai.request", make_msg("ai.request", "REQ", "v1", {
                        "type": "adapt_otm", 
                        "metric": failed_metric,
                        "value": val,
                        "previous_otm": self.active_otm
                    })))
                    return True  # Keep procedure active, wait for AI to generate new OTM

            # Skip non-capable steps without health checks
            if not step.ns3_capable or step.otm_fragment is None:
                skip_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' SKIPPED: "
                            f"{step.ns3_skip_reason or 'no OTM fragment'}")
                logger.info(f"[ORCHESTRATOR] {skip_msg}")
                self._append_to_active_log(skip_msg)
                await self._publish_procedure_update("step_skipped", step)
                continue

            # Found an actionable step: merge its constraints into active OTM
            self._merge_step_into_otm(step)
            step_msg = (f"[{now_str}] Procedure step {step.order} '{step.name}' ACTIVATED: "
                        f"{step.description}")
            logger.info(f"[ORCHESTRATOR] {step_msg}")
            self._append_to_active_log(step_msg)

            # Re-enter ASSURANCE for this step
            self.state = IntentState.ASSURANCE
            self.assurance_end_time = time.time() + 30.0
            self.e2_acks_received = 0

            # Publish updated OTM so RL engine picks up the new constraints
            await self.bus.pub("intent.execute", make_msg(
                "opt", "NEW_TARGET", "v1", self.active_otm))
            await self._publish_procedure_update("step_activated", step)
            return True

        return False  # No more steps

    def _merge_step_into_otm(self, step):
        """Merge a procedure step's OTM fragment constraints into the active OTM."""
        if not self.active_otm or not step.otm_fragment:
            return

        existing = self.active_otm.setdefault("constraints", [])
        existing_ids = {c.get("id") for c in existing}

        for new_c in step.otm_fragment.get("constraints", []):
            cid = new_c.get("id")
            if cid in existing_ids:
                # Update existing constraint (same PROC_* id from a previous step)
                for ec in existing:
                    if ec.get("id") == cid:
                        ec["operator"] = new_c["operator"]
                        ec["threshold"] = new_c["threshold"]
                        break
            else:
                existing.append(dict(new_c))

    def _evaluate_health_check(self, step: ProcedureStep) -> tuple:
        """Evaluate health check conditions against latest telemetry.

        Returns (passed: bool, details: str).
        """
        checks = step.health_check.get("checks", [])
        cell = self.last_kpi_snapshot.get("CellMetrics", {})
        ues = self.last_kpi_snapshot.get("UEMetrics", [])

        # Build UE-metric averages
        ue_avg: dict = {}
        if ues:
            for ue in ues:
                for k, v in ue.items():
                    if k in ("ue_id", "cell_id"):
                        continue
                    try:
                        ue_avg.setdefault(k, []).append(float(v))
                    except (ValueError, TypeError):
                        pass
            ue_avg = {k: sum(vs) / len(vs) for k, vs in ue_avg.items()}

        results = []
        all_passed = True
        for chk in checks:
            metric = chk["metric"]
            op = chk["operator"]
            threshold = float(chk["threshold"])

            val = cell.get(metric)
            if val is None:
                val = ue_avg.get(metric)
            if val is None:
                results.append(f"{metric}: NO DATA")
                # No data is not a failure — skip this check
                continue

            val = float(val)
            if op == "le":
                ok = val <= threshold
            elif op == "lt":
                ok = val < threshold
            elif op == "ge":
                ok = val >= threshold
            elif op == "gt":
                ok = val > threshold
            else:
                ok = False

            status = "OK" if ok else "FAIL"
            results.append(f"{metric}={val:.2f} {op} {threshold} -> {status}")
            if not ok:
                all_passed = False

        return all_passed, "; ".join(results) if results else "no checks defined"

    @staticmethod
    def _find_step_index_by_name(proc: ActiveProcedure, step_name: str):
        """Find the index of a procedure step by its name."""
        for i, s in enumerate(proc.steps):
            if s.name == step_name:
                return i
        logger.warning(f"[ORCHESTRATOR] Branch target step '{step_name}' not found in procedure.")
        return None

    async def _complete_procedure(self):
        """Mark the active procedure as completed."""
        proc = self.active_procedure
        if not proc:
            return
        proc.state = ProcedureState.COMPLETED
        now_str = datetime.now(timezone.utc).isoformat()
        msg = f"[{now_str}] Procedure '{proc.scenario_name}' COMPLETED ({proc.current_step_idx + 1}/{len(proc.steps)} steps processed)"
        logger.info(f"[ORCHESTRATOR] {msg}")
        self._append_to_active_log(msg)

        await self._publish_procedure_update("procedure_completed")

        self.active_procedure = None

    async def _rollback_procedure(self):
        """Rollback to pre-procedure OTM after repeated failures."""
        proc = self.active_procedure
        if not proc:
            return
        
        proc.state = ProcedureState.FAILED
        now_str = datetime.now(timezone.utc).isoformat()
        msg = f"[{now_str}] Procedure '{proc.scenario_name}' FAILED at step {proc.current_step_idx}. Rolling back."
        logger.warning(f"[ORCHESTRATOR] {msg}")
        self._append_to_active_log(msg)

        await self._publish_procedure_update("procedure_failed")

        if proc.pre_procedure_otm:
            self.active_otm = proc.pre_procedure_otm
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", self.active_otm))
        else:
            self.active_otm = None

        self.active_procedure = None
        self.state = IntentState.MONITORING

    def _append_to_active_log(self, entry: str):
        """Append an entry to the active OTM's adaptation_log."""
        if self.active_otm:
            self.active_otm.setdefault("metadata", {}).setdefault("adaptation_log", []).append(entry)

    async def _publish_procedure_update(self, event: str, step=None):
        """Publish a procedure status update for the dashboard."""
        proc = self.active_procedure
        if not proc:
            return
        payload = {
            "event": event,
            "scenario_id": proc.scenario_id,
            "scenario_name": proc.scenario_name,
            "current_step": proc.current_step_idx,
            "total_steps": len(proc.steps),
            "state": proc.state.name,
            "all_steps": [
                {"order": s.order, "name": s.name, "description": s.description} 
                for s in proc.steps
            ]
        }
        if step:
            payload["step_name"] = step.name
            payload["step_description"] = step.description
            payload["ns3_capable"] = step.ns3_capable
            payload["is_health_check"] = step.health_check is not None
            payload["is_branch_only"] = step.branch_only
            
        await self.bus.pub("procedure.update", make_msg("orch", "PROCEDURE", "v1", payload))

    #  Graceful Cancel & Schedule Broadcast

    async def _apply_cancel_revert(self):
        """Apply deferred revert from a gracefully cancelled scheduled intent."""
        revert_otm = self.pending_cancel_revert
        self.pending_cancel_revert = None
        now_str = datetime.now(timezone.utc).isoformat()

        if revert_otm:
            msg = f"[{now_str}] Graceful cancel revert applied. Restored pre-scheduled OTM."
            logger.info(f"[ORCHESTRATOR] {msg}")
            self._append_to_active_log(msg)
            self.active_otm = revert_otm
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", revert_otm))
        else:
            self.active_otm = None
            logger.info("[ORCHESTRATOR] Graceful cancel revert applied. No previous OTM to restore.")

        await self._broadcast_schedule_list()

    async def _broadcast_schedule_list(self):
        """Broadcast the full schedule queue state to the dashboard."""
        items = []
        for si in self.schedule_queue:
            items.append({
                "id": si.id,
                "raw_text": si.raw_text,
                "state": si.state.name,
                "activate_at": si.activate_at.isoformat(),
                "deactivate_at": si.deactivate_at.isoformat() if si.deactivate_at else None,
                "created_at": si.created_at.isoformat(),
                "cancel_pending": (si.state == ScheduleState.CANCELLED
                                   and self.pending_cancel_revert is not None),
            })
        await self.bus.pub("schedule.update", make_msg("orch", "SCHEDULE", "v1", {"items": items}))

    #  Temporal Scheduler

    async def _enqueue_scheduled_intent(self, otm: dict, corr_id: str) -> ScheduledIntent:
        """Create a ScheduledIntent from an OTM with temporal metadata and add it to the queue."""
        temporal = otm.get("metadata", {}).get("temporal_resolved", {})
        activate_at = datetime.fromisoformat(temporal["activate_at_utc"])
        deactivate_at_str = temporal.get("deactivate_at_utc")
        deactivate_at = datetime.fromisoformat(deactivate_at_str) if deactivate_at_str else None
        raw_text = otm.get("metadata", {}).get("original_text", "scheduled intent")

        si = ScheduledIntent(
            id=str(_uuid.uuid4()),
            raw_text=raw_text,
            otm=otm,
            activate_at=activate_at,
            deactivate_at=deactivate_at,
            corr_id=corr_id,
        )
        self.schedule_queue.append(si)
        self.schedule_queue.sort(key=lambda x: x.activate_at)

        logger.info(f"[ORCHESTRATOR] Scheduled intent '{si.id[:8]}' for activation at {activate_at.isoformat()}"
                    f"{f', expires at {deactivate_at.isoformat()}' if deactivate_at else ', no expiry'}")
        await self._broadcast_schedule_list()
        return si

    async def _cancel_scheduled_intent(self, schedule_id: str):
        """Cancel a scheduled intent. For ACTIVE intents, defer revert until current work completes."""
        for si in self.schedule_queue:
            if si.id != schedule_id:
                continue
            if si.state in (ScheduleState.EXPIRED, ScheduleState.CANCELLED):
                logger.info(f"[ORCHESTRATOR] Intent '{si.id[:8]}' already {si.state.name}. Ignoring cancel.")
                return

            prev_state = si.state
            si.state = ScheduleState.CANCELLED
            now_str = datetime.now(timezone.utc).isoformat()

            if prev_state == ScheduleState.ACTIVE:
                # Graceful cancel: defer revert until system returns to MONITORING
                self.pending_cancel_revert = si.pre_intent_otm
                cancel_msg = (f"[{now_str}] Scheduled intent '{si.raw_text}' CANCELLED. "
                              f"Graceful revert pending — current work will finish before reverting.")
                logger.info(f"[ORCHESTRATOR] {cancel_msg}")
                self._append_to_active_log(cancel_msg)
            else:
                logger.info(f"[ORCHESTRATOR] Cancelled PENDING intent '{si.id[:8]}': {si.raw_text}")

            await self._broadcast_schedule_list()
            return

        logger.warning(f"[ORCHESTRATOR] Cancel requested for unknown schedule_id '{schedule_id}'.")

    async def _tick_scheduler(self):
        """Check for pending activations and active expirations every loop tick."""
        now_utc = datetime.now(timezone.utc)
        schedule_changed = False

        # Activate pending intents whose time has come
        for si in self.schedule_queue:
            if si.state != ScheduleState.PENDING:
                continue
            if now_utc < si.activate_at:
                continue
            # Only activate when the state machine is idle
            if self.state != IntentState.MONITORING:
                logger.debug(f"[ORCHESTRATOR] Scheduler: deferring activation of '{si.id[:8]}' — state is {self.state.name}")
                continue

            logger.info(f"[ORCHESTRATOR] Activating scheduled intent '{si.id[:8]}': {si.raw_text}")
            # Auto-expire any existing ACTIVE scheduled intent and unwind its OTM
            for other in self.schedule_queue:
                if other.state == ScheduleState.ACTIVE:
                    other.state = ScheduleState.EXPIRED
                    self.active_otm = other.pre_intent_otm  # Revert before snapshotting
                    logger.info(f"[ORCHESTRATOR] Auto-expired '{other.id[:8]}' (superseded by '{si.id[:8]}')")
            si.pre_intent_otm = self.active_otm  # Now captures the correct baseline
            si.state = ScheduleState.ACTIVE
            self._apply_otm(si.otm)
            # Republish so RL engine picks up the correct OTM at activation time
            await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_ACTIVATION", "v1", si.otm))
            schedule_changed = True
            break  # one activation per tick to respect the state machine

        # Expire active intents whose window has closed
        for si in self.schedule_queue:
            if si.state != ScheduleState.ACTIVE:
                continue
            if si.deactivate_at is None or now_utc < si.deactivate_at:
                continue

            logger.info(f"[ORCHESTRATOR] Scheduled intent '{si.id[:8]}' EXPIRED. Auto-reverting.")
            si.state = ScheduleState.EXPIRED

            revert_otm = si.pre_intent_otm
            if revert_otm:
                logger.info("[ORCHESTRATOR] Reverting to pre-scheduled OTM.")
                await self.bus.pub("intent.execute", make_msg("orch", "SCHEDULE_REVERT", "v1", revert_otm))
                self.active_otm = revert_otm
            else:
                self.active_otm = None

            self.state = IntentState.MONITORING
            schedule_changed = True

        # Garbage-collect finished intents (keep last 20)
        active = [si for si in self.schedule_queue if si.state in (ScheduleState.PENDING, ScheduleState.ACTIVE)]
        finished = [si for si in self.schedule_queue if si.state in (ScheduleState.EXPIRED, ScheduleState.CANCELLED)]
        self.schedule_queue = active + finished[-20:]

        if schedule_changed:
            await self._broadcast_schedule_list()