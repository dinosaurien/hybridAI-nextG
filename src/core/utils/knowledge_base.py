import logging
from typing import Dict, List, Optional
from rdflib import Graph, RDF, URIRef, Literal, Namespace
from core.common.types import ProcedureStep

logger = logging.getLogger(__name__)

'''
Procedure catalog is a structured knowledge base of multi-step procedures for different network scenarios.
Each scenario (e.g., "High Latency Congestion") maps to a specific procedure with ordered steps.
Each steps marked ns3_capable=True can be executed in the current sim.
Steps that are marked False are there as a proof of concept for how the system could be extended in a real deployment or with
a more capable simulated RIC through the E2 interface.
'''

PROCEDURE_CATALOG: Dict[str, dict] = {
    "cell_maintenance": {
        "name": "Cell Maintenance / Software Upgrade",
        "description": "Safely reduce radio parameters to conservative settings for maintenance.",
        "trigger_keywords": ["upgrade", "maintenance", "software update"],
        "trigger_metrics": [],
        "steps": [
            ProcedureStep(0, "reduce_mcs", "Reduce MCS for conservative operation",
                          {"constraints": [{"service": "mbb", "kpi": "dl_mcs_max", "operator": "le",
                                            "threshold": 14.0, "unit": "", "id": "PROC_MCS"}]},
                          ns3_capable=True),
            ProcedureStep(1, "reduce_tx_power", "Reduce Tx power during maintenance window",
                          {"constraints": [{"service": "mbb", "kpi": "tx_power_dbm", "operator": "le",
                                            "threshold": 35.0, "unit": "dBm", "id": "PROC_TXPOW"}]},
                          ns3_capable=True),
            ProcedureStep(2, "verify_safe_mode", "Verify network is stable in safe mode",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 100.0},
                          ]}),
            ProcedureStep(3, "perform_upgrade", "Perform software upgrade (Simulated)",
                          None, ns3_capable=False,
                          ns3_skip_reason="Out-of-band operation. Once complete, clear OTM to restore normal behavior."),
        ],
    },

    "interference_response": {
        "name": "Interference Anomaly Response",
        "description": "React to detected inter-cell interference by increasing signal robustness and power.",
        "trigger_keywords": ["interference", "inter-cell", "neighbor cell"],
        "trigger_metrics": ["UE_DRB_BlerDl_UEID", "bler_dl"],
        "steps": [
            ProcedureStep(0, "conservative_mcs", "Reduce MCS to increase signal robustness",
                          {"constraints": [{"service": "mbb", "kpi": "dl_mcs_max", "operator": "le",
                                            "threshold": 16.0, "unit": "", "id": "PROC_MCS"}]},
                          ns3_capable=True),
            ProcedureStep(1, "verify_mcs_effect", "Verify if MCS reduction stabilized BLER",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "UE_DRB_BlerDl_UEID", "operator": "le", "threshold": 0.05},
                          ]}),
            ProcedureStep(2, "boost_tx_power", "Boost Tx power to overcome neighbor interference",
                          {"constraints": [{"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge",
                                            "threshold": 48.0, "unit": "dBm", "id": "PROC_TXPOW"}]},
                          ns3_capable=True),
            ProcedureStep(3, "verify_final_interference", "Verify BLER after power boost",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "UE_DRB_BlerDl_UEID", "operator": "le", "threshold": 0.03},
                          ]}),
        ],
    },

    "graceful_degradation": {
        "name": "Graceful Degradation Under Overload",
        "description": "Handle sustained network congestion by lowering MCS and boosting TX power.",
        "trigger_keywords": ["overload", "congestion", "saturated"],
        "trigger_metrics": ["DRB_PdcpSduDelayDl"],
        "steps": [
            ProcedureStep(0, "conservative_mcs", "Reduce MCS to prevent retransmissions under load",
                          {"constraints": [{"service": "mbb", "kpi": "dl_mcs_max", "operator": "le",
                                            "threshold": 18.0, "unit": "", "id": "PROC_MCS"}]},
                          ns3_capable=True),
            ProcedureStep(1, "boost_tx_power", "Increase Tx power to improve signal quality",
                          {"constraints": [{"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge",
                                            "threshold": 48.0, "unit": "dBm", "id": "PROC_TXPOW"}]},
                          ns3_capable=True),
            ProcedureStep(2, "verify_final_congestion", "Verify final latency",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 60.0},
                          ]}),
        ],
    },

    "demand_surge": {
        "name": "Predicted Demand Surge / Proactive Scaling",
        "description": "Proactively scale network capacity ahead of a predicted demand surge.",
        "trigger_keywords": ["surge", "event", "stadium"],
        "trigger_metrics": [],
        "steps": [
            ProcedureStep(0, "increase_tx_coverage", "Increase Tx power for wider coverage",
                          {"constraints": [{"service": "mbb", "kpi": "tx_power_dbm", "operator": "ge",
                                            "threshold": 48.0, "unit": "dBm", "id": "PROC_TXPOW"}]},
                          ns3_capable=True),
            ProcedureStep(1, "increase_mcs_throughput", "Allow higher MCS for peak throughput",
                          {"constraints": [{"service": "mbb", "kpi": "dl_mcs_max", "operator": "ge",
                                            "threshold": 24.0, "unit": "", "id": "PROC_MCS"}]},
                          ns3_capable=True),
            ProcedureStep(2, "verify_readiness", "Verify network stability before surge",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 50.0},
                              {"metric": "UE_DRB_BlerDl_UEID", "operator": "le", "threshold": 0.05},
                          ]}),
        ],
    },

    "energy_optimization": {
        "name": "Energy Optimization Cycle / Night Mode",
        "description": "Reduce energy consumption progressively while verifying QoS floors.",
        "trigger_keywords": ["energy", "night", "power save", "eco"],
        "trigger_metrics": ["tx_power_dbm"],
        "steps": [
            ProcedureStep(0, "reduce_tx_power", "Step 1: Reduce Tx power",
                          {"constraints": [{"service": "mbb", "kpi": "tx_power_dbm", "operator": "le",
                                            "threshold": 38.0, "unit": "dBm", "id": "PROC_TXPOW"}]},
                          ns3_capable=True),
            ProcedureStep(1, "verify_after_power", "Verify QoS floors after Tx power reduction",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 80.0},
                          ]}),
            ProcedureStep(2, "reduce_mcs", "Step 2: Reduce MCS to limit retransmission energy",
                          {"constraints": [{"service": "mbb", "kpi": "dl_mcs_max", "operator": "le",
                                            "threshold": 16.0, "unit": "", "id": "PROC_MCS"}]},
                          ns3_capable=True),
            ProcedureStep(3, "verify_after_mcs", "Verify QoS floors after MCS reduction",
                          None, ns3_capable=True,
                          health_check={"checks": [
                              {"metric": "DRB_PdcpSduDelayDl", "operator": "le", "threshold": 80.0},
                              {"metric": "UE_DRB_BlerDl_UEID", "operator": "le", "threshold": 0.10},
                          ]}),
        ],
    },
}

class KnowledgeBase:
    def __init__(self):
        self.graph = Graph()
        self.NS = Namespace("http://hybridAI.org/ontology#")
        self.populate_mock_data()

    def populate_mock_data(self):
        Scenario = self.NS.Scenario
        Procedure = self.NS.Procedure

        # ---------------------------------------------------------
        # SCENARIO 1: High Latency / Congestion
        # ---------------------------------------------------------
        scen_latency = URIRef(self.NS.HighLatencyCongestion)
        self.graph.add((scen_latency, RDF.type, Scenario))

        # Map to the orchestrators set of procedures declared above. 
        self.graph.add((scen_latency, self.NS.mapsToProcedureId, Literal("graceful_degradation")))
        
        # Link triggers (These match the telemetry alerts exactly)
        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("DRB_PdcpSduDelayDl")))
        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("UE_DRB_PdcpSduDelayDl_UEID")))

        # Procedures crafted specifically to guide the LLM's OTM generation.
        # The Objective is generated here. The procedures will set the constraints.
        cong_steps = [
            "DIAGNOSIS: Network is experiencing High Latency Congestion.",
            "ACTION: Enforce strict constraints to reduce delay.",
            "OTM_INSTRUCTION: Set an objective to minimize latency (DRB_PdcpSduDelayDl).",
            "OTM_INSTRUCTION: Leave the 'constraints' array empty. This will be populated later."
        ]

        for i, step in enumerate(cong_steps):
            step_node = URIRef(self.NS[f"latency_proc_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_latency, self.NS.hasStep, step_node))

        # ---------------------------------------------------------
        # SCENARIO 2: Low Throughput
        # ---------------------------------------------------------
        scen_thr = URIRef(self.NS.LowThroughput)
        self.graph.add((scen_thr, RDF.type, Scenario))
        self.graph.add((scen_thr, self.NS.triggeredByMetric, Literal("UE_DRB_UEThpDl_UEID")))
        self.graph.add((scen_thr, self.NS.mapsToProcedureId, Literal("demand_surge")))

        thr_steps = [
            "DIAGNOSIS: Network is experiencing Low Downlink Throughput.",
            "ACTION: Increase resource allocation and modulation order.",
            "OTM_INSTRUCTION: Set an objective to maximize throughput (UE_DRB_UEThpDl_UEID).",
            "OTM_INSTRUCTION: Leave the 'constraints' array empty. This will be populated later."
        ]
        for i, step in enumerate(thr_steps):
            step_node = URIRef(self.NS[f"throughput_proc_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_thr, self.NS.hasStep, step_node))

        # ---------------------------------------------------------
        # SCENARIO 3: High Block Error Rate
        # ---------------------------------------------------------
        scen_bler = URIRef(self.NS.HighBLER)
        self.graph.add((scen_bler, RDF.type, Scenario))
        self.graph.add((scen_bler, self.NS.triggeredByMetric, Literal("UE_DRB_BlerDl_UEID")))
        self.graph.add((scen_bler, self.NS.triggeredByMetric, Literal("bler_dl")))
        self.graph.add((scen_bler, self.NS.mapsToProcedureId, Literal("interference_response")))

        bler_steps = [
            "DIAGNOSIS: Network is experiencing High Downlink Block Error Rate.",
            "ACTION: Reduce modulation order and increase transmit power to improve signal reliability.",
            "OTM_INSTRUCTION: Set an objective to minimize BLER (UE_DRB_BlerDl_UEID).",
            "OTM_INSTRUCTION: Leave the 'constraints' array empty. This will be populated later."
        ]
        for i, step in enumerate(bler_steps):
            step_node = URIRef(self.NS[f"bler_proc_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_bler, self.NS.hasStep, step_node))

        # ---------------------------------------------------------
        # SCENARIO 4: Energy Saving / Power Reduction
        # ---------------------------------------------------------
        scen_energy = URIRef(self.NS.EnergySaving)
        self.graph.add((scen_energy, RDF.type, Scenario))
        self.graph.add((scen_energy, self.NS.triggeredByMetric, Literal("tx_power_dbm")))
        self.graph.add((scen_energy, self.NS.mapsToProcedureId, Literal("energy_optimization")))

        energy_steps = [
            "DIAGNOSIS: Operator has requested an energy-saving configuration.",
            "ACTION: Reduce transmit power and limit modulation to save energy while maintaining baseline service.",
            "OTM_INSTRUCTION: Set an objective to minimize latency (DRB_PdcpSduDelayDl). The procedure constraints will enforce the energy-saving actions; the objective ensures QoS is maintained.",
            "OTM_INSTRUCTION: Leave the 'constraints' array empty. This will be populated later."
        ]
        for i, step in enumerate(energy_steps):
            step_node = URIRef(self.NS[f"energy_proc_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_energy, self.NS.hasStep, step_node))

        # ---------------------------------------------------------
        # SCENARIO 5: Maintenance / Software Upgrade (For future testing)
        # ---------------------------------------------------------
        scen_maint = URIRef(self.NS.CellUpgrade)
        self.graph.add((scen_maint, RDF.type, Scenario))
        self.graph.add((scen_maint, self.NS.triggeredByKeyword, Literal("upgrade")))
        self.graph.add((scen_maint, self.NS.mapsToProcedureId, Literal("cell_maintenance")))
        
        maintenance_steps = [
            "DIAGNOSIS: Operator initiated Cell Maintenance / Software Upgrade.",
            "ACTION: Safely drain traffic and reduce radio parameters to conservative settings.",
            "OTM_INSTRUCTION: Set an objective to minimize latency (DRB_PdcpSduDelayDl). The procedure constraints will enforce conservative settings; the objective ensures QoS is maintained.",
            "OTM_INSTRUCTION: Leave the 'constraints' array empty. This will be populated later by the Orchestrator."
        ]

        for i, step in enumerate(maintenance_steps):
            step_node = URIRef(self.NS[f"maint_step_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_maint, self.NS.hasStep, step_node))

    def get_health_thresholds(self, cell_id):
        # Fallback if the cell isn't in our database yet
        return {'latency_max_ms': 50.0, 'throughput_min_mbps': 5.0}

    def query_scenario(self, metric_name: str) -> list:
        # Optimized SPARQL query to fetch procedure descriptions based on metric trigger
        query = f"""
        PREFIX ns: <{self.NS}>
        SELECT DISTINCT ?desc
        WHERE {{
            ?s ns:triggeredByMetric ?metric .
            ?s ns:hasStep ?step .
            ?step ns:description ?desc .
            
            # Case-insensitive string match
            FILTER(LCASE(STR(?metric)) = LCASE("{metric_name}"))
        }}
        """

        results = self.graph.query(query)
        procedures = [str(row.desc) for row in results]
        
        if procedures:
            logger.info(f"[KB] Found {len(procedures)} procedures for anomaly on {metric_name}.")
        else:
            logger.warning(f"[KB] No procedures found in graph for metric: {metric_name}")

        return procedures

    def resolve_procedure_id(self, metric_name: str) -> Optional[str]:
        """Symbolically resolve the procedure_id for a given metric via SPARQL.
        Returns the first matching procedure_id, or None if no match."""
        query = f"""
        PREFIX ns: <{self.NS}>
        SELECT DISTINCT ?procId
        WHERE {{
            ?scenario ns:triggeredByMetric ?metric .
            ?scenario ns:mapsToProcedureId ?procId .
            FILTER(LCASE(STR(?metric)) = LCASE("{metric_name}"))
        }}
        LIMIT 1
        """
        results = self.graph.query(query)
        for row in results:
            proc_id = str(row.procId)
            logger.info(f"[KB] Symbolic resolve: metric '{metric_name}' -> procedure '{proc_id}'")
            return proc_id
        logger.warning(f"[KB] No procedure_id mapped for metric: {metric_name}")
        return None

    # ------------------------------------------------------------------ #
    #  Procedure Catalog queries                                          #
    # ------------------------------------------------------------------ #

    def get_all_scenario_summaries(self) -> Dict[str, dict]:
        """Return all scenarios with name, description, keywords, and step descriptions.
        Designed to be included in LLM prompts for scenario matching."""
        summaries = {}
        for sid, scenario in PROCEDURE_CATALOG.items():
            summaries[sid] = {
                "name": scenario["name"],
                "description": scenario["description"],
                "trigger_keywords": scenario["trigger_keywords"],
                "steps": [
                    {"order": s.order, "name": s.name, "description": s.description,
                     "ns3_capable": s.ns3_capable,
                     "has_health_check": s.health_check is not None,
                     "branch_only": s.branch_only}
                    for s in scenario["steps"]
                ],
            }
        return summaries

    def get_procedure_steps(self, scenario_id: str) -> Optional[List[ProcedureStep]]:
        """Return ordered ProcedureStep list for a scenario, or None if not found."""
        scenario = PROCEDURE_CATALOG.get(scenario_id)
        if scenario is None:
            logger.warning(f"[KB] No procedure found for scenario_id: {scenario_id}")
            return None
        return scenario["steps"]

    def get_scenario_name(self, scenario_id: str) -> str:
        """Return human-readable name for a scenario_id."""
        scenario = PROCEDURE_CATALOG.get(scenario_id)
        return scenario["name"] if scenario else scenario_id

    def match_scenario_by_keywords(self, text: str) -> Optional[str]:
        """Match user text to a scenario via keyword scoring. Returns scenario_id or None."""
        lower = text.lower()
        best_id, best_score = None, 0
        for sid, scenario in PROCEDURE_CATALOG.items():
            score = sum(1 for kw in scenario["trigger_keywords"] if kw in lower)
            if score > best_score:
                best_score = score
                best_id = sid
        if best_id:
            logger.info(f"[KB] Keyword match: '{text}' -> {best_id} (score={best_score})")
        return best_id

    def match_scenario_by_metric(self, metric: str) -> Optional[str]:
        """Find a scenario triggered by a specific telemetry metric. Returns scenario_id or None."""
        metric_lower = metric.lower()
        for sid, scenario in PROCEDURE_CATALOG.items():
            for m in scenario["trigger_metrics"]:
                if m.lower() == metric_lower:
                    logger.info(f"[KB] Metric match: '{metric}' -> {sid}")
                    return sid
        return None
    
    def get_rdf_scenarios_for_llm(self) -> str:
        """Extracts the semantic RDF knowledge base into a readable string for the LLM context."""
        query = f"""
        PREFIX ns: <{self.NS}>
        SELECT ?scenario ?procId ?desc
        WHERE {{
            ?scenario a ns:Scenario .
            OPTIONAL {{ ?scenario ns:mapsToProcedureId ?procId . }}
            ?scenario ns:hasStep ?step .
            ?step ns:description ?desc .
        }}
        """
        results = self.graph.query(query)
        
        # Group descriptions by scenario
        scenarios = {}
        for row in results:
            s_name = str(row.scenario).split("#")[-1]
            p_id = str(row.procId) if row.procId else "unknown"
            desc = str(row.desc)
            
            if s_name not in scenarios:
                scenarios[s_name] = {"procedure_id": p_id, "instructions": []}
            scenarios[s_name]["instructions"].append(desc)

        output = []
        for s_name, data in scenarios.items():
            output.append(f"--- SCENARIO: {s_name} ---")
            output.append(f"Mapped procedure_id: {data['procedure_id']}")
            output.append("Semantic Instructions:")
            for instr in data["instructions"]:
                output.append(f"  - {instr}")
            output.append("")
            
        return "\n".join(output)