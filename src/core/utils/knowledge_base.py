import logging
from rdflib import Graph, RDF, URIRef, Literal, Namespace

logger = logging.getLogger(__name__)

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
        
        # Link triggers (These match the telemetry alerts exactly)
        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("DRB_PdcpSduDelayDl")))
        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("UE_DRB_PdcpSduDelayDl_UEID")))

        # Procedures crafted specifically to guide the LLM's OTM generation.
        # We tell it exactly what's happening and what constraints the DQN needs.
        cong_steps = [
            "DIAGNOSIS: Network is experiencing High Latency Congestion.",
            "ACTION: Enforce strict constraints to reduce delay.",
            "OTM_INSTRUCTION: Set an objective to minimize latency (DRB_PdcpSduDelayDl).",
            "OTM_INSTRUCTION: Add a constraint to force 'DRB_PdcpSduDelayDl' <= 50.0 ms.",
            "OTM_INSTRUCTION: Add a constraint to limit 'dl_mcs_max' to <= 20 to improve reliability.",
            "OTM_INSTRUCTION: Add a constraint to set 'prb_weight' >= 1.2 for latency-sensitive slices."
        ]

        for i, step in enumerate(cong_steps):
            step_node = URIRef(self.NS[f"latency_proc_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((scen_latency, self.NS.hasStep, step_node))

        # ---------------------------------------------------------
        # SCENARIO 2: Maintenance / Software Upgrade (For future testing)
        # ---------------------------------------------------------
        scen_maint = URIRef(self.NS.CellUpgrade)
        self.graph.add((scen_maint, RDF.type, Scenario))
        self.graph.add((scen_maint, self.NS.triggeredByKeyword, Literal("upgrade")))
        
        maintenance_steps = [
            "Drain active users (monitor connected UE count -> 0)",
            "Reduce scheduling priority to zero",
            "Perform upgrade",
            "Health check (verify KPIs within bounds)"
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