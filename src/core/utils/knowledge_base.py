import logging
from rdflib import Graph, RDF, URIRef, Literal, Namespace

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.graph = Graph()
        self.NS = Namespace("http://hybridAI.org/ontology#")
        self.populate_mock_data()

    # Populates the graph with some mock data for testing of Scenarios and Procedures.
    def populate_mock_data(self):
        
        # Define two types of entities, scenarios and procedures.
        Scenario = self.NS.Scenario
        Procedure = self.NS.Procedure

        # Define bounds for healthcheck, similar to SLO's from before i think
        self.graph.add((self.NS.MBB_Standard_Profile, self.NS.hasMaxLatency, Literal(30.0)))
        self.graph.add((self.NS.MBB_Standard_Profile, self.NS.hasMinThroughput, Literal(10.0)))
        self.graph.add((self.NS.MBB_Standard_Profile, self.NS.hasMaxBLER, Literal(0.01)))

        self.graph.add((self.NS.HighLatencyCongestion, self.NS.usesProfile, self.NS.MBB_Standard_Profile))

        # Scenario 1: High latency / congestion:
        scen_latency = URIRef(self.NS.HighLatencyCongestion)
        self.graph.add((scen_latency, RDF.type, Scenario))

        # scen_latency is a scenario.
        # We add what triggers (target metrics for minirocket) that are connected to this scenario

        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("DRB_PdcpSduDelayDl")))
        self.graph.add((scen_latency, self.NS.triggeredByMetric, Literal("UE_DRB_PdcpSduDelayDl_UEID")))

        # Procedure steps for this scenario:
        cong_steps = ["Prioritize MBB traffic scheduling",
                 "Reduce cell edge user MCS cap to 16",
                 "Increase TX Power by 2dBm to improve SNR",
                 "Enforce strict latency constraint <20ms",
                 "Monitor KPIs for 30 minutes and revert if no improvement",
                 "Notify operator if issue persists after 30 minutes"
                 ]

        # For each of these steps, they are added to the graph as (order not important) type of Procedure, 
        for step in cong_steps:
            step_node = URIRef(self.NS[f"proc_{hash(step)}"]) # hash to create unique id for each step
            self.graph.add((step_node, RDF.type, Procedure)) # This node is a procedure
            self.graph.add((step_node, self.NS.description, Literal(step))) # The description of the procedure is the text above
            self.graph.add((scen_latency, self.NS.hasStep, step_node)) # latency scenario has the step *this procedure*

        # SCENARIO 2: Maintenance / Software Upgrade:
        scen_maint = URIRef(self.NS.CellUpgrade)
        self.graph.add((scen_maint, RDF.type, Scenario))
        self.graph.add((scen_maint, self.NS.triggeredByKeyword, Literal("upgrade")))
        
        maintenance_steps = [
            "Bias handovers away from cell X (adjust CIO)",
            "Drain active users (monitor connected UE count -> 0)",
            "Reduce scheduling priority to zero",
            "Perform upgrade",
            "Health check (verify KPIs within bounds)",
            "Restore normal handover parameters",
            "Rebalance load across cluster"
        ]

        # This set of procedures is an ordered set in the graph. The LLM can choose to follow them in order, 
        # or pick specific ones based on the situation.
        for i, step in enumerate(maintenance_steps):
            step_node = URIRef(self.NS[f"maint_step_{i}"])
            self.graph.add((step_node, RDF.type, Procedure))
            self.graph.add((step_node, self.NS.description, Literal(step)))
            self.graph.add((step_node, self.NS.order, Literal(i)))
            self.graph.add((scen_maint, self.NS.hasStep, step_node))

    # Queries the RDF graph for the specific health bounds of a cell.
    def get_health_thresholds(self, cell_id):
        query = f"""
        PREFIX ns: <{self.NS}>
        SELECT ?l_max ?t_min WHERE {{
            ?cell ns:cellID "{cell_id}" .
            ?cell ns:hasProfile ?p .
            ?p ns:latencyMax ?l_max .
            ?p ns:throughputMin ?t_min .
        }}
        """
        results = self.graph.query(query)
        
        for row in results:
            return {
                'latency_max_ms': float(row.l_max),
                'throughput_min_mbps': float(row.t_min)
            }

        # Safe Fallback if the cell isn't in our database yet
        return {'latency_max_ms': 30.0, 'throughput_min_mbps': 5.0}

    def query_scenario(self, metric_name: str) -> list:
        query = f"""
        PREFIX ns: <{self.NS}>
        SELECT DISTINCT ?fid
        WHERE {{
            ?s ns:triggeredByMetric ?metric .
            ?s ns:hasStep ?step .
            ?step ns:description ?fid .
            
            # Use STR() to ensure we are comparing text to text
            # Use LCASE for case-insensitive matching
            FILTER(LCASE(STR(?metric)) = LCASE("{metric_name}"))
        }}
        """

        results = self.graph.query(query)
        procedures = [str(row.fid) for row in results]
        
        if procedures:
            logger.info(f"[KB] Found {len(procedures)} procedures for {metric_name}: {procedures}")
        else:
            logger.warning(f"[KB] No procedures found for metric: {metric_name}")
            # DEBUG: Log what is in the graph
            # for s, p, o in self.graph.triples((None, self.NS.triggeredByMetric, None)):
            #     logger.debug(f"[KB DEBUG] Graph contains trigger: {o}")

        return procedures
    
