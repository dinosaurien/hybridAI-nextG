/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/mmwave-ue-net-device.h"
#include "ns3/mmwave-enb-net-device.h"
#include "ns3/v4ping-helper.h"
#include "ns3/v4ping.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/netanim-module.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/buildings-module.h"
#include "ns3/buildings-helper.h"
#include "ns3/random-rectangle-position-allocator.h"

#include "ns3/mmwave-component-carrier-enb.h"
#include "ns3/mmwave-flex-tti-mac-scheduler.h"
#include "ns3/on-off-application.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <vector>

using namespace ns3;
using namespace mmwave;
namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("Concert_Mmwave_1gNB_200UE");

static GlobalValue g_simTime("simTime", "Simulation time (s)",
  DoubleValue(60.0), MakeDoubleChecker<double>(1.0, 3600.0));
static GlobalValue g_outDir("outDir", "Output directory",
  StringValue("out/logs_concert"), MakeStringChecker());

static GlobalValue q_useSemaphores ("useSemaphores","If true, enables the use of semaphores for external environment control",
                                    BooleanValue(false), MakeBooleanChecker());
static GlobalValue g_controlFileName ("controlFileName","The path to the control file (can be absolute)",
                                      StringValue(""), MakeStringChecker());
static GlobalValue g_e2lteEnabled ("e2lteEnabled","If true, send LTE E2 reports", BooleanValue(true), MakeBooleanChecker());
static GlobalValue g_e2nrEnabled ("e2nrEnabled","If true, send NR E2 reports", BooleanValue(true), MakeBooleanChecker());
static GlobalValue g_e2du ("e2du","If true, send DU reports", BooleanValue(true), MakeBooleanChecker());
static GlobalValue g_e2cuUp ("e2cuUp","If true, send CU-UP reports", BooleanValue(true), MakeBooleanChecker());
static GlobalValue g_e2cuCp ("e2cuCp","If true, send CU-CP reports", BooleanValue(true), MakeBooleanChecker());
static GlobalValue g_indicationPeriodicity ("indicationPeriodicity","E2 Indication Periodicity (s)", 
                                            DoubleValue(0.1), MakeDoubleChecker<double>(0.01, 2.0));
static GlobalValue g_e2TermIp ("e2TermIp","RIC E2 termination IP", StringValue("10.0.2.10"), MakeStringChecker());
static GlobalValue g_enableE2FileLogging ("enableE2FileLogging","Offline file logging instead of connecting to RIC", BooleanValue(false), MakeBooleanChecker());
static GlobalValue g_reducedPmValues ("reducedPmValues", "If true, use a subset of the pm containers", BooleanValue(false), MakeBooleanChecker());

// ---------------- Traffic Randomization (Removed for Concert Scenario) ----------------
// We use a scheduled, deterministic event for traffic now.

// ---------------- Timeseries sampler (Modified for Aggregate Throughput) ----------------
// Global variables
struct GlobalState {
  double lastT = 0.0;
  uint64_t lastBytes = 0;
  double ewma = 0.0;
  bool seenPing = false;
  double lastPingMs = 0.0;
} gS;

// NetAnim is disabled for performance, so g_anim is removed.

static void PingRttCallback(Time rtt) {
  gS.lastPingMs = rtt.GetMilliSeconds();
  gS.seenPing   = true;
}

// MODIFIED: Takes a vector of sinks and calculates aggregate throughput
static void SampleAll(const NodeContainer &ueNodes,
                      const NetDeviceContainer &ueDevs,
                      Ptr<Node> gnbNode,
                      double covRadius,
                      const std::vector<Ptr<PacketSink>> &allSinks,
                      double periodSec)
{
  static std::ofstream f;
  static bool headerDone = false;
  const uint32_t numUes = ueNodes.GetN();

  if (!headerDone) {
    f.open("sim_concert_timeseries.csv", std::ios::out | std::ios::trunc);
    f << std::fixed << std::setprecision(6);
    f << "time_s"
      << ",ue0_x"
      << ",ue0_y"
      << ",ue0_dist_to_gnb_m"
      << ",ue0_inside"
      << ",aggregate_throughput_mbps" // Key metric
      << ",aggregate_throughput_ewma"
      << ",ping_ms"
      << ",mcs_dl"
      << ",mcs_ul"
      << ",fixed_mcs_dl"
      << "\n";
    headerDone = true;
  }

  const double now = Simulator::Now().GetSeconds();
  Vector gp = gnbNode->GetObject<MobilityModel>()->GetPosition();
  f << now;

  // Only log metrics for UE0 for performance/simplicity
  Vector p0 = ueNodes.Get(0)->GetObject<MobilityModel>()->GetPosition();
  const double dx = p0.x - gp.x, dy = p0.y - gp.y, dz = p0.z - gp.z;
  const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
  const int inside = (dist <= covRadius ? 1 : 0);
  
  f << "," << p0.x
    << "," << p0.y
    << "," << dist
    << "," << inside;

  // MODIFIED: Aggregate throughput calculation
  double mbps = 0.0;
  uint64_t currentTotalBytes = 0; 
  
  for (const auto& sink : allSinks) {
    currentTotalBytes += sink->GetTotalRx();
  }
  
  if (gS.lastT > 0.0) {
    double dt = now - gS.lastT;
    if (dt > 0.0) {
      mbps = 8.0 * (currentTotalBytes - gS.lastBytes) / dt / 1e6;
    }
  }
  gS.lastBytes = currentTotalBytes;
  gS.lastT = now;


  const double tau = 1.0; // EWMA time constant (s)
  const double alpha = 1.0 - std::exp(-(periodSec / tau));
  gS.ewma = alpha * mbps + (1.0 - alpha) * gS.ewma;

  double pingMs = gS.seenPing ? gS.lastPingMs : 0.0;

  // Get MCS values from scheduler
  uint8_t mcsDl = 255;  // 255 = adaptive/not fixed
  uint8_t mcsUl = 255;
  bool fixedMcsDl = false;
  
  Ptr<mmwave::MmWaveEnbNetDevice> enbDev = gnbNode->GetDevice(0)->GetObject<mmwave::MmWaveEnbNetDevice>();
  if (enbDev) {
    std::map<uint8_t, Ptr<mmwave::MmWaveComponentCarrier>> ccMap = enbDev->GetCcMap();
    if (!ccMap.empty()) {
      Ptr<mmwave::MmWaveComponentCarrierEnb> cc = 
          DynamicCast<mmwave::MmWaveComponentCarrierEnb>(ccMap.at(0));
      if (cc) {
        Ptr<mmwave::MmWaveMacScheduler> sched = cc->GetMacScheduler();
        if (sched) {
          Ptr<mmwave::MmWaveFlexTtiMacScheduler> flexSched = 
              DynamicCast<mmwave::MmWaveFlexTtiMacScheduler>(sched);
          if (flexSched) {
            mcsDl = flexSched->GetCurrentMcsDl();
            mcsUl = flexSched->GetCurrentMcsUl();
            fixedMcsDl = flexSched->IsFixedMcsDl();
          }
        }
      }
    }
  }

  f << "," << mbps
    << "," << gS.ewma
    << "," << pingMs
    << "," << static_cast<int>(mcsDl)
    << "," << static_cast<int>(mcsUl)
    << "," << (fixedMcsDl ? 1 : 0)
    << "\n";
  f.flush();

  // Schedule next sample
  Simulator::Schedule(Seconds(periodSec), &SampleAll,
                      ueNodes, ueDevs, gnbNode, covRadius, allSinks, periodSec);
}

// ---------------- Dynamic MCS Logic (Used for Orchestration) ----------------
// ChangeMcs and GetCurrentMcs remain the same as they are the control mechanism.
static void ChangeMcs(Ptr<Node> gnb, int mcs)
{
  Ptr<mmwave::MmWaveEnbNetDevice> enbDev = gnb->GetDevice(0)->GetObject<mmwave::MmWaveEnbNetDevice>();
  if (!enbDev) return;
  std::map<uint8_t, Ptr<mmwave::MmWaveComponentCarrier>> ccMap = enbDev->GetCcMap();
  if (ccMap.empty()) return;
  Ptr<mmwave::MmWaveComponentCarrierEnb> cc = DynamicCast<mmwave::MmWaveComponentCarrierEnb>(ccMap.at(0));
  if (!cc) return;
  Ptr<mmwave::MmWaveMacScheduler> sched = cc->GetMacScheduler();
  if (!sched) return;

  Ptr<mmwave::MmWaveFlexTtiMacScheduler> flexSched = DynamicCast<mmwave::MmWaveFlexTtiMacScheduler>(sched);
  if (flexSched) {
    if (mcs >= 0) {
      flexSched->SetAttribute("FixedMcsDl", BooleanValue(true));
      flexSched->SetAttribute("McsDefaultDl", UintegerValue(mcs));
      flexSched->SetAttribute("FixedMcsUl", BooleanValue(true));
      flexSched->SetAttribute("McsDefaultUl", UintegerValue(mcs));
      NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "s: [ORCHESTRATION] Setting Fixed MCS to " << mcs);
      std::cerr << "  → MCS change applied: Fixed MCS=" << mcs << " (DL and UL)" << std::endl;
    } else {
      flexSched->SetAttribute("FixedMcsDl", BooleanValue(false));
      flexSched->SetAttribute("FixedMcsUl", BooleanValue(false));
      NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "s: [ORCHESTRATION] Restoring Adaptive MCS");
      std::cerr << "  → MCS change applied: Adaptive MCS restored (DL and UL)" << std::endl;
    }
  }
}

static int GetCurrentMcs(Ptr<Node> gnb)
{
  Ptr<mmwave::MmWaveEnbNetDevice> enbDev = gnb->GetDevice(0)->GetObject<mmwave::MmWaveEnbNetDevice>();
  if (!enbDev) return -1;
  std::map<uint8_t, Ptr<mmwave::MmWaveComponentCarrier>> ccMap = enbDev->GetCcMap();
  if (ccMap.empty()) return -1;
  Ptr<mmwave::MmWaveComponentCarrierEnb> cc = DynamicCast<mmwave::MmWaveComponentCarrierEnb>(ccMap.at(0));
  if (!cc) return -1;
  Ptr<mmwave::MmWaveMacScheduler> sched = cc->GetMacScheduler();
  if (!sched) return -1;
  Ptr<mmwave::MmWaveFlexTtiMacScheduler> flexSched = DynamicCast<mmwave::MmWaveFlexTtiMacScheduler>(sched);
  if (flexSched) {
    return flexSched->GetCurrentMcsDl();
  }
  return -1;
}

// ---------------- Concert Orchestration Events ----------------

static void TriggerPeakLoad(NodeContainer remoteHosts)
{
    // The RH is Node 2. We loop through all its applications (one per UE)
    Ptr<Node> rh = remoteHosts.Get(0);
    for (uint32_t i = 0; i < rh->GetNApplications(); ++i) {
        Ptr<OnOffApplication> app = DynamicCast<OnOffApplication>(rh->GetApplication(i));
        if (app) {
            // This global change simulates a massive synchronous action (like mass video upload)
            app->SetAttribute("DataRate", StringValue("20Mbps")); 
        }
    }
    std::cerr << "\n"
              << "==============================================================\n"
              << "== [ORCHESTRATION EVENT] PEAK DEMAND TRIGGERED              ==\n"
              << "== Time: " << Simulator::Now().GetSeconds() << "s                                       ==\n"
              << "== Action: ALL USERS BOOSTED (e.g., mass upload/streaming)  ==\n"
              << "==============================================================\n" << std::endl;
    NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "s: [ORCHESTRATION] PEAK DEMAND - All light users boosted to 20Mbps.");
}


// ---------------- main ----------------
int main (int argc, char** argv)
{
  // ... (CommandLine and Variable setup remains the same) ...
  CommandLine cmd;
  int rngSeed = 0;
  cmd.AddValue("rngSeed", "Seed for random number generator (default 0 = random)", rngSeed);
  cmd.Parse(argc, argv);

  if (rngSeed == 0) {
    srand(time(NULL));
  } else {
    srand(rngSeed);
  }

  DoubleValue simV; GlobalValue::GetValueByName("simTime", simV);
  double simTime = simV.Get();
  StringValue outV; GlobalValue::GetValueByName("outDir", outV);
  fs::path outDir = outV.Get();

  // Read E2/Control GlobalValues (kept for completeness)
  // ... (lines 538-580 are kept) ...
  BooleanValue booleanValue;
  StringValue stringValue;
  DoubleValue doubleValue;

  GlobalValue::GetValueByName ("useSemaphores", booleanValue);
  bool useSemaphores = booleanValue.Get ();
  GlobalValue::GetValueByName ("controlFileName", stringValue);
  std::string controlFilename = stringValue.Get ();
  GlobalValue::GetValueByName ("e2lteEnabled", booleanValue);
  bool e2lteEnabled = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2nrEnabled", booleanValue);
  bool e2nrEnabled = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2du", booleanValue);
  bool e2du = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2cuUp", booleanValue);
  bool e2cuUp = booleanValue.Get ();
  GlobalValue::GetValueByName ("e2cuCp", booleanValue);
  bool e2cuCp = booleanValue.Get ();
  GlobalValue::GetValueByName ("reducedPmValues", booleanValue);
  bool reducedPmValues = booleanValue.Get ();
  GlobalValue::GetValueByName ("indicationPeriodicity", doubleValue);
  double indicationPeriodicity = doubleValue.Get ();
  GlobalValue::GetValueByName ("e2TermIp", stringValue);
  std::string e2TermIp = stringValue.Get ();
  GlobalValue::GetValueByName ("enableE2FileLogging", booleanValue);
  bool enableE2FileLogging = booleanValue.Get ();

  NS_LOG_UNCOND ("e2lteEnabled " << e2lteEnabled << " e2nrEnabled " << e2nrEnabled << " e2du "
                                 << e2du << " e2cuCp " << e2cuCp << " e2cuUp " << e2cuUp
                                 << " controlFilename " << controlFilename
                                 << " useSemaphores " << useSemaphores
                                 << " indicationPeriodicity " << indicationPeriodicity
                                 << " reducedPmValues " << reducedPmValues
                                 << " e2TermIp " << e2TermIp);

  //-----------------------E2 CONFIGURATION----------------------------
  // ... (lines 582-613 are kept) ...
  // E2 periodicity on the device
  Config::SetDefault("ns3::MmWaveEnbNetDevice::E2Periodicity",
                     DoubleValue(indicationPeriodicity));

  // Helper-level E2 config
  Config::SetDefault("ns3::MmWaveHelper::E2ModeLte",
                     BooleanValue(e2lteEnabled));
  Config::SetDefault("ns3::MmWaveHelper::E2ModeNr",
                     BooleanValue(e2nrEnabled));
  Config::SetDefault("ns3::MmWaveHelper::E2Periodicity",
                     DoubleValue(indicationPeriodicity));
  Config::SetDefault("ns3::MmWaveHelper::E2TermIp",
                     StringValue(e2TermIp));

  // Device-level E2 report switches
  Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableDuReport",
                     BooleanValue(e2du));
  Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableCuUpReport",
                     BooleanValue(e2cuUp));
  Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableCuCpReport",
                     BooleanValue(e2cuCp));
  Config::SetDefault("ns3::MmWaveEnbNetDevice::EnableE2FileLogging",
                     BooleanValue(enableE2FileLogging));
  Config::SetDefault("ns3::MmWaveEnbNetDevice::ReducedPmValues",
                     BooleanValue(reducedPmValues));
  //-------------------------------------------------------------------

  // RF/system defaults
  Config::SetDefault("ns3::MmWavePhyMacCommon::CenterFreq", DoubleValue(28e9));
  Config::SetDefault("ns3::MmWavePhyMacCommon::Bandwidth",  DoubleValue(56e6));
  Config::SetDefault("ns3::MmWaveEnbPhy::TxPower",          DoubleValue(10.0));
  Config::SetDefault("ns3::MmWaveUePhy::NoiseFigure",       DoubleValue(7.0));

  fs::create_directories(outDir);
  fs::current_path(outDir);

  Ptr<MmWaveHelper> mmw = CreateObject<MmWaveHelper>();
  Ptr<MmWavePointToPointEpcHelper> epc = CreateObject<MmWavePointToPointEpcHelper>();
  mmw->SetEpcHelper(epc);

  // Use a different pathloss model more suited for a dense environment
  mmw->SetPathlossModelType("ns3::ThreeGppRmaPropagationLossModel"); // For open area
  mmw->SetChannelConditionModelType("ns3::ThreeGppChannelConditionModel");

  Ptr<Node> pgw = epc->GetPgwNode();

  // CONCERT SCENARIO ADAPTATION: SCALING NODES
  const uint32_t numUes = 200; // The crowd size
  NodeContainer gnb; gnb.Create(1);
  NodeContainer ue;  ue.Create(numUes); // Create 200 UEs
  NodeContainer rh;  rh.Create(1);

  InternetStackHelper ip; ip.Install(ue); ip.Install(rh);

  // CONCERT SCENARIO ADAPTATION: MOBILITY
  const Vector gnbPos = Vector(50, 50, 20); // Central gNB in 100x100 venue
  {
    // gNB Mobility
    MobilityHelper m;
    auto enbPos = CreateObject<ListPositionAllocator>();
    enbPos->Add(gnbPos);
    m.SetPositionAllocator(enbPos);
    m.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    m.Install(gnb);
  }

  // UE Crowd Mobility (Random Walk in a bounded area)
  MobilityHelper uem;
  uem.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                       "Bounds", RectangleValue(Rectangle(0.0, 100.0, 0.0, 100.0)), // 100x100m venue
                       "Speed", StringValue("ns3::UniformRandomVariable[Min=0.5|Max=2.0]"), // slow walking speed
                       "Distance", DoubleValue(5.0)); // change direction every 5m
  
  // Initial Position Allocator (spread UEs randomly in the 100x100 area)
  Ptr<RandomRectanglePositionAllocator> posAlloc = CreateObject<RandomRectanglePositionAllocator>();
  posAlloc->SetX(CreateObjectWith<UniformRandomVariable>(Rectangle(0.0, 100.0, 0.0, 100.0)));
  posAlloc->SetY(CreateObjectWith<UniformRandomVariable>(Rectangle(0.0, 100.0, 0.0, 100.0)));
  posAlloc->SetZ(CreateObjectWith<ConstantRandomVariable>("Constant", DoubleValue(1.5))); // standing height
  
  uem.SetPositionAllocator(posAlloc);
  uem.Install(ue);
  
  // Core + RH (Constant Position)
  {
    Ptr<Node> sgw = NodeList::GetNode(1);
    NodeContainer stationaryCoreNodes; stationaryCoreNodes.Add(pgw); stationaryCoreNodes.Add(sgw); stationaryCoreNodes.Add(rh.Get(0));
    MobilityHelper coreMobility; coreMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> corePositions = CreateObject<ListPositionAllocator>();
    corePositions->Add(Vector(20.0,25.0,0.0));
    corePositions->Add(Vector(20.0,30.0,0.0));
    corePositions->Add(Vector(20.0,20.0,0.0));
    coreMobility.SetPositionAllocator(corePositions);
    coreMobility.Install(stationaryCoreNodes);
  }

  BuildingsHelper::Install(gnb);
  BuildingsHelper::Install(ue);

  NetDeviceContainer gnbDevs = mmw->InstallEnbDevice(gnb);
  NetDeviceContainer ueDevs  = mmw->InstallUeDevice(ue);

  Ipv4InterfaceContainer ueIf = epc->AssignUeIpv4Address(ueDevs);
  Ipv4StaticRoutingHelper srt;
  for (uint32_t u=0; u<ue.GetN(); ++u) {
    Ptr<Ipv4StaticRouting> r = srt.GetStaticRouting(ue.Get(u)->GetObject<Ipv4>());
    r->SetDefaultRoute(epc->GetUeDefaultGatewayAddress(), 1);
  }

  mmw->AttachToClosestEnb(ueDevs, gnbDevs);

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
  p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
  NetDeviceContainer d = p2p.Install(pgw, rh.Get(0));
  Ipv4AddressHelper a; a.SetBase("10.0.0.0","255.0.0.0");
  a.Assign(d);
  Ipv4StaticRoutingHelper srh;
  srh.GetStaticRouting(rh.Get(0)->GetObject<Ipv4>())
     ->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

  // CONCERT SCENARIO ADAPTATION: MASSIVE TRAFFIC LOAD
  const uint16_t basePort = 4000;
  std::vector<Ptr<PacketSink>> allSinks; 

  for (uint32_t i = 0; i < numUes; ++i) {
    // 1. Install Sink on each UE (Port increases by 1 for each UE)
    PacketSinkHelper sink("ns3::UdpSocketFactory", 
                          InetSocketAddress(Ipv4Address::GetAny(), basePort + i));
    ApplicationContainer sinkApps = sink.Install(ue.Get(i));
    sinkApps.Start(Seconds(0.2));
    allSinks.push_back(DynamicCast<PacketSink>(sinkApps.Get(0)));

    // 2. Install OnOff Traffic Source from RH to each UE (80/20 Traffic Split)
    std::string dataRate;
    if (i % 5 == 0) { // Every 5th user (20%) is a streamer
        dataRate = "50Mbps"; // Heavy streamer
    } else {
        dataRate = "5Mbps";  // Light social media/browsing
    }
    
    OnOffHelper cbr("ns3::UdpSocketFactory", 
                    InetSocketAddress(ueIf.GetAddress(i), basePort + i));
    cbr.SetAttribute("OnTime",  StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
    cbr.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
    cbr.SetAttribute("DataRate", StringValue(dataRate));
    cbr.SetAttribute("PacketSize", UintegerValue(1200));
    cbr.Install(rh.Get(0)).Start(Seconds(0.35));
  }

  // NOTE: Disabling tracing for performance on a personal PC.
  // mmw->EnableTraces(); 

  const double covRadius = 100.0;
  Simulator::Schedule(Seconds(0.1), &SampleAll,
                      std::ref(ue), std::ref(ueDevs), gnb.Get(0),
                      covRadius, std::ref(allSinks), 0.1); // Pass all sinks

  // NetAnim is disabled for performance:
  // AnimationInterface anim("NetAnimFile_concert.xml");

  // --- CONCERT SCENARIO ORCHESTRATION EVENTS ---
  
  // 1. Scheduled Resource Uplift (Concert Start Anticipation)
  // At 10s, the "AI Orchestrator" pre-emptively sets a high, fixed MCS (e.g., MCS 25) 
  // to ensure maximum capacity for the concert's anticipated peak demand.
  Simulator::Schedule(Seconds(10.0), [gnb]() { 
    ChangeMcs(gnb.Get(0), 25);
    std::cerr << "\n"
              << "==============================================================\n"
              << "== [ORCHESTRATION EVENT] RESOURCE UPLIFT TRIGGERED (Pre-Empt) ==\n"
              << "== Time: " << Simulator::Now().GetSeconds() << "s                                       ==\n"
              << "== Action: Fixed MCS set to 25 (High throughput mode)       ==\n"
              << "==============================================================\n" << std::endl;
  });

  // 2. Peak Demand Simulation (Simulate the high load from the event)
  // At 20s, the light users boost their rate (e.g., mass video uploads/streaming)
  Simulator::Schedule(Seconds(20.0), &TriggerPeakLoad, rh);

  // 3. Post-Event Restoration
  // At 50s, the network resources are returned to Adaptive Mode.
  Simulator::Schedule(Seconds(50.0), [gnb]() { 
    ChangeMcs(gnb.Get(0), -1); // Restore adaptive
    std::cerr << "\n"
              << "==============================================================\n"
              << "== [ORCHESTRATION EVENT] END OF PEAK (Restoring Adaptive RRM) ==\n"
              << "== Time: " << Simulator::Now().GetSeconds() << "s                                       ==\n"
              << "== Action: Adaptive MCS restored                            ==\n"
              << "==============================================================\n" << std::endl;
  });

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  Simulator::Destroy();
  return 0;
}