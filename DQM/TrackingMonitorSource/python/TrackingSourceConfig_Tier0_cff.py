import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
import RecoTracker.IterativeTracking.iterativeTkUtils as _utils

### load which are the tracks collection 2 be monitored
from DQM.TrackingMonitorSource.TrackCollections2monitor_cff import *

### load the different flavour of settings of the TrackingMonitor module
from DQM.TrackingMonitorSource.TrackerCollisionTrackingMonitor_cff import *


### define one EDAnalyzer per each track collection
### following suggestion 2. in
### https://hypernews.cern.ch/HyperNews/CMS/get/sw-develtools/1908/1.html
for tracks in selectedTracks :
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    locals()[label] = TrackerCollisionTrackMonCommon.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks])
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks])
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    if tracks == 'generalTracks':
        locals()[label].doEffFromHitPatternVsBX = False
    else:
        locals()[label].doEffFromHitPatternVsBX = doEffFromHitPatternVsBX[tracks]
#    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)
    
    # ZeroBias
    label = 'TrackerCollisionSelectedTrackMonMB' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonMB.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks])
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks])
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = cms.bool(True)
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)

    # ZeroBias, no hip, no OOT pu (1st collision after abort gap) ###
    label = 'TrackerCollisionSelectedTrackMonZBnoHIPnoOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBnoHIPnoOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/noHIP_noOOT_INpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/noHIP_noOOT_INpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)

    # ZeroBias, hip, no OOT pu (1st collision in train) ###
    label = 'TrackerCollisionSelectedTrackMonZBHIPnoOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBHIPnoOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/HIP_noOOT_INpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/HIP_noOOT_INpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)


    # ZeroBias, hip, OOT pu (1st collision after train) ###
    label = 'TrackerCollisionSelectedTrackMonZBHIPOOT' + str(tracks)                       
    locals()[label] = TrackerCollisionTrackMonZBHIPOOT.clone()
    locals()[label].TrackProducer    = cms.InputTag(tracks)
    locals()[label].FolderName       = cms.string(mainfolderName[tracks]+"/HIP_OOT_noINpu")
    locals()[label].PVFolderName     = cms.string(vertexfolderName[tracks]+"/HIP_OOT_noINpu")
    locals()[label].TrackPtMin       = trackPtMin[tracks]
    locals()[label].TrackPtBin       = trackPtN[tracks]
    locals()[label].TrackPtMax       = trackPtMax[tracks]
    locals()[label].TrackPBin        = trackPtN[tracks]
    locals()[label].TrackPMin        = trackPtMin[tracks]
    locals()[label].TrackPMax        = trackPtMax[tracks]
    locals()[label].doDCAPlots       = doPlotsPCA[tracks]
    locals()[label].doDCAwrtPVPlots  = doPlotsPCA[tracks]
    locals()[label].doDCAwrt000Plots = doPlotsPCA[tracks]
    locals()[label].doSIPPlots       = doPlotsPCA[tracks]
    locals()[label].numCut           = numCutString[tracks]
    locals()[label].denCut           = denCutString[tracks]
    locals()[label].doGoodTracksPlots                   = doGoodTracksPlots                   [tracks]
    locals()[label].doTrackerSpecific                   = doTrackerSpecific                   [tracks]
    locals()[label].doHitPropertiesPlots                = doHitPropertiesPlots                [tracks]
    locals()[label].doGeneralPropertiesPlots            = doGeneralPropertiesPlots            [tracks]
    locals()[label].doBeamSpotPlots                     = doBeamSpotPlots                     [tracks]
    locals()[label].doSeedParameterHistos               = doSeedParameterHistos               [tracks]
    locals()[label].doRecHitVsPhiVsEtaPerTrack          = doRecHitVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doRecHitVsPtVsEtaPerTrack           = doRecHitVsPtVsEtaPerTrack           [tracks]
    locals()[label].doGoodTrackRecHitVsPhiVsEtaPerTrack = doGoodTrackRecHitVsPhiVsEtaPerTrack [tracks]
    locals()[label].doLayersVsPhiVsEtaPerTrack          = doLayersVsPhiVsEtaPerTrack          [tracks]
    locals()[label].doGoodTrackLayersVsPhiVsEtaPerTrack = doGoodTrackLayersVsPhiVsEtaPerTrack [tracks]
    locals()[label].doPUmonitoring                      = doPUmonitoring                      [tracks]
    locals()[label].doPlotsVsBXlumi                     = doPlotsVsBXlumi                     [tracks]
    locals()[label].doPlotsVsGoodPVtx                   = doPlotsVsGoodPVtx                   [tracks]
    locals()[label].doPlotsVsLUMI                       = cms.bool(True)
    locals()[label].doPlotsVsBX                         = cms.bool(True)
    locals()[label].doEffFromHitPatternVsPU             = doEffFromHitPatternVsPU             [tracks]
    locals()[label].doEffFromHitPatternVsBX             = doEffFromHitPatternVsBX             [tracks]
    locals()[label].doEffFromHitPatternVsLUMI           = False
    locals()[label].doStopSource                        = doStopSource                        [tracks]    
    locals()[label].setLabel(label)


#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
import DQM.TrackingMonitor.TrackingMonitorSeed_cfi

from DQM.TrackingMonitorSource.IterTrackingModules4seedMonitoring_cfi import *
# Create first modules for all possible iterations, select later which
# ones to actually use based on era
for step in seedInputTag.iterkeys():
    label = 'TrackSeedMon'+str(step)
    locals()[label] = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone(
        doTrackCandHistos = cms.bool(True)
    )
    locals()[label].TrackProducer = cms.InputTag("generalTracks")
    locals()[label].FolderName    = cms.string("Tracking/TrackParameters/generalTracks")
    locals()[label].SeedProducer  = seedInputTag[step]
    locals()[label].TCProducer    = trackCandInputTag[step]
    locals()[label].AlgoName      = cms.string( str(step) )
    locals()[label].TkSeedSizeBin = trackSeedSizeBin[step]
    locals()[label].TkSeedSizeMin = trackSeedSizeMin[step]
    locals()[label].TkSeedSizeMax = trackSeedSizeMax[step]
    locals()[label].ClusterLabels = clusterLabel[step]
    if clusterLabel[step] == cms.vstring('Pix') :
        locals()[label].NClusPxBin = clusterBin[step]
        locals()[label].NClusPxMax = clusterMax[step]
    elif clusterLabel[step] == cms.vstring('Strip') or clusterLabel[step] == cms.vstring('Tot') :
        locals()[label].NClusStrBin = clusterBin[step]
        locals()[label].NClusStrMax = clusterMax[step]

# DQM Services
dqmInfoTracking = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Tracking')
)

# LogMessageMonitor ####
### load which are the module to monitor
from DQM.TrackingMonitorSource.EDModules2monitor_cfi import *

### load the different flavour of settings of the LogMessageMonitor module
from DQM.TrackingMonitorSource.LogMessageMonitor_cff import *

for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    locals()[label] = LogMessageMonCommon.clone()
    locals()[label].pluginsMonName = pluginsMonName[module]
    locals()[label].modules        = modulesLabel[module]
    locals()[label].categories     = categories[module]
    locals()[label].setLabel(label)

    label = str(module)+'LogMessageMonMB'
    locals()[label] = LogMessageMonMB.clone()
    locals()[label].pluginsMonName = pluginsMonName[module]
    locals()[label].modules        = modulesLabel[module]
    locals()[label].categories     = categories[module]
    locals()[label].setLabel(label)


# dEdx monitor ####
### load which dedx
from DQM.TrackingMonitorSource.dedxHarmonic2monitor_cfi import *

### load the different flavour of settings of the dEdxAnalyzer module
from DQM.TrackingMonitorSource.dEdxAnalyzer_cff import *


# temporary patch in order to have BXlumi 
from RecoLuminosity.LumiProducer.lumiProducer_cff import *

# import v0 monitoring
from DQM.TrackingMonitor.V0Monitor_cff import *

# temporary test in order to temporary produce the "goodPrimaryVertexCollection"
# define with a new name if changes are necessary, otherwise simply include
# it from CommonTools/ParticleFlow/python/goodOfflinePrimaryVertices_cfi.py
# uncomment when necessary
from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
from CommonTools.ParticleFlow.goodOfflinePrimaryVertices_cfi import goodOfflinePrimaryVertices
trackingDQMgoodOfflinePrimaryVertices = goodOfflinePrimaryVertices.clone()
trackingDQMgoodOfflinePrimaryVertices.filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) )
trackingDQMgoodOfflinePrimaryVertices.src=cms.InputTag('offlinePrimaryVertices')
trackingDQMgoodOfflinePrimaryVertices.filter = cms.bool(False)


# Event History Producer
from DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi import *


# Sequence
TrackingDQMSourceTier0 = cms.Sequence()
# dEdx monitoring
TrackingDQMSourceTier0 += consecutiveHEs * dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon   
#    # temporary patch in order to have BXlumi
#    * lumiProducer
# track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0 += cms.ignore(sequenceName[tracks])
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    TrackingDQMSourceTier0 += cms.ignore(locals()[label])
# seeding monitoring
for _eraName, _postfix, _era in _cfg.allEras():
    mvaSel = _utils.getMVASelectors(_postfix)
    _seq = cms.Sequence()
    for step in locals()["selectedIterTrackingStep"+_postfix]:
        if step in mvaSel:
            locals()["TrackSeedMon"+step].doMVAPlots = True
            locals()["TrackSeedMon"+step].TrackProducerForMVA = mvaSel[step][0]
            locals()["TrackSeedMon"+step].MVAProducers = mvaSel[step][1]

        _seq += locals()["TrackSeedMon"+step]
    if _eraName == "":
        locals()["TrackSeedMonSequence"] = _seq
    else:
        _era.toReplaceWith(TrackSeedMonSequence, _seq)
TrackingDQMSourceTier0 += TrackSeedMonSequence
# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    TrackingDQMSourceTier0 += locals()[label]
TrackingDQMSourceTier0 += voMonitoringSequence
TrackingDQMSourceTier0 += voWcutMonitoringSequence
TrackingDQMSourceTier0 += dqmInfoTracking


TrackingDQMSourceTier0Common = cms.Sequence()
# dEdx monitoring
TrackingDQMSourceTier0Common += (consecutiveHEs * dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon)    
## monitor track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0Common+=cms.ignore(sequenceName[tracks])
    label = 'TrackerCollisionSelectedTrackMonCommon' + str(tracks)
    TrackingDQMSourceTier0Common += cms.ignore(locals()[label])
# seeding monitoring
TrackingDQMSourceTier0Common += TrackSeedMonSequence
# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonCommon'
    TrackingDQMSourceTier0Common += locals()[label]
TrackingDQMSourceTier0Common += voMonitoringCommonSequence
TrackingDQMSourceTier0Common += voWcutMonitoringCommonSequence
TrackingDQMSourceTier0Common += dqmInfoTracking

TrackingDQMSourceTier0MinBias = cms.Sequence()
# dEdx monitoring
TrackingDQMSourceTier0MinBias += consecutiveHEs * dedxHarmonicSequence * dEdxMonCommon * dEdxHitMonCommon    
#    * lumiProducer
#    * trackingDQMgoodOfflinePrimaryVertices
# monitor track collections
for tracks in selectedTracks :
    if tracks != 'generalTracks':
        TrackingDQMSourceTier0MinBias += cms.ignore(sequenceName[tracks])

    for topology in [ 'MB', 'ZBnoHIPnoOOT', 'ZBHIPnoOOT', 'ZBHIPOOT']:
        label = 'TrackerCollisionSelectedTrackMon' + str(topology) + str(tracks)
        TrackingDQMSourceTier0MinBias += cms.ignore(locals()[label])
# seeding monitoring
TrackingDQMSourceTier0MinBias += TrackSeedMonSequence
# MessageLog
for module in selectedModules :
    label = str(module)+'LogMessageMonMB'
    TrackingDQMSourceTier0MinBias += locals()[label]
# V0 monitoring
TrackingDQMSourceTier0MinBias += voMonitoringMBSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBnoHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voMonitoringZBHIPOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringMBSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBnoHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBHIPnoOOTSequence
TrackingDQMSourceTier0MinBias += voWcutMonitoringZBHIPOOTSequence

TrackingDQMSourceTier0MinBias += dqmInfoTracking

