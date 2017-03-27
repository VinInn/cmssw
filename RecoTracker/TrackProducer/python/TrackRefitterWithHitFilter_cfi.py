import FWCore.ParameterSet.Config as cms

TrackRefitterWithHitFilter = cms.EDProducer("TrackRefitterWithHitFilter",
    src = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),

    ### fitting revoving some hits
    hitList  = cms.InputTag(''),                   

    ### Usually this parameter has not to be set True because 
    ### matched hits in the Tracker are already split when 
    ### the tracks are reconstructed the first time                         
    useHitsSplitting = cms.bool(False),

    TrajectoryInEvent = cms.bool(True),

    # this parameter decides if the propagation to the beam line
    # for the track parameters defiition is from the first hit
    # or from the closest to the beam line
    # true for cosmics, false for collision tracks (needed by loopers)
    GeometricInnerState = cms.bool(False),

    # Navigation school is necessary to fill the secondary hit patterns                         
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    MeasurementTracker = cms.string(''),                                              
    MeasurementTrackerEvent = cms.InputTag('MeasurementTrackerEvent'),                   
    #
    # in order to avoid to fill the secondary hit patterns and
    # refit the tracks more quickly 
    #NavigationSchool = cms.string('') 
)

# Switch back to GenericCPE until bias in template CPE gets fixed
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(TrackRefitterWithHitFilter, TTRHBuilder = 'WithTrackAngle') # FIXME
