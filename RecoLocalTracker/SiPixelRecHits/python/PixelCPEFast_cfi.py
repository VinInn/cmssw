import FWCore.ParameterSet.Config as cms

PixelCPEFastESProducer = cms.ESProducer("PixelCPEFastESProducer",

    ComponentName = cms.string('PixelCPEFast'),
    Alpha2Order = cms.bool(True),

    # Allows cuts to be optimized
    eff_charge_cut_lowX = cms.double(0.0),
    eff_charge_cut_lowY = cms.double(0.0),
    eff_charge_cut_highX = cms.double(1.0),
    eff_charge_cut_highY = cms.double(1.0),
    size_cutX = cms.double(3.0),
    size_cutY = cms.double(3.0),

    # Edge cluster errors in microns (determined by looking at residual RMS) 
    EdgeClusterErrorX = cms.double( 50.0 ),                                      
    EdgeClusterErrorY = cms.double( 85.0 ),                                                     

    # Can use errors predicted by the template code
    # If UseErrorsFromTemplates is False, must also set
    # TruncatePixelCharge, IrradiationBiasCorrection, DoCosmics and LoadTemplatesFromDB to be False                                        
    UseErrorsFromTemplates = cms.bool(True),

    # When set True this gives a slight improvement in resolution at no cost 
    TruncatePixelCharge = cms.bool(True),

    # petar, for clusterProbability() from TTRHs
    ClusterProbComputationFlag = cms.int32(0),

    #MagneticFieldRecord: e.g. "" or "ParabolicMF"
    MagneticFieldRecord = cms.ESInputTag(""),
)
