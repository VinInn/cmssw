#ifndef __BlockElementLinkerBase_H__
#define __BlockElementLinkerBase_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include <string>

class BlockElementLinkerBase {
public:

   enum Type { PreshowerAndECALLinker
                 , HFEMAndHFHADLinker
                 , ECALAndHCALLinker
                 , TrackAndHCALLinker
                 , TrackAndECALLinker
                 , DefaultType };

 BlockElementLinkerBase(const edm::ParameterSet& conf, Type itype=DefaultType):
  m_type(itype),
  _linkerName( conf.getParameter<std::string>("linkerName") ) { }
  BlockElementLinkerBase(const BlockElementLinkerBase& ) = delete;
  BlockElementLinkerBase& operator=(const BlockElementLinkerBase&) = delete;

  virtual bool linkPrefilter( const reco::PFBlockElement*,
			      const reco::PFBlockElement* ) const 
  { return true; }

  virtual double testLink( const reco::PFBlockElement*,
			   const reco::PFBlockElement* ) const = 0;

  Type type() const{ return m_type;}

  const std::string& name() const { return _linkerName; }
  
 private:
  Type m_type;
  const std::string _linkerName;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< BlockElementLinkerBase* (const edm::ParameterSet&) > BlockElementLinkerFactory;

#endif
