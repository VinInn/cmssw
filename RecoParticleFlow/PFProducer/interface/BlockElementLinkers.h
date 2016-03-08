#ifndef BlockElementLinkers_H
#define BlockElementLinkers_H

/*   
 * All possible types of BlockElementLinkerBase
 * with the intrusive RTTI....
 *
 */

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"

class TrackAndECALLinker final : public BlockElementLinkerBase {
public:
  TrackAndECALLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf, BlockElementLinkerBase::TrackAndECALLinker),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  bool linkPrefilter( const reco::PFBlockElement*,
		      const reco::PFBlockElement* ) const override;

  double testLink( const reco::PFBlockElement*,
		   const reco::PFBlockElement* ) const override;

private:
  const bool _useKDTree,_debug;
};


class TrackAndHCALLinker final : public BlockElementLinkerBase {
public:
  TrackAndHCALLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf, BlockElementLinkerBase::TrackAndHCALLinker),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}

  double testLink
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};


class PreshowerAndECALLinker final : public BlockElementLinkerBase {
public:
  PreshowerAndECALLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf, BlockElementLinkerBase::PreshowerAndECALLinker),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}
  
  bool linkPrefilter( const reco::PFBlockElement*,
		      const reco::PFBlockElement* ) const override;

  double testLink( const reco::PFBlockElement*,
		   const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};



class HFEMAndHFHADLinker final : public BlockElementLinkerBase {
public:
  HFEMAndHFHADLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf, BlockElementLinkerBase::HFEMAndHFHADLinker),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}

  double testLink
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};



class ECALAndHCALLinker final : public BlockElementLinkerBase {
public:
  ECALAndHCALLinker(const edm::ParameterSet& conf) :
    BlockElementLinkerBase(conf, BlockElementLinkerBase::ECALAndHCALLinker),
    _useKDTree(conf.getParameter<bool>("useKDTree")),
    _debug(conf.getUntrackedParameter<bool>("debug",false)) {}

  double testLink
  ( const reco::PFBlockElement*,
    const reco::PFBlockElement* ) const override;

private:
  bool _useKDTree,_debug;
};

namespace blockElementLinker {

  inline double testLink(BlockElementLinkerBase const & linker,
    const reco::PFBlockElement* a,
    const reco::PFBlockElement* b) {

    switch (linker.type()) {
      case BlockElementLinkerBase::PreshowerAndECALLinker :
         return reinterpret_cast<PreshowerAndECALLinker const&>(linker).testLink(a,b);
      case BlockElementLinkerBase::HFEMAndHFHADLinker :
      case BlockElementLinkerBase::ECALAndHCALLinker :
      case BlockElementLinkerBase::TrackAndHCALLinker :
      case BlockElementLinkerBase::TrackAndECALLinker :
      case BlockElementLinkerBase::DefaultType :
        return linker.testLink(a,b);
    }
    
    return 1; // make compiler happy

  }


  inline bool linkPrefilter(BlockElementLinkerBase const & linker,
    const reco::PFBlockElement* a,
    const reco::PFBlockElement* b) {

    switch (linker.type()) {
      case BlockElementLinkerBase::PreshowerAndECALLinker :
         return reinterpret_cast<PreshowerAndECALLinker const&>(linker).linkPrefilter(a,b);
      case BlockElementLinkerBase::HFEMAndHFHADLinker :
         return reinterpret_cast<HFEMAndHFHADLinker const&>(linker).linkPrefilter(a,b);
      case BlockElementLinkerBase::ECALAndHCALLinker :
         return reinterpret_cast<ECALAndHCALLinker const&>(linker).linkPrefilter(a,b);
      case BlockElementLinkerBase::TrackAndHCALLinker :
         return reinterpret_cast<TrackAndHCALLinker const&>(linker).linkPrefilter(a,b);
      case BlockElementLinkerBase::TrackAndECALLinker :
         return reinterpret_cast<TrackAndECALLinker const&>(linker).linkPrefilter(a,b);
      case BlockElementLinkerBase::DefaultType :
        return linker.linkPrefilter(a,b);
    }

    return true; // make compiler happy
  }

  
}


#endif // BlockElementLinkers_H

