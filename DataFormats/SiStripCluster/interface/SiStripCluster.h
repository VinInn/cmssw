#ifndef DATAFORMATS_SISTRIPCLUSTER_H
#define DATAFORMATS_SISTRIPCLUSTER_H

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <vector>
#include <array>
#include <numeric>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class SiStripCluster {
public:
  static constexpr int MAXSIZE=23;
  using Container=std::array<uint8_t,MAXSIZE+1>;

  typedef std::vector<SiStripDigi>::const_iterator SiStripDigiIter;
  typedef std::pair<SiStripDigiIter, SiStripDigiIter> SiStripDigiRange;

  static const uint16_t stripIndexMask = 0x7FFF;   // The first strip index is in the low 15 bits of firstStrip_
  static const uint16_t mergedValueMask = 0x8000;  // The merged state is given by the high bit of firstStrip_

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */

  SiStripCluster() {}


  explicit SiStripCluster(const SiStripDigiRange& range);


  template <typename Iter>
  SiStripCluster(const uint16_t& firstStrip, Iter begin, Iter end) : firstStrip_(firstStrip) { init(begin,end);}

  template <typename Iter>
  SiStripCluster(const uint16_t& firstStrip, Iter begin, Iter end, bool merged)
      : firstStrip_(firstStrip) {
    init(begin,end);
    if (merged)
      firstStrip_ |= mergedValueMask;  // if this is a candidate merged cluster
  }

  template <typename Iter>
  void init(Iter begin, Iter end) { 
   int isize = std::min(MAXSIZE,int(end-begin));
   for(int i=0; i<isize;++i)
    amplitudes_[i]= *(begin++);
   amplitudes_[MAXSIZE]=isize;
  }

  // extend the cluster 
  template <typename Iter>
  void extend(Iter begin, Iter end) { 
   int isize = std::min(MAXSIZE,size()+int(end-begin));
   for(int i=size(); i<isize;++i)
    amplitudes_[i]= *(begin++);
   amplitudes_[MAXSIZE]=isize;
  }

  /** The number of the first strip in the cluster.
   *  The high bit of firstStrip_ indicates whether the cluster is a candidate for being merged.
   */
  uint16_t firstStrip() const { return firstStrip_ & stripIndexMask; }
  
  uint16_t endStrip() const { return firstStrip()+size(); }

  /** The amplitudes of the strips forming the cluster.
   *  The amplitudes are on consecutive strips; if a strip is missing
   *  the amplitude is set to zero.
   *  A strip may be missing in the middle of a cluster because of a
   *  clusterizer that accepts holes.
   *  A strip may also be missing anywhere in the cluster, including the 
   *  edge, to record a dead/noisy channel.
   *
   *  You can find the special meanings of values { 0, 254, 255} in section 3.4.1 of
   *  http://www.te.rl.ac.uk/esdg/cms-fed/firmware/Documents/FE_FPGA_Technical_Description.pdf
   */
   uint8_t const * begin() const { return amplitudes_.data();}
   uint8_t const * end() const { return begin()+size();}
   uint8_t size() const { return amplitudes_[MAXSIZE];}
   uint8_t  operator[](int i) const { return *(begin()+i);}
   bool empty() const { return 0==size();}
   bool full() const { return int(size())==MAXSIZE;}

   SiStripCluster const & amplitudes() const { return *this; }


  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  should not be used as position estimate for tracking.
   */
  float barycenter() const;

  /** total charge
   *
   */
  int charge() const { return std::accumulate(begin(), end(), int(0)); }

  /** Test (set) the merged status of the cluster
   *
   */
  bool isMerged() const { return (firstStrip_ & mergedValueMask) != 0; }
  void setMerged(bool mergedState) { mergedState ? firstStrip_ |= mergedValueMask : firstStrip_ &= stripIndexMask; }

  float getSplitClusterError() const { return error_x; }
  void setSplitClusterError(float errx) { error_x = errx; }

private:
  Container amplitudes_;

  uint16_t firstStrip_ = 0;

  uint16_t error_x = 0; // in um
};

// Comparison operators
inline bool operator<(const SiStripCluster& one, const SiStripCluster& other) {
  return one.firstStrip() < other.firstStrip();
}

inline bool operator<(const SiStripCluster& cluster, const uint16_t& firstStrip) {
  return cluster.firstStrip() < firstStrip;
}

inline bool operator<(const uint16_t& firstStrip, const SiStripCluster& cluster) {
  return firstStrip < cluster.firstStrip();
}
#endif  // DATAFORMATS_SISTRIPCLUSTER_H
