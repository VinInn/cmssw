#ifndef RecoPixelVertexingPixelLowPtUtilitiesClusEllipseParams_H
#define RecoPixelVertexingPixelLowPtUtilitiesClusEllipseParams_H

/*
usecols=(n.index('isBarrel'), n.index('layer'), n.index('x'),n.index('y'),
                                n.index('dx'), n.index('dy'), n.index('l2'),
                                n.index('sx'), n.index('sy'),
)
*/

class SiPixelRecHit;
class TrackerTopology;

// input to the DNN
struct ClusEllipseParams {

  float m_isBarrel, m_layer=0, m_x, m_y, m_dx, m_dy, m_l2, m_sx, m_sy;

  float const * data() const { return &m_isBarrel;}
  void fill(const SiPixelRecHit & recHit, const TrackerTopology & tkTpl, bool raw=false);

};
#endif
