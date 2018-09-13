#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseParams.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "FWCore/Utilities/interface/isFinite.h"

void ClusEllipseParams::fill(const SiPixelRecHit & recHit, const TrackerTopology & tkTpl, bool raw){

      auto const clus = *recHit.cluster();
      auto const& topol = reinterpret_cast<const PixelGeomDetUnit*>(recHit.detUnit())->specificTopology();
      if (clus.minPixelCol()==0) return;
      if (clus.maxPixelCol()+1==topol.ncolumns()) return;
      if (clus.minPixelRow()==0) return;
      if (clus.maxPixelRow()+1==topol.nrows()) return;
      if (clus.minPixelRow()<=79 && clus.maxPixelRow()>=80) return;

      auto dc = 52-clus.minPixelCol()%52;
      if (dc==52) return;
      bool hasB = dc==52 || clus.size()>=dc;

      m_isBarrel = (recHit.geographicalId().subdetId() == int(PixelSubdetector::PixelBarrel));
      m_layer =  tkTpl.layer(recHit.geographicalId());

     float qx=0, qy=0, q2x=0, q2y=0 ,qxy=0, q=0;     
      int isize = clus.pixelADC().size();
      
      for (int i=0; i<isize; ++i) {
        float yo = clus.pixelOffset()[i*2+1]<dc ? 0.0f : ( clus.pixelOffset()[i*2+1]==dc ? 0.5f : 1.f);
        auto c = float(clus.pixelADC()[i]);
        auto x = float(clus.pixelOffset()[i*2]);
        auto y = float(clus.pixelOffset()[i*2+1])+yo;
        q+=c; qx+=c*x;qy+=c*y; 
        q2x+=c*x*x; q2y+=c*y*y;
        qxy+=c*x*y;
      }
      qx /=q; qy /=q; 
      q2x = q2x/q - qx*qx; q2y = q2y/q - qy*qy;
      qxy = qxy/q - qx*qy;
      auto tr = q2x+q2y; auto det = q2x*q2y-qxy*qxy;
      auto l1 = 0.5f*(tr + std::sqrt(tr*tr-4.f*det));
      auto l2 = l1>0 ? det/l1 : 0;
      auto ly = q2y>0 ? l1-q2x : 0.f; auto lx = q2y>0 ?  qxy : 1.f; auto norm = 4.f*std::sqrt(l1/(lx*lx+ly*ly));
      // auto ll = 4.f*std::sqrt(l1);
      lx *=norm; ly*=norm;
      m_x  = qx;
      m_y  = qy; 
      m_dx = lx;
      m_dy = ly;
      m_l2 = 4.f*sqrt(l2);
      if (!edm::isFinite(m_l2)) m_l2=0;
      m_sx = clus.sizeX();
      m_sy = clus.sizeY() + (hasB?1:0);

      if (raw) return;

      // normalize as input of DNN

      m_x /= m_sx;
      m_y /= m_sy;

      constexpr float xw = 4.;

      m_dx *= xw;
      m_sx *= xw;

}
