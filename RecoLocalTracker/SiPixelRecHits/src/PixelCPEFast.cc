#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// this is needed to get errors from templates
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "DataFormats/DetId/interface/DetId.h"


// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "boost/multi_array.hpp"

#include <iostream>
using namespace std;

namespace {
   constexpr float micronsToCm = 1.0e-4;
   const bool MYDEBUG = false;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(edm::ParameterSet const & conf,
                                 const MagneticField * mag,
                                 const TrackerGeometry& geom,
                                 const TrackerTopology& ttopo,
                                 const SiPixelLorentzAngle * lorentzAngle,
                                 const SiPixelGenErrorDBObject * genErrorDBObject,
                                 const SiPixelLorentzAngle * lorentzAngleWidth=nullptr)
: PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr,lorentzAngleWidth,0) {
   
   if (theVerboseLevel > 0)
      LogDebug("PixelCPEFast")
      << " constructing a generic algorithm for ideal pixel detector.\n"
      << " CPEGeneric:: VerboseLevel = " << theVerboseLevel;
   
   EdgeClusterErrorX_ = conf.getParameter<double>("EdgeClusterErrorX");
   EdgeClusterErrorY_ = conf.getParameter<double>("EdgeClusterErrorY");
   
   
   UseErrorsFromTemplates_    = conf.getParameter<bool>("UseErrorsFromTemplates");
   TruncatePixelCharge_       = conf.getParameter<bool>("TruncatePixelCharge");
   
   
   // Use errors from templates or from GenError
   if ( UseErrorsFromTemplates_ ) {
     if ( !SiPixelGenError::pushfile( *genErrorDBObject_, thePixelGenError_) )
            throw cms::Exception("InvalidCalibrationLoaded")
            << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
            << ( *genErrorDBObject_ ).version();
         if(MYDEBUG) cout<<"Loaded genErrorDBObject v"<<( *genErrorDBObject_ ).version()<<endl;
   }  else {
      if(MYDEBUG) cout<<" Use simple parametrised errors "<<endl;
   } // if ( UseErrorsFromTemplates_ )
   
   
   // Rechit errors in case other, more correct, errors are not vailable
   // This are constants. Maybe there is a more efficienct way to store them.
      xerr_barrel_l1_= {0.00115, 0.00120, 0.00088};
      xerr_barrel_l1_def_=0.01030;
      yerr_barrel_l1_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
      yerr_barrel_l1_def_=0.00210;
      xerr_barrel_ln_= {0.00115, 0.00120, 0.00088};
      xerr_barrel_ln_def_=0.01030;
      yerr_barrel_ln_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
      yerr_barrel_ln_def_=0.00210;
      xerr_endcap_= {0.0020, 0.0020};
      xerr_endcap_def_=0.0020;
      yerr_endcap_= {0.00210};
      yerr_endcap_def_=0.00075;

   
   
}

PixelCPEBase::ClusterParam* PixelCPEFast::createClusterParam(const SiPixelCluster & cl) const
{
   return new ClusterParamGeneric(cl);
}



//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers)
//! into the local frame (in centimeters).
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEFast::localPosition(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
   
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);

   assert(!theClusterParam.with_track_angle); 
   
   float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
   float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
   float shiftX = 0.5f*theDetParam.lorentzShiftInCmX;
   float shiftY = 0.5f*theDetParam.lorentzShiftInCmY;
   
   if ( UseErrorsFromTemplates_ ) {
      
      float qclus = theClusterParam.theCluster->charge();
      float locBz = theDetParam.bz;
      float locBx = theDetParam.bx;
      //cout << "PixelCPEFast::localPosition(...) : locBz = " << locBz << endl;
      
      theClusterParam.pixmx  = -999.9; // max pixel charge for truncation of 2-D cluster
      theClusterParam.sigmay = -999.9; // CPE Generic y-error for multi-pixel cluster
      theClusterParam.sigmax = -999.9; // CPE Generic x-error for multi-pixel cluster
      theClusterParam.sy1    = -999.9; // CPE Generic y-error for single single-pixel
      theClusterParam.sy2    = -999.9; // CPE Generic y-error for single double-pixel cluster
      theClusterParam.sx1    = -999.9; // CPE Generic x-error for single single-pixel cluster
      theClusterParam.sx2    = -999.9; // CPE Generic x-error for single double-pixel cluster
      
      float dummy;
      
      SiPixelGenError gtempl(thePixelGenError_);
      int gtemplID_ = theDetParam.detTemplateId;
      
      theClusterParam.qBin_ = gtempl.qbin( gtemplID_, theClusterParam.cotalpha, theClusterParam.cotbeta, locBz, locBx, qclus, 
                                          false,
                                          theClusterParam.pixmx, theClusterParam.sigmay, dummy,
                                          theClusterParam.sigmax, dummy, theClusterParam.sy1,
                                          dummy, theClusterParam.sy2, dummy, theClusterParam.sx1,
                                          dummy, theClusterParam.sx2, dummy );
      
      
      theClusterParam.sigmax = theClusterParam.sigmax * micronsToCm;
      theClusterParam.sx1 = theClusterParam.sx1 * micronsToCm;
      theClusterParam.sx2 = theClusterParam.sx2 * micronsToCm;
      
      theClusterParam.sigmay = theClusterParam.sigmay * micronsToCm;
      theClusterParam.sy1 = theClusterParam.sy1 * micronsToCm;
      theClusterParam.sy2 = theClusterParam.sy2 * micronsToCm;
      
   } // if ( UseErrorsFromTemplates_ )
   else {
     theClusterParam.qBin_ = 0;
   }
   
   int Q_f_X;        //!< Q of the first  pixel  in X
   int Q_l_X;        //!< Q of the last   pixel  in X
   int Q_f_Y;        //!< Q of the first  pixel  in Y
   int Q_l_Y;        //!< Q of the last   pixel  in Y
   collect_edge_charges( theClusterParam,
                        Q_f_X, Q_l_X,
                        Q_f_Y, Q_l_Y );
   
   //--- Find the inner widths along X and Y in one shot.  We
   //--- compute the upper right corner of the inner pixels
   //--- (== lower left corner of upper right pixel) and
   //--- the lower left corner of the inner pixels
   //--- (== upper right corner of lower left pixel), and then
   //--- subtract these two points in the formula.
   
   //--- Upper Right corner of Lower Left pixel -- in measurement frame
   MeasurementPoint meas_URcorn_LLpix( theClusterParam.theCluster->minPixelRow()+1.0,
                                      theClusterParam.theCluster->minPixelCol()+1.0 );
   
   //--- Lower Left corner of Upper Right pixel -- in measurement frame
   MeasurementPoint meas_LLcorn_URpix( theClusterParam.theCluster->maxPixelRow(),
                                      theClusterParam.theCluster->maxPixelCol() );
   
   //--- These two now converted into the local
   LocalPoint local_URcorn_LLpix;
   LocalPoint local_LLcorn_URpix;
   
      local_URcorn_LLpix = theDetParam.theTopol->localPosition(meas_URcorn_LLpix);
      local_LLcorn_URpix = theDetParam.theTopol->localPosition(meas_LLcorn_URpix);
   
   
   
   float xPos =
   generic_position_formula( theClusterParam.theCluster->sizeX(),
                            Q_f_X, Q_l_X,
                            local_URcorn_LLpix.x(), local_LLcorn_URpix.x(),
                            chargeWidthX,   // lorentz shift in cm
                            theDetParam.theThickness,
                            theClusterParam.cotalpha,
                            theDetParam.thePitchX,
                            theDetParam.theRecTopol->isItBigPixelInX( theClusterParam.theCluster->minPixelRow() ),
                            theDetParam.theRecTopol->isItBigPixelInX( theClusterParam.theCluster->maxPixelRow() )
                           );   
   
   // apply the lorentz offset correction
   xPos = xPos + shiftX;
   
   float yPos =
   generic_position_formula( theClusterParam.theCluster->sizeY(),
                            Q_f_Y, Q_l_Y,
                            local_URcorn_LLpix.y(), local_LLcorn_URpix.y(),
                            chargeWidthY,   // lorentz shift in cm
                            theDetParam.theThickness,
                            theClusterParam.cotbeta,
                            theDetParam.thePitchY,
                            theDetParam.theRecTopol->isItBigPixelInY( theClusterParam.theCluster->minPixelCol() ),
                            theDetParam.theRecTopol->isItBigPixelInY( theClusterParam.theCluster->maxPixelCol() )
                           );   
   // apply the lorentz offset correction
   yPos = yPos + shiftY;
   
   //--- Now put the two together
   LocalPoint pos_in_local( xPos, yPos );
   return pos_in_local;
}



//-----------------------------------------------------------------------------
//!  A generic version of the position formula.  Since it works for both
//!  X and Y, in the interest of the simplicity of the code, all parameters
//!  are passed by the caller.  The only class variable used by this method
//!  is the theThickness, since that's common for both X and Y.
//-----------------------------------------------------------------------------
float
PixelCPEFast::
generic_position_formula( int size,                //!< Size of this projection.
                         int Q_f,              //!< Charge in the first pixel.
                         int Q_l,              //!< Charge in the last pixel.
                         float upper_edge_first_pix, //!< As the name says.
                         float lower_edge_last_pix,  //!< As the name says.
                         float lorentz_shift,   //!< L-shift at half thickness
                         float theThickness,   //detector thickness
                         float cot_angle,        //!< cot of alpha_ or beta_
                         float pitch,            //!< thePitchX or thePitchY
                         bool first_is_big,       //!< true if the first is big
                         bool last_is_big        //!< true if the last is big
) const
{
   
   float geom_center = 0.5f * ( upper_edge_first_pix + lower_edge_last_pix );
   
   //--- The case of only one pixel in this projection is separate.  Note that
   //--- here first_pix == last_pix, so the average of the two is still the
   //--- center of the pixel.
   if ( size == 1 ) {return geom_center;}

   float W_eff; // the compiler detects the logic below (and warns if buggy!!!!0 
   bool simple=true;
   if (size==2) {   
     //--- Width of the clusters minus the edge (first and last) pixels.
     //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
     float W_inner      = lower_edge_last_pix - upper_edge_first_pix;  // in cm
   
     //--- Predicted charge width from geometry
     float W_pred = theThickness * cot_angle                     // geometric correction (in cm)
                    - lorentz_shift;                    // (in cm) &&& check fpix!
   
     W_eff = std::abs( W_pred ) - W_inner;

     //--- If the observed charge width is inconsistent with the expectations
     //--- based on the track, do *not* use W_pred-W_innner.  Instead, replace
     //--- it with an *average* effective charge width, which is the average
     //--- length of the edge pixels.
     //
     simple = ( W_eff < 0.0f ) | ( W_eff > pitch );

   }
   if (simple) {
     //--- Total length of the two edge pixels (first+last)
     float sum_of_edge = 2.0f;
     if (first_is_big) sum_of_edge += 1.0f;
     if (last_is_big)  sum_of_edge += 1.0f;
     W_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
   }
   
   
   //--- Finally, compute the position in this projection
   float Qdiff = Q_l - Q_f;
   float Qsum  = Q_l + Q_f;
   
   //--- Temporary fix for clusters with both first and last pixel with charge = 0
   if(Qsum==0) Qsum=1.0f;
   float hit_pos = geom_center + 0.5f*(Qdiff/Qsum) * W_eff;
   
   return hit_pos;
}


//-----------------------------------------------------------------------------
//!  Collect the edge charges in x and y, in a single pass over the pixel vector.
//!  Calculate charge in the first and last pixel projected in x and y
//!  and the inner cluster charge, projected in x and y.
//-----------------------------------------------------------------------------
void
PixelCPEFast::
collect_edge_charges(ClusterParam & theClusterParamBase,  //!< input, the cluster
                     int & Q_f_X,              //!< output, Q first  in X
                     int & Q_l_X,              //!< output, Q last   in X
                     int & Q_f_Y,              //!< output, Q first  in Y
                     int & Q_l_Y               //!< output, Q last   in Y
) const
{
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);
   
   // Initialize return variables.
   Q_f_X = Q_l_X = 0;
   Q_f_Y = Q_l_Y = 0;
   
   
   // Obtain boundaries in index units
   int xmin = theClusterParam.theCluster->minPixelRow();
   int xmax = theClusterParam.theCluster->maxPixelRow();
   int ymin = theClusterParam.theCluster->minPixelCol();
   int ymax = theClusterParam.theCluster->maxPixelCol();
   
   
   // Iterate over the pixels.
   int isize = theClusterParam.theCluster->size();
   for (int i = 0;  i != isize; ++i)
   {
      auto const & pixel = theClusterParam.theCluster->pixel(i);
      // ggiurgiu@fnal.gov: add pixel charge truncation
      int pix_adc = pixel.adc;
      if ( UseErrorsFromTemplates_ && TruncatePixelCharge_ )
         pix_adc = std::min(pix_adc, theClusterParam.pixmx );
      
      //
      // X projection
      if ( pixel.x == xmin ) Q_f_X += pix_adc;
      if ( pixel.x == xmax ) Q_l_X += pix_adc;
      //
      // Y projection
      if ( pixel.y == ymin ) Q_f_Y += pix_adc;
      if ( pixel.y == ymax ) Q_l_Y += pix_adc;
   }
   
   return;
}


//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError
PixelCPEFast::localError(DetParam const & theDetParam,  ClusterParam & theClusterParamBase) const
{
   
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);
   
   // Default errors are the maximum error used for edge clusters.
   // These are determined by looking at residuals for edge clusters
   float xerr = EdgeClusterErrorX_ * micronsToCm;
   float yerr = EdgeClusterErrorY_ * micronsToCm;
   
   
   // Find if cluster is at the module edge.
   int maxPixelCol = theClusterParam.theCluster->maxPixelCol();
   int maxPixelRow = theClusterParam.theCluster->maxPixelRow();
   int minPixelCol = theClusterParam.theCluster->minPixelCol();
   int minPixelRow = theClusterParam.theCluster->minPixelRow();
   
   bool edgex = ( theDetParam.theRecTopol->isItEdgePixelInX( minPixelRow ) ) || ( theDetParam.theRecTopol->isItEdgePixelInX( maxPixelRow ) );
   bool edgey = ( theDetParam.theRecTopol->isItEdgePixelInY( minPixelCol ) ) || ( theDetParam.theRecTopol->isItEdgePixelInY( maxPixelCol ) );
   
   unsigned int sizex = theClusterParam.theCluster->sizeX();
   unsigned int sizey = theClusterParam.theCluster->sizeY();
   
   // Find if cluster contains double (big) pixels.
   bool bigInX = theDetParam.theRecTopol->containsBigPixelInX( minPixelRow, maxPixelRow );
   bool bigInY = theDetParam.theRecTopol->containsBigPixelInY( minPixelCol, maxPixelCol );
   
   if (UseErrorsFromTemplates_ ) {
      //
      // Use template errors
      
      if ( !edgex ) { // Only use this for non-edge clusters
         if ( sizex == 1 ) {
            if ( !bigInX ) {xerr = theClusterParam.sx1;}
            else           {xerr = theClusterParam.sx2;}
         } else {xerr = theClusterParam.sigmax;}
      }
      
      if ( !edgey ) { // Only use for non-edge clusters
         if ( sizey == 1 ) {
            if ( !bigInY ) {yerr = theClusterParam.sy1;}
            else           {yerr = theClusterParam.sy2;}
         } else {yerr = theClusterParam.sigmay;}
      }
      
   } else  { // simple errors
      
      // This are the simple errors, hardcoded in the code
      //cout << "Track angles are not known " << endl;
      //cout << "Default angle estimation which assumes track from PV (0,0,0) does not work." << endl;
      
      if ( GeomDetEnumerators::isTrackerPixel(theDetParam.thePart) ) {
         if(GeomDetEnumerators::isBarrel(theDetParam.thePart)) {
            
            DetId id = (theDetParam.theDet->geographicalId());
            int layer=ttopo_.layer(id);
            if ( layer==1 ) {
               if ( !edgex ) {
                  if ( sizex<=xerr_barrel_l1_.size() ) xerr=xerr_barrel_l1_[sizex-1];
                  else xerr=xerr_barrel_l1_def_;
               }
               
               if ( !edgey ) {
                  if ( sizey<=yerr_barrel_l1_.size() ) yerr=yerr_barrel_l1_[sizey-1];
                  else yerr=yerr_barrel_l1_def_;
               }
            } else{  // layer 2,3
               if ( !edgex ) {
                  if ( sizex<=xerr_barrel_ln_.size() ) xerr=xerr_barrel_ln_[sizex-1];
                  else xerr=xerr_barrel_ln_def_;
               }
               
               if ( !edgey ) {
                  if ( sizey<=yerr_barrel_ln_.size() ) yerr=yerr_barrel_ln_[sizey-1];
                  else yerr=yerr_barrel_ln_def_;
               }
            }
            
         } else { // EndCap
            
            if ( !edgex ) {
               if ( sizex<=xerr_endcap_.size() ) xerr=xerr_endcap_[sizex-1];
               else xerr=xerr_endcap_def_;
            }
            
            if ( !edgey ) {
               if ( sizey<=yerr_endcap_.size() ) yerr=yerr_endcap_[sizey-1];
               else yerr=yerr_endcap_def_;
            }
         } // end endcap
      }
      
   } // end 
   
   auto xerr_sq = xerr*xerr; 
   auto yerr_sq = yerr*yerr;
   
   return LocalError( xerr_sq, 0, yerr_sq );
   
}






