/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2019 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2019 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2019 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All right reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 *  @file DruckerPrager.hpp
 */

#ifndef GEOSX_CONSTITUTIVE_SOLID_DRUCKERPRAGER_HPP
#define GEOSX_CONSTITUTIVE_SOLID_DRUCKERPRAGER_HPP

#include "ElasticIsotropic.hpp"
#include "SolidUtilities.hpp"
#include "LvArray/src/tensorOps.hpp" // TODO: cleanup includes

namespace geosx
{

namespace constitutive
{

/**
 * @class DruckerPragerUpdates
 *
 * Class to provide material updates that may be
 * called from a kernel function.
 */
class DruckerPragerUpdates : public ElasticIsotropicUpdates
{
public:
  /**
   * @brief Constructor
   * @param[in] bulkModulus The ArrayView holding the bulk modulus data for each element.
   * @param[in] shearModulus The ArrayView holding the shear modulus data for each element.
   * @param[in] stress The ArrayView holding the stress data for each quadrature point.
   */
  DruckerPragerUpdates( arrayView1d< real64 const > const & tanFrictionAngle,
                        arrayView1d< real64 const > const & tanDilationAngle,
                        arrayView1d< real64 const > const & hardeningRate,
                        arrayView2d< real64 > const & newCohesion,
                        arrayView2d< real64 > const & oldCohesion,
                        arrayView1d< real64 const > const & bulkModulus,
                        arrayView1d< real64 const > const & shearModulus,
                        arrayView3d< real64, solid::STRESS_USD > const & newStress,
                        arrayView3d< real64, solid::STRESS_USD > const & oldStress ):
    ElasticIsotropicUpdates( bulkModulus, shearModulus, newStress,oldStress ),
    m_tanFrictionAngle( tanFrictionAngle ),
    m_tanDilationAngle( tanDilationAngle ),
    m_hardeningRate( hardeningRate ),
    m_newCohesion( newCohesion ),
    m_oldCohesion( oldCohesion )
  {}

  /// Default copy constructor
  DruckerPragerUpdates( DruckerPragerUpdates const & ) = default;

  /// Default move constructor
  DruckerPragerUpdates( DruckerPragerUpdates && ) = default;

  /// Deleted default constructor
  DruckerPragerUpdates() = delete;

  /// Deleted copy assignment operator
  DruckerPragerUpdates & operator=( DruckerPragerUpdates const & ) = delete;

  /// Deleted move assignment operator
  DruckerPragerUpdates & operator=( DruckerPragerUpdates && ) =  delete;

  GEOSX_HOST_DEVICE
  virtual void smallStrainUpdate( localIndex const k,
                                  localIndex const q,
                                  real64 const ( & strainIncrement )[6],
                                  real64 ( & stress )[6],
                                  real64 ( & stiffness )[6][6] ) override final;
  
  GEOSX_HOST_DEVICE
  virtual void saveConvergedState() override final;
    
private:
  /// A reference to the ArrayView holding the friction angle for each element.
  arrayView1d< real64 const > const m_tanFrictionAngle;
  
  /// A reference to the ArrayView holding the dilation angle for each element.
  arrayView1d< real64 const > const m_tanDilationAngle;
  
  /// A reference to the ArrayView holding the hardening rate for each element.
  arrayView1d< real64 const > const m_hardeningRate;
  
  /// A reference to the ArrayView holding the new cohesion for each integration point
  arrayView2d< real64 > const m_newCohesion;
  
  /// A reference to the ArrayView holding the old cohesion for each integration point
  arrayView2d< real64 > const m_oldCohesion;
  
};

///////////////////////// new proposal /////////////////////////////////////////

GEOSX_HOST_DEVICE
void DruckerPragerUpdates::smallStrainUpdate( localIndex const k,
                                              localIndex const q,
                                              real64 const ( & /*strainIncrement*/ )[6],
                                              real64 ( & stress )[6],
                                              real64 ( & stiffness )[6][6] )
{
  real64 const shear = m_shearModulus[k];
  real64 const bulk  = m_bulkModulus[k];
  real64 const lame  = bulk - 2.0/3.0*shear;
  
  real64 const friction    = m_tanFrictionAngle[k];
  real64 const dilation    = m_tanDilationAngle[k];
  real64 const hardening   = m_hardeningRate[k];
  real64 const oldCohesion = m_oldCohesion[k][q];
  
  real64 cohesion = oldCohesion;
  
  // elastic predictor (assume strainIncrement is all elastic)
  
  /*
  real64 newElasticStrain[6];
  for(localIndex i=0; i<6; ++i)
  {
    newElasticStrain[i] = m_oldElasticStrain[k][q][i] + strainIncrement[i];
  }
  */
  // elastic strain invariants (scalar)
  
  real64 volStrain = 0;
  real64 devStrain = 0;
  real64 deviator[6] = {};
  
  /*
  twoInvariant::strainDecomposition(newElasticStrain,
                                    volStrain,
                                    devStrain,
                                    deviator);
  */
  // trial stress invariants
  
  real64 trialP = bulk * volStrain;
  real64 trialQ = 3 * shear * devStrain;
  
  // check yield function F <= 0
  
  real64 yield = trialQ + friction*trialP - cohesion;
  
  if(yield > 1e-9) // plasticity branch
  {
    // the return mapping can in general be written as a newton iteration.
    // here we have a linear problem, so the algorithm will converge in one
    // iteration, but this is a template for more general models with either
    // nonlinear hardening or curved yield surfaces.
    
    // for GPU we can simplify the DP model to avoid Newton, but Cam-Clay and
    // others will need it.
    
    // .... change to c-arrays ...
    array1d< real64 > solution(3), residual(3), delta(3);
    array2d< real64 > jacobian(3,3), jacobianInv(3,3);
    
    solution[0] = trialP; // initial guess for newP
    solution[1] = trialQ; // initial guess for newQ
    solution[2] = 0;      // initial guess for plastic multiplier
    
    real64 norm,normZero = 1e30;
    jacobian.setValues< serialPolicy >( 0 );
    
    for(localIndex iter=0; iter<10; ++iter) // could be fixed at one iter
    {
      // apply a linear cohesion decay model.  this requires an if() check for negative cohesion.
      // a log decay model would avoid this, at the price of nonlinearity
      
      cohesion = oldCohesion-solution[2]*hardening;
      real64 cohesionDeriv = -hardening;
      
      if(cohesion < 0)  // branch
      {
        cohesion = 0;
        cohesionDeriv = 0;
      }
      
      // assemble residual system
      
      residual[0] = solution[0] - trialP + solution[2]*bulk*dilation; // P - trialP + lambda*dG/dP = 0
      residual[1] = solution[1] - trialQ + solution[2]*3*shear;       // Q - trialQ + lambda*dG/dQ = 0
      residual[2] = solution[1] + friction*solution[0] - cohesion;    // F = 0
      
      // check for convergence (can be avoided for linear model)
      
      norm = LvArray::tensorOps::l2Norm<3>(residual);
      
      //residual.L2_Norm();  //std::cout << iter << " " << norm << std::endl;
      
      if(iter==0)
      {
        normZero = norm;
      }
      
      if(norm < 1e-8*(normZero+1)) 
      {
        break;
      }
      
      // solve Newton system
      
      jacobian(0,0) = 1;
      jacobian(0,2) = bulk*dilation;
      jacobian(1,1) = 1;
      jacobian(1,2) = 3*shear;
      jacobian(2,0) = friction;
      jacobian(2,1) = 1;
      jacobian(2,2) = cohesionDeriv;
      
      LvArray::tensorOps::invert<3>(jacobianInv,jacobian);
      LvArray::tensorOps::AijBj<3,3>(delta,jacobianInv,residual);
     
      for(localIndex i=0; i<3; ++i)
      {
        solution[i] -= delta[i];
      }
    }
    
    // construct stress = P*eye + sqrt(2/3)*Q*nhat
    
    twoInvariant::stressRecomposition(solution[0],
                                      solution[1],
                                      deviator,
                                      stress);
                                          
    // construct consistent tangent operator
    
    for(localIndex i=0; i<6; ++i)
    {
      for(localIndex j=0; j<6; ++j)
      {
        stiffness[i][j] = 0;
      }
    }
    
    real64 c1 = 2*shear*solution(1)/trialQ; // TODO: confirm trialQ != 0 for linear DP
    real64 c2 = jacobianInv(0,0)*bulk - c1/3;
    real64 c3 = sqrt(2./3)*3*shear*jacobianInv(0,1);
    real64 c4 = sqrt(2./3)*bulk*jacobianInv(1,0);
    real64 c5 = 2*jacobianInv(1,1)*shear - c1;
    
    array1d< real64 > identity(6);
    
    for(localIndex i=0; i<3; ++i)
    {
      stiffness[i][i] = c1;
      stiffness[i+3][i+3] = 0.5*c1;
      identity[i] = 1.0;
      identity[i+3] = 0.0;
    }
    
    for(localIndex i=0; i<6; ++i)
    {
      for(localIndex j=0; j<6; ++j)
      {
        stiffness[i][j] +=   c2 * identity[i] * identity[j]
                           + c3 * identity[i] * deviator[j]
                           + c4 * deviator[i] * identity[j]
                           + c5 * deviator[i] * deviator[j];
      }
    }
    
  }
  else // elasticity branch
  {
    twoInvariant::stressRecomposition(trialP,
                                      trialQ,
                                      deviator,
                                      stress);
                                      
    for(localIndex i=0; i<6; ++i)
    {
      for(localIndex j=0; j<6; ++j)
      {
        stiffness[i][j] = 0;
      }
    }
        
    stiffness[0][0] = lame + 2*shear;
    stiffness[0][1] = lame;
    stiffness[0][2] = lame;

    stiffness[1][0] = lame;
    stiffness[1][1] = lame + 2*shear;
    stiffness[1][2] = lame;

    stiffness[2][0] = lame;
    stiffness[2][1] = lame;
    stiffness[2][2] = lame + 2*shear;

    stiffness[3][3] = shear;
    stiffness[4][4] = shear;
    stiffness[5][5] = shear;
  }
  
  // remember history variables before returning
  
  m_newCohesion[k][q] = cohesion;
  
  saveStress( k, q, stress );
  
  return;
}


GEOSX_HOST_DEVICE
GEOSX_FORCE_INLINE
void DruckerPragerUpdates::saveConvergedState()
{
  SolidBaseUpdates::saveConvergedState();
  m_oldCohesion.setValues< serialPolicy >( m_newCohesion );
}


/**
 * @class DruckerPrager
 *
 * Drucker-Prager material model.
 */
class DruckerPrager : public ElasticIsotropic
{
public:

  /// @typedef Alias for DruckerPragerUpdates
  using KernelWrapper = DruckerPragerUpdates;

  /**
   * constructor
   * @param[in] name name of the instance in the catalog
   * @param[in] parent the group which contains this instance
   */
  DruckerPrager( string const & name, Group * const parent );

  /**
   * Default Destructor
   */
  virtual ~DruckerPrager() override;

  virtual void
  DeliverClone( string const & name,
                Group * const parent,
                std::unique_ptr< ConstitutiveBase > & clone ) const override;

  virtual void AllocateConstitutiveData( dataRepository::Group * const parent,
                                         localIndex const numConstitutivePointsPerParentIndex ) override;
  
  /**
   * @name Static Factory Catalog members and functions
   */
  ///@{

  /// string name to use for this class in the catalog
  static constexpr auto m_catalogNameString = "DruckerPrager";

  /**
   * @return A string that is used to register/lookup this class in the registry
   */
  static std::string CatalogName() { return m_catalogNameString; }

  virtual string GetCatalogName() override { return CatalogName(); }

  ///@}

  /**
   * @struct Set of "char const *" and keys for data specified in this class.
   */
  struct viewKeyStruct : public SolidBase::viewKeyStruct
  {
    /// string/key for default friction angle
    static constexpr auto defaultTanFrictionAngleString = "defaultTanFrictionAngle";
    
    /// string/key for default dilation angle
    static constexpr auto defaultTanDilationAngleString = "defaultTanDilationAngle";
    
    /// string/key for default hardening rate
    static constexpr auto defaultHardeningRateString = "defaultHardeningRate";
    
    /// string/key for default cohesion
    static constexpr auto defaultCohesionString = "defaultCohesion";
        
    /// string/key for friction angle
    static constexpr auto tanFrictionAngleString  = "TanFrictionAngle";
    
    /// string/key for dilation angle
    static constexpr auto tanDilationAngleString  = "TanDilationAngle";
    
    /// string/key for cohesion
    static constexpr auto hardeningRateString  = "HardeningRate";
    
    /// string/key for cohesion
    static constexpr auto newCohesionString  = "NewCohesion";
    
        /// string/key for cohesion
    static constexpr auto oldCohesionString  = "OldCohesion";
  };

  /**
   * @brief Create a instantiation of the DruckerPragerUpdate class that refers to the data in this.
   * @return An instantiation of DruckerPragerUpdate.
   */
  DruckerPragerUpdates createKernelWrapper() const
  {
    return DruckerPragerUpdates( m_tanFrictionAngle,
                                 m_tanDilationAngle,
                                 m_hardeningRate,
                                 m_newCohesion,
                                 m_oldCohesion,
                                 m_bulkModulus,
                                 m_shearModulus,
                                 m_newStress,
                                 m_oldStress );
  }

protected:
  virtual void PostProcessInput() override;

  //TODO: maybe define some helper structs for defaults, arrays, arrayViews, etc.
    
  /// Material parameter: The default value of yield surface slope
  real64 m_defaultTanFrictionAngle;
  
    /// Material parameter: The default value of plastic potential slope
  real64 m_defaultTanDilationAngle;
  
  /// Material parameter: The default value of the initial cohesion
  real64 m_defaultCohesion;
  
  /// Material parameter: The default value of the hardening rate
  real64 m_defaultHardeningRate;
    
  /// Material parameter: The yield surface slope for each element
  array1d< real64 > m_tanFrictionAngle;
  
  /// Material parameter: The plastic potential slope for each element
  array1d< real64 > m_tanDilationAngle;
  
  /// Material parameter: The hardening rate each element
  array1d< real64 > m_hardeningRate;
  
  /// History variable: The current cohesion value for each quadrature point
  array2d< real64 > m_newCohesion;
  
  /// History variable: The previous cohesion value for each quadrature point
  array2d< real64 > m_oldCohesion;
};

} /* namespace constitutive */

} /* namespace geosx */

#endif /* GEOSX_CONSTITUTIVE_SOLID_DRUCKERPRAGER_HPP_ */


