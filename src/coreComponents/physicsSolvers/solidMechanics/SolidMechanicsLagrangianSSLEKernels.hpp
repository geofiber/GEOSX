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
 * @file SolidMechanicsLagrangianSSLEKernels.hpp
 */

#pragma once

#include "mesh/CellBlock.hpp"
#include "RAJA/RAJA.hpp"

#if USE_ELEM_PATCHES
  #define ELEM_PATCH_MAX_ELEM 64
  #define ELEM_PATCH_MAX_NODE 128

  #define SSLE_USE_PATCH_KERNEL 1
  #if SSLE_USE_PATCH_KERNEL
    #define SSLE_PATCH_KERNEL_SHARED_FVEC 0
  #endif
#endif

#if SSLE_USE_PATCH_KERNEL
using PATCH_NODAL_DATA = RAJA::LocalArray< geosx::real64, RAJA::Perm< 0, 1 >, RAJA::SizeList< ELEM_PATCH_MAX_NODE, 3>>;
#endif

#include "common/DataTypes.hpp"
#include "SolidMechanicsLagrangianFEMKernels.hpp"
#include "constitutive/ConstitutiveBase.hpp"
#include "finiteElement/ElementLibrary/FiniteElementBase.h"
#include "finiteElement/FiniteElementShapeFunctionKernel.hpp"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"

#include "finiteElement/Kinematics.h"
#include "common/TimingMacros.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

namespace geosx
{

namespace SolidMechanicsLagrangianSSLEKernels
{

struct StressCalculationKernel
{
  template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS, typename CONSTITUTIVE_TYPE >
  static inline real64
  Launch( CONSTITUTIVE_TYPE * const constitutiveRelation,
          localIndex const numElems,
          arrayView2d<localIndex const> const & elemsToNodes,
          arrayView3d< R1Tensor const> const & DNDX,
          arrayView2d<real64 const> const & GEOSX_UNUSED_ARG( detJ ),
          arrayView1d<R1Tensor const> const & u )
  {
    GEOSX_MARK_FUNCTION;

    typename CONSTITUTIVE_TYPE::KernelWrapper const & constitutive = constitutiveRelation->createKernelWrapper();

    arrayView2d< R2SymTensor > const & devStress = constitutiveRelation->deviatorStress();

    arrayView2d< real64 > const & meanStress = constitutiveRelation->meanStress();

    using KERNEL_POLICY = parallelDevicePolicy< 256 >;
    RAJA::forall< KERNEL_POLICY >( RAJA::TypedRangeSegment< localIndex >( 0, numElems ),
                                   GEOSX_DEVICE_LAMBDA ( localIndex const k )
    {
      real64 u_local[ NUM_NODES_PER_ELEM ][ 3 ];

      for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
      {
        for ( int b = 0; b < 3; ++b )
        {
          u_local[ a ][ b ] = u[ elemsToNodes[ k ][ a ] ][ b ];
        }
      }

      real64 c[ 6 ][ 6 ];
      constitutive.GetStiffness( k, c );

      //Compute Quadrature
      for ( localIndex q = 0; q < NUM_QUADRATURE_POINTS; ++q )
      {
        devStress[ k ][ q ] = 0.0;
        real64 * const restrict p_stress = devStress[ k ][ q ].Data();
        for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
        {
          real64 const v0_x_dNdXa0 = u_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 0 ];
          real64 const v1_x_dNdXa1 = u_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 1 ];
          real64 const v2_x_dNdXa2 = u_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 2 ];

          p_stress[ 0 ] += ( v0_x_dNdXa0 * c[ 0 ][ 0 ] + v1_x_dNdXa1 * c[ 0 ][ 1 ] + v2_x_dNdXa2*c[ 0 ][ 2 ] ) ;
          p_stress[ 2 ] += ( v0_x_dNdXa0 * c[ 1 ][ 0 ] + v1_x_dNdXa1 * c[ 1 ][ 1 ] + v2_x_dNdXa2*c[ 1 ][ 2 ] ) ;
          p_stress[ 5 ] += ( v0_x_dNdXa0 * c[ 2 ][ 0 ] + v1_x_dNdXa1 * c[ 2 ][ 1 ] + v2_x_dNdXa2*c[ 2 ][ 2 ] ) ;
          p_stress[ 4 ] += ( u_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 1 ] + u_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 2 ] ) * c[ 3 ][ 3 ] ;
          p_stress[ 3 ] += ( u_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 0 ] + u_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 2 ] ) * c[ 4 ][ 4 ] ;
          p_stress[ 1 ] += ( u_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 0 ] + u_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 1 ] ) * c[ 5 ][ 5 ] ;
        }

        real64 const dMeanStress = ( p_stress[ 0 ] + p_stress[ 2 ] + p_stress[ 5 ] ) / 3.0;
        meanStress[ k ][ q ] = dMeanStress;

        p_stress[ 0 ] -= dMeanStress;
        p_stress[ 2 ] -= dMeanStress;
        p_stress[ 5 ] -= dMeanStress;

      }//quadrature loop

    });

    return 0.0;
  }
};

/**
 * @struct Structure to wrap templated function that implements the explicit time integration kernels.
 */
struct ExplicitKernel
{
#if 0
  /**
   * @brief Launch of the element processing kernel for explicit time integration.
   * @tparam NUM_NODES_PER_ELEM The number of nodes/dof per element.
   * @tparam NUM_QUADRATURE_POINTS The number of quadrature points per element.
   * @tparam CONSTITUTIVE_TYPE the type of the constitutive relation that is being used.
   * @param A pointer to the constitutive relation that is being used.
   * @param elementList The list of elements to be processed
   * @param elemsToNodes The map from the elements to the nodes that form that element.
   * @param dNdX The derivatives of the shape functions wrt the reference configuration.
   * @param detJ The determinant of the transformation matrix (Jacobian) to the parent element.
   * @param u The nodal array of total displacements.
   * @param vel The nodal array of velocity.
   * @param acc The nodal array of force/acceleration.
   * @param meanStress The mean stress at each element quadrature point
   * @param devStress The deviator stress at each element quadrature point.
   * @param dt The timestep
   * @return The achieved timestep.
   */
  template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS, typename CONSTITUTIVE_TYPE >
  static inline real64
  Launch( CONSTITUTIVE_TYPE * const constitutiveRelation,
          LvArray::SortedArrayView<localIndex const, localIndex> const & elementList,
          arrayView2d<localIndex const> const & elemsToNodes,
          arrayView3d< R1Tensor const> const & DNDX,
          arrayView2d<real64 const> const & detJ,
          arrayView1d<R1Tensor const> const & GEOSX_UNUSED_ARG( u ),
          arrayView1d<R1Tensor const> const & vel,
          arrayView1d<R1Tensor> const & acc,
          arrayView2d<real64> const & meanStress,
          arrayView2d<R2SymTensor> const & devStress,
          real64 const dt )
  {
    GEOSX_MARK_FUNCTION;

    typename CONSTITUTIVE_TYPE::KernelWrapper const & constitutive = constitutiveRelation->createKernelWrapper();

    using KERNEL_POLICY = parallelDevicePolicy< 256 >;
    RAJA::forall< KERNEL_POLICY >( RAJA::TypedRangeSegment< localIndex >( 0, elementList.size() ),
                                   GEOSX_DEVICE_LAMBDA ( localIndex const i )
    {
      localIndex const k = elementList[ i ];

      real64 v_local[ NUM_NODES_PER_ELEM ][ 3 ];
      real64 f_local[ NUM_NODES_PER_ELEM ][ 3 ] = {};

      for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
      {
        for ( int b = 0; b < 3; ++b )
        {
          v_local[ a ][ b ] = vel[ elemsToNodes[ k ][ a ] ][ b ];
        }
      }

      real64 c[ 6 ][ 6 ];
      constitutive.GetStiffness( k, c );

      //Compute Quadrature
      for ( localIndex q = 0; q < NUM_QUADRATURE_POINTS; ++q )
      {
        real64 p_stress[ 6 ] = { 0 };
        for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
        {
          real64 const v0_x_dNdXa0 = v_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 0 ];
          real64 const v1_x_dNdXa1 = v_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 1 ];
          real64 const v2_x_dNdXa2 = v_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 2 ];

          p_stress[ 0 ] += ( v0_x_dNdXa0 * c[ 0 ][ 0 ] + v1_x_dNdXa1 * c[ 0 ][ 1 ] + v2_x_dNdXa2*c[ 0 ][ 2 ] ) * dt;
          p_stress[ 1 ] += ( v0_x_dNdXa0 * c[ 1 ][ 0 ] + v1_x_dNdXa1 * c[ 1 ][ 1 ] + v2_x_dNdXa2*c[ 1 ][ 2 ] ) * dt;
          p_stress[ 2 ] += ( v0_x_dNdXa0 * c[ 2 ][ 0 ] + v1_x_dNdXa1 * c[ 2 ][ 1 ] + v2_x_dNdXa2*c[ 2 ][ 2 ] ) * dt;
          p_stress[ 3 ] += ( v_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 1 ] + v_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 2 ] ) * c[ 3 ][ 3 ] * dt;
          p_stress[ 4 ] += ( v_local[ a ][ 2 ] * DNDX[ k ][ q ][ a ][ 0 ] + v_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 2 ] ) * c[ 4 ][ 4 ] * dt;
          p_stress[ 5 ] += ( v_local[ a ][ 1 ] * DNDX[ k ][ q ][ a ][ 0 ] + v_local[ a ][ 0 ] * DNDX[ k ][ q ][ a ][ 1 ] ) * c[ 5 ][ 5 ] * dt;
        }

        real64 const dMeanStress = ( p_stress[ 0 ] + p_stress[ 1 ] + p_stress[ 2 ] ) / 3.0;
        meanStress[ k ][ q ] += dMeanStress;

        p_stress[ 0 ] -= dMeanStress;
        p_stress[ 1 ] -= dMeanStress;
        p_stress[ 2 ] -= dMeanStress;

        real64 * const restrict p_devStress = devStress[ k ][ q ].Data();
        p_devStress[ 0 ] += p_stress[ 0 ];
        p_devStress[ 2 ] += p_stress[ 1 ];
        p_devStress[ 5 ] += p_stress[ 2 ];
        p_devStress[ 4 ] += p_stress[ 3 ];
        p_devStress[ 3 ] += p_stress[ 4 ];
        p_devStress[ 1 ] += p_stress[ 5 ];

        for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
        {
          f_local[ a ][ 0 ] -= ( p_devStress[ 1 ] * DNDX[ k ][ q ][ a ][ 1 ]
                          + p_devStress[ 3 ] * DNDX[ k ][ q ][ a ][ 2 ]
                          + DNDX[ k ][ q ][ a ][ 0 ] * ( p_devStress[ 0 ] + meanStress[ k ][ q ] ) ) * detJ[ k ][ q ];
          f_local[ a ][ 1 ] -= ( p_devStress[ 1 ] * DNDX[ k ][ q ][ a ][ 0 ]
                        + p_devStress[ 4 ] * DNDX[ k ][ q ][ a ][ 2 ]
                        + DNDX[ k ][ q ][ a ][ 1 ] * (p_devStress[ 2 ] + meanStress[ k ][ q ]) ) * detJ[ k ][ q ];
          f_local[ a ][ 2 ] -= ( p_devStress[ 3 ] * DNDX[ k ][ q ][ a ][ 0 ]
                        + p_devStress[ 4 ] * DNDX[ k ][ q ][ a ][ 1 ]
                        + DNDX[ k ][ q ][ a ][ 2 ] * (p_devStress[ 5 ] + meanStress[ k ][ q ]) ) * detJ[ k ][ q ];
        }
      }//quadrature loop

      for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
      {
        for ( int b = 0; b < 3; ++b )
        {
          RAJA::atomicAdd<RAJA::auto_atomic>( &acc[ elemsToNodes[ k ][ a ] ][ b ], f_local[ a ][ b ] );
        }
      }
    });

    return dt;
  }
#else

  template< localIndex NUM_NODES_PER_ELEM, localIndex NUM_QUADRATURE_POINTS, typename CONSTITUTIVE_TYPE >
  static inline real64
  Launch( CONSTITUTIVE_TYPE * const constitutiveRelation,
          LvArray::SortedArrayView<localIndex const, localIndex> const & elementList,
          arrayView2d<localIndex const> const & elemsToNodes,
#if USE_ELEM_PATCHES
#if SSLE_USE_PATCH_KERNEL
          arrayView1d<localIndex const> const & elemPatchOffsets,
          LvArray::ArrayOfArraysView<localIndex const, localIndex const, true> const & elemPatchNodes,
          arrayView2d<localIndex const> const & elemPatchElemsToNodes,
#else
          arrayView1d<localIndex const> const & GEOSX_UNUSED_ARG( elemPatchOffsets ),
          LvArray::ArrayOfArraysView<localIndex const, localIndex const, true> const & GEOSX_UNUSED_ARG( elemPatchNodes ),
          arrayView2d<localIndex const> const & GEOSX_UNUSED_ARG( elemPatchElemsToNodes ),

#endif
#endif
          arrayView3d< R1Tensor const> const &,
          arrayView2d<real64 const> const &,
          arrayView1d<R1Tensor const> const & X,
          arrayView1d<R1Tensor const> const & u,
          arrayView1d<R1Tensor> const & acc,
          arrayView2d<real64> const & ,
          arrayView2d<R2SymTensor> const & ,
          real64 const dt )
  {
    static bool printMessage = true;

    if( printMessage )
    {
#if SSLE_USE_PATCH_KERNEL
        GEOS_LOG_RANK_0( "Running patch kernel with " << elemPatchOffsets.size()-1 << " blocks" );
#else
        GEOS_LOG_RANK_0( "Running default kernel" );
#endif
        printMessage = false;
    }

    GEOSX_MARK_FUNCTION;


    typename CONSTITUTIVE_TYPE::KernelWrapper const & constitutive = constitutiveRelation->createKernelWrapper();

#if SSLE_USE_PATCH_KERNEL

    GEOSX_UNUSED_VAR( elemsToNodes );

    if( elementList.empty() )
    {
      return dt;
    }


#if SSLE_PATCH_KERNEL_SHARED_FVEC
    using InitMemParamList = RAJA::ParamList<0, 1, 2>;
    PATCH_NODAL_DATA ext_f_patch;
#else
    using InitMemParamList = RAJA::ParamList<0, 1>;
#endif
    PATCH_NODAL_DATA ext_X_patch;
    PATCH_NODAL_DATA ext_u_patch;


#if defined(__CUDACC__)

    using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<ELEM_PATCH_MAX_ELEM,
        RAJA::statement::For<0, RAJA::cuda_block_x_direct,
          RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, InitMemParamList,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop, RAJA::statement::Lambda<0>>,
            RAJA::statement::CudaSyncThreads,
            RAJA::statement::For<1, RAJA::cuda_thread_x_direct, RAJA::statement::Lambda<1>>
#if SSLE_PATCH_KERNEL_SHARED_FVEC
            ,
            RAJA::statement::CudaSyncThreads,
            RAJA::statement::For<2, RAJA::cuda_thread_x_loop, RAJA::statement::Lambda<2>>
#endif
            //RAJA::statement::CudaSyncThreads
          >
        >
      >
    >;

#else

    using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::loop_exec,
        RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, InitMemParamList,
          RAJA::statement::For<2, RAJA::loop_exec, RAJA::statement::Lambda<0>>,
          RAJA::statement::For<1, RAJA::loop_exec, RAJA::statement::Lambda<1>>
#if SSLE_PATCH_KERNEL_SHARED_FVEC
          ,
          RAJA::statement::For<2, RAJA::loop_exec, RAJA::statement::Lambda<2>>
#endif
        >
      >
    >;

#endif

    RAJA::kernel_param<KERNEL_POLICY>
      ( RAJA::make_tuple(
          RAJA::TypedRangeSegment<localIndex>( 0, elemPatchOffsets.size() - 1 ),
          RAJA::TypedRangeSegment<localIndex>( 0, ELEM_PATCH_MAX_ELEM ),
          RAJA::TypedRangeSegment<localIndex>( 0, ELEM_PATCH_MAX_NODE )
        ),
        RAJA::make_tuple(
#if SSLE_PATCH_KERNEL_SHARED_FVEC
          ext_f_patch,
#endif
          ext_X_patch,
          ext_u_patch
        ),

        /*
         * The first lambda loads nodal data from global memory into shared memory arrays
         */
        GEOSX_DEVICE_LAMBDA( localIndex const patch,
                             localIndex const GEOSX_UNUSED_ARG(),
                             localIndex const node,
#if SSLE_PATCH_KERNEL_SHARED_FVEC
                             PATCH_NODAL_DATA & f_patch,
#endif
                             PATCH_NODAL_DATA & X_patch,
                             PATCH_NODAL_DATA & u_patch )
        {
          if( node < elemPatchNodes.sizeOfArray( patch ) )
          {
            localIndex const nodeIndex = elemPatchNodes(patch, node);
#ifdef __CUDACC__
            //printf("Thread %d/%d: prefetching node %d\n", blockIdx.x, threadIdx.x, nodeIndex);
#else
            //printf( "Prefetching node %d\n", nodeIndex );
#endif
            #pragma unroll
            for( int b = 0 ; b < 3 ; ++b )
            {
#if SSLE_PATCH_KERNEL_SHARED_FVEC
              f_patch( node, b ) = 0.0;
#endif
              X_patch( node, b ) = X[ nodeIndex ][b];
              u_patch( node, b ) = u[ nodeIndex ][b];
            }
          }
        },

        /*
         * The second lambda performs update of local force vector
         */
        GEOSX_DEVICE_LAMBDA( localIndex const patch,
                             localIndex const elem,
                             localIndex const GEOSX_UNUSED_ARG(),
#if SSLE_PATCH_KERNEL_SHARED_FVEC
                             PATCH_NODAL_DATA & f_patch,
#endif
                             PATCH_NODAL_DATA & X_patch,
                             PATCH_NODAL_DATA & u_patch )
        {
          localIndex const patchSize = elemPatchOffsets[patch+1] - elemPatchOffsets[patch];

          // The check is needed since we are launching on non-uniformly sized patches
          if( elem >= patchSize )
          {
            return;
          }

          localIndex const k = elemPatchOffsets[patch] + elem;


#ifdef __CUDACC__
          //printf("Thread %d/%d: processing elem %d\n", blockIdx.x, threadIdx.x, k);
#else
          //printf( "Processing elem %d\n", k );
#endif

          real64 const G = constitutive.m_shearModulus[k];
          real64 const Lame = constitutive.m_bulkModulus[k] - 2.0/3.0 * G;
          real64 const Lame2G = 2*G + Lame;

#if SSLE_PATCH_KERNEL_SHARED_FVEC
#define FORCE_SUB(k, a, b, expr) RAJA::atomicSub<RAJA::auto_atomic>( &f_patch( elemPatchElemsToNodes(k, a), b ), (expr) )
#else
          real64 f_local[NUM_NODES_PER_ELEM][3]{};
#define FORCE_SUB(k, a, b, expr) f_local[ a ][ b ] -= (expr)
#endif

          //Compute Quadrature
          for( localIndex q = 0; q < NUM_QUADRATURE_POINTS; ++q )
          {
            real64 dNdX_data[3][8];
#define DNDX(k,q,a,i) dNdX_data[i][a]

            real64 const detJ_k_q =
              FiniteElementShapeKernel::shapeFunctionDerivatives( k,
                                                                  q,
                                                                  elemPatchElemsToNodes,
                                                                  X_patch,
                                                                  dNdX_data );

            #pragma unroll
            for( localIndex a=0 ; a< NUM_NODES_PER_ELEM ; ++a )
            {
              for( localIndex b=0 ; b< NUM_NODES_PER_ELEM ; ++b )
              {
                localIndex const nib = elemPatchElemsToNodes(k, b);


                real64 const dNdXa0_dNdXb0 = DNDX(k,q,a,0)*DNDX(k,q,b,0);
                real64 const dNdXa1_dNdXb1 = DNDX(k,q,a,1)*DNDX(k,q,b,1);
                real64 const dNdXa2_dNdXb2 = DNDX(k,q,a,2)*DNDX(k,q,b,2);

                FORCE_SUB(k, a, 0,
                                 ( u_patch(nib,1)*( DNDX(k,q,a,1)*DNDX(k,q,b,0)*G + DNDX(k,q,a,0)*DNDX(k,q,b,1)*Lame ) +
                                   u_patch(nib,2)*( DNDX(k,q,a,2)*DNDX(k,q,b,0)*G + DNDX(k,q,a,0)*DNDX(k,q,b,2)*Lame ) +
                                   u_patch(nib,0)*( dNdXa1_dNdXb1*G + dNdXa2_dNdXb2*G + dNdXa0_dNdXb0*(Lame2G))
                                 ) * detJ_k_q );

                FORCE_SUB(k, a, 1,
                                 ( u_patch(nib,0)*( DNDX(k,q,a,0)*DNDX(k,q,b,1)*G + DNDX(k,q,a,1)*DNDX(k,q,b,0)*Lame ) +
                                   u_patch(nib,2)*( DNDX(k,q,a,2)*DNDX(k,q,b,1)*G + DNDX(k,q,a,1)*DNDX(k,q,b,2)*Lame ) +
                                   u_patch(nib,1)*( dNdXa0_dNdXb0*G + dNdXa2_dNdXb2*G + dNdXa1_dNdXb1*(Lame2G))
                                 ) * detJ_k_q );

                FORCE_SUB(k, a, 2,
                                 ( u_patch(nib,0)*( DNDX(k,q,a,0)*DNDX(k,q,b,2)*G + DNDX(k,q,a,2)*DNDX(k,q,b,0)*Lame ) +
                                   u_patch(nib,1)*( DNDX(k,q,a,1)*DNDX(k,q,b,2)*G + DNDX(k,q,a,2)*DNDX(k,q,b,1)*Lame ) +
                                   u_patch(nib,2)*( dNdXa0_dNdXb0*G + dNdXa1_dNdXb1*G + dNdXa2_dNdXb2*(Lame2G))
                                 ) * detJ_k_q );
              }
            }
          }//quadrature loop

#if ! SSLE_PATCH_KERNEL_SHARED_FVEC
          #pragma unroll
          for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
          {
/*
#ifdef __CUDACC__
            if( k==5376 || k==5377 || k==5380 || k==5381 || k==5392 || k==5393 || k==5396 || k==5397 )
            {
              printf("Thread %d/%d: assembling k = %ld, a = %ld, node = %ld, patch_node = %ld\n", blockIdx.x, threadIdx.x, k, a, elemsToNodes(k,a), elemPatchElemsToNodes(k,a) );
            }
#endif
*/
            #pragma unroll
            for ( int b = 0; b < 3; ++b )
            {
              RAJA::atomicAdd<RAJA::auto_atomic>( &acc[ elemsToNodes( k, a ) ][ b ], f_local[ a ][ b ] );
              //RAJA::atomicAdd<RAJA::auto_atomic>( &acc[ elemsToNodes( k, a ) ][ b ], 1.0 );
            }
          }
#else
        },
        /*
         * The third lambda pushes nodal updates from shared memory arrays into global memory
         */
        GEOSX_DEVICE_LAMBDA( localIndex const patch,
                             localIndex const GEOSX_UNUSED_ARG(),
                             localIndex const node,
                             PATCH_NODAL_DATA & f_patch,
                             PATCH_NODAL_DATA & GEOSX_UNUSED_ARG(),
                             PATCH_NODAL_DATA & GEOSX_UNUSED_ARG() )
        {
          if( node < elemPatchNodes.sizeOfArray( patch ) )
          {
            for( int b = 0 ; b < 3 ; ++b )
            {
              RAJA::atomicAdd<RAJA::auto_atomic>( &acc[ elemPatchNodes( patch, node ) ][ b ], f_patch( node , b ) );
            }
          }
#endif
        }
    );



#else

#if defined(__CUDACC__)
    using KERNEL_POLICY = RAJA::cuda_exec< 256 >;
#elif defined(GEOSX_USE_OPENMP)
    using KERNEL_POLICY = RAJA::omp_parallel_for_exec;
#else
    using KERNEL_POLICY = RAJA::loop_exec;
#endif

//    RAJA::forall< KERNEL_POLICY >( RAJA::TypedRangeSegment< localIndex >( 0, elemsToNodes.size(0) ),
//                                   GEOSX_DEVICE_LAMBDA ( localIndex const k )
//    {
    RAJA::forall< KERNEL_POLICY >( RAJA::TypedRangeSegment< localIndex >( 0, elementList.size() ),
                                   GEOSX_DEVICE_LAMBDA ( localIndex const i )
    {
      localIndex const k = elementList[ i ];

      real64 f_local[ NUM_NODES_PER_ELEM ][ 3 ] = {};

      real64 const G = constitutive.m_shearModulus[k];
      real64 const Lame = constitutive.m_bulkModulus[k] - 2.0/3.0 * G;
      real64 const Lame2G = 2*G + Lame;
      //Compute Quadrature
      for ( localIndex q = 0; q < NUM_QUADRATURE_POINTS; ++q )
      {
        real64 dNdX_data[3][8];
#define DNDX(k,q,a,i) dNdX_data[i][a]

        real64 const detJ_k_q =
        FiniteElementShapeKernel::shapeFunctionDerivatives( k,
                                                            q,
                                                            elemsToNodes,
                                                            X,
                                                            dNdX_data );

//#define ULOCAL
#ifdef ULOCAL
        real64 uLocal[3][8];
        for( localIndex a=0 ; a< NUM_NODES_PER_ELEM ; ++a )
        {
          localIndex const nib = elemsToNodes(k, a);
          for( int i=0 ; i<3 ; ++i )
          {
            uLocal[i][a] = u[nib][i];
          }
        }

#define U(i,b) uLocal[i][b]
#else
#define U(i,b) u[nib][i]
#endif

        #pragma unroll
        for( localIndex a=0 ; a< NUM_NODES_PER_ELEM ; ++a )
        {
          for( localIndex b=0 ; b< NUM_NODES_PER_ELEM ; ++b )
          {
#if !defined(ULOCAL)
            localIndex const nib = elemsToNodes(k, b);
#endif
//            real64 const unib[3] = { u[nib][0], u[nib][1], u[nib][2] };
            real64 const dNdXa0_dNdXb0 = DNDX(k,q,a,0)*DNDX(k,q,b,0);
            real64 const dNdXa1_dNdXb1 = DNDX(k,q,a,1)*DNDX(k,q,b,1);
            real64 const dNdXa2_dNdXb2 = DNDX(k,q,a,2)*DNDX(k,q,b,2);

            f_local[a][0] -= ( U(1,b)*( DNDX(k,q,a,1)*DNDX(k,q,b,0)*G + DNDX(k,q,a,0)*DNDX(k,q,b,1)*Lame ) +
                               U(2,b)*( DNDX(k,q,a,2)*DNDX(k,q,b,0)*G + DNDX(k,q,a,0)*DNDX(k,q,b,2)*Lame ) +
                               U(0,b)*( dNdXa1_dNdXb1*G + dNdXa2_dNdXb2*G + dNdXa0_dNdXb0*(Lame2G))
                             ) * detJ_k_q;

            f_local[a][1] -= ( U(0,b)*( DNDX(k,q,a,0)*DNDX(k,q,b,1)*G + DNDX(k,q,a,1)*DNDX(k,q,b,0)*Lame ) +
                               U(2,b)*( DNDX(k,q,a,2)*DNDX(k,q,b,1)*G + DNDX(k,q,a,1)*DNDX(k,q,b,2)*Lame ) +
                               U(1,b)*( dNdXa0_dNdXb0*G + dNdXa2_dNdXb2*G + dNdXa1_dNdXb1*(Lame2G))
                             ) * detJ_k_q;

            f_local[a][2] -= ( U(0,b)*( DNDX(k,q,a,0)*DNDX(k,q,b,2)*G + DNDX(k,q,a,2)*DNDX(k,q,b,0)*Lame ) +
                               U(1,b)*( DNDX(k,q,a,1)*DNDX(k,q,b,2)*G + DNDX(k,q,a,2)*DNDX(k,q,b,1)*Lame ) +
                               U(2,b)*( dNdXa0_dNdXb0*G + dNdXa1_dNdXb1*G + dNdXa2_dNdXb2*(Lame2G))
                             ) * detJ_k_q;
          }
        }
      }//quadrature loop

      #pragma unroll
      for ( localIndex a = 0; a < NUM_NODES_PER_ELEM; ++a )
      {
        #pragma unroll
        for ( int b = 0; b < 3; ++b )
        {
          RAJA::atomicAdd<RAJA::auto_atomic>( &acc[ elemsToNodes(k, a) ][ b ], f_local[ a ][ b ] );
        }
      }
    });
#endif

    return dt;
  }
#endif
};

/**
 * @struct Structure to wrap templated function that implements the implicit time integration kernel.
 */
struct ImplicitKernel
{
  /**
   * @brief Launch of the element processing kernel for implicit time integration.
   * @tparam NUM_NODES_PER_ELEM The number of nodes/dof per element.
   * @tparam NUM_QUADRATURE_POINTS The number of quadrature points per element.
   * @tparam CONSTITUTIVE_TYPE the type of the constitutive relation that is being used.
   * @param constitutiveRelation A pointer to the constitutive relation that is being used.
   * @param numElems The number of elements the kernel will process.
   * @param dt The timestep.
   * @param dNdX The derivatives of the shape functions wrt the reference configuration.
   * @param detJ The determinant of the transformation matrix (Jacobian) to the parent element.
   * @param fe A pointer to the finite element class used in this kernel.
   * @param elemGhostRank An array containing the values of the owning ranks for ghost elements.
   * @param elemsToNodes The map from the elements to the nodes that form that element.
   * @param globalDofNumber The map from localIndex to the globalDOF number.
   * @param disp The array of total displacements.
   * @param uhat The array of incremental displacements (displacement for this step).
   * @param vtilde The array for the velocity predictor.
   * @param uhattilde The array for the incremental displacement predictor.
   * @param density The array containing the density
   * @param fluidPressure Array containing element fluid pressure at the beginning of the step.
   * @param deltaFluidPressure Array containing the change in element fluid pressure over this step.
   * @param biotCoefficient The biotCoefficient used to calculate effective stress.
   * @param tiOption The time integration option used for the integration.
   * @param stiffnessDamping The stiffness damping coefficient for the Newmark method assuming Rayleigh damping.
   * @param massDamping The mass damping coefficient for the Newmark method assuming Rayleigh damping.
   * @param newmarkBeta The value of \beta in the Newmark update.
   * @param newmarkGamma The value of \gamma in the Newmark update.
   * @param dofManager degree-of-freedom manager associated with the linear system
   * @param matrix sparse matrix containing the derivatives of the residual wrt displacement
   * @param rhs parallel vector containing the global residual
   * @return The maximum nodal force contribution from all elements.
   */
  template< int NUM_NODES_PER_ELEM, int NUM_QUADRATURE_POINTS, typename CONSTITUTIVE_TYPE >
  static inline real64
  Launch( CONSTITUTIVE_TYPE * const constitutiveRelation,
          localIndex const numElems,
          real64 const dt,
          arrayView3d<R1Tensor const> const & DNDX,
          arrayView2d<real64 const > const& detJ,
          FiniteElementBase const * const fe,
          arrayView1d< integer const > const & elemGhostRank,
          arrayView2d< localIndex const > const & elemsToNodes,
          arrayView1d< globalIndex const > const & globalDofNumber,
          arrayView1d< R1Tensor const > const & disp,
          arrayView1d< R1Tensor const > const & uhat,
          arrayView1d< R1Tensor const > const & vtilde,
          arrayView1d< R1Tensor const > const & uhattilde,
          arrayView1d< real64 const > const & density,
          arrayView1d< real64 const > const & fluidPressure,
          arrayView1d< real64 const > const & deltaFluidPressure,
          arrayView1d< real64 const > const & biotCoefficient,
          timeIntegrationOption const tiOption,
          real64 const stiffnessDamping,
          real64 const massDamping,
          real64 const newmarkBeta,
          real64 const newmarkGamma,
          DofManager const * const GEOSX_UNUSED_ARG( dofManager ),
          ParallelMatrix * const matrix,
          ParallelVector * const rhs )
  {
    constexpr int dim = 3;
    RAJA::ReduceMax< serialReduce, double > maxForce( 0 );

    typename CONSTITUTIVE_TYPE::KernelWrapper const & constitutive = constitutiveRelation->createKernelWrapper();

    RAJA::forall< serialPolicy >( RAJA::TypedRangeSegment< localIndex >( 0, numElems ),
                                  GEOSX_LAMBDA ( localIndex const k )
    {
      Epetra_LongLongSerialDenseVector elementLocalDofIndex( dim * NUM_NODES_PER_ELEM );
      Epetra_SerialDenseVector         R                   ( dim * NUM_NODES_PER_ELEM );
      Epetra_SerialDenseMatrix         dRdU                ( dim * NUM_NODES_PER_ELEM,
                                                             dim * NUM_NODES_PER_ELEM );
      Epetra_SerialDenseVector         element_dof_np1     ( dim * NUM_NODES_PER_ELEM );

      Epetra_SerialDenseVector R_InertiaMassDamping(R);
      Epetra_SerialDenseMatrix dRdU_InertiaMassDamping(dRdU);
      Epetra_SerialDenseVector R_StiffnessDamping(R);
      Epetra_SerialDenseMatrix dRdU_StiffnessDamping(dRdU);

      R1Tensor u_local[NUM_NODES_PER_ELEM];
      R1Tensor uhat_local[NUM_NODES_PER_ELEM];
      R1Tensor vtilde_local[NUM_NODES_PER_ELEM];
      R1Tensor uhattilde_local[NUM_NODES_PER_ELEM];

      dRdU.Scale(0);
      R.Scale(0);

      dRdU_InertiaMassDamping.Scale(0);
      R_InertiaMassDamping.Scale(0);
      dRdU_StiffnessDamping.Scale(0);
      R_StiffnessDamping.Scale(0);

      real64 c[6][6];
      constitutive.GetStiffness( k, c );

      if(elemGhostRank[k] < 0)
      {
        for( localIndex a=0 ; a<NUM_NODES_PER_ELEM ; ++a)
        {

          localIndex localNodeIndex = elemsToNodes[k][a];

          for( int i=0 ; i<dim ; ++i )
          {
            elementLocalDofIndex[static_cast<int>(a)*dim+i] = globalDofNumber[localNodeIndex]+i;

            // TODO must add last solution estimate for this to be valid
            element_dof_np1(static_cast<int>(a)*dim+i) = disp[localNodeIndex][i];
          }
        }

        if( tiOption == timeIntegrationOption::ImplicitDynamic )
        {
          GEOS_ERROR("Option not supported");
          CopyGlobalToLocal< NUM_NODES_PER_ELEM, R1Tensor>( elemsToNodes[k],
                                      disp, uhat, vtilde, uhattilde,
                                      u_local, uhat_local, vtilde_local, uhattilde_local );
        }
        else
        {
          CopyGlobalToLocal<NUM_NODES_PER_ELEM,R1Tensor>( elemsToNodes[k], disp, uhat, u_local, uhat_local );
        }

        R2SymTensor referenceStress;
        if( !fluidPressure.empty() )
        {
          referenceStress.PlusIdentity( - biotCoefficient[0] * (fluidPressure[k] + deltaFluidPressure[k]));
        }


        R1Tensor dNdXa;
        R1Tensor dNdXb;


        for( integer q=0 ; q<NUM_QUADRATURE_POINTS ; ++q )
        {
          const realT detJq = detJ[k][q];
          std::vector<double> const & N = fe->values(q);

          for( integer a=0 ; a<NUM_NODES_PER_ELEM ; ++a )
          {
      //      realT const * const dNdXa = dNdX(q,a).Data();
            dNdXa = DNDX[k][q][a];

            for( integer b=0 ; b<NUM_NODES_PER_ELEM ; ++b )
            {
      //        realT const * const dNdXb = dNdX(q,b).Data();
              dNdXb = DNDX[k][q][b];

              dRdU(a*dim+0,b*dim+0) -= ( c[0][0]*dNdXa[0]*dNdXb[0] + c[5][5]*dNdXa[1]*dNdXb[1] + c[4][4]*dNdXa[2]*dNdXb[2] ) * detJq;
              dRdU(a*dim+0,b*dim+1) -= ( c[5][5]*dNdXa[1]*dNdXb[0] + c[0][1]*dNdXa[0]*dNdXb[1] ) * detJq;
              dRdU(a*dim+0,b*dim+2) -= ( c[4][4]*dNdXa[2]*dNdXb[0] + c[0][2]*dNdXa[0]*dNdXb[2] ) * detJq;

              dRdU(a*dim+1,b*dim+0) -= ( c[0][1]*dNdXa[1]*dNdXb[0] + c[5][5]*dNdXa[0]*dNdXb[1] ) * detJq;
              dRdU(a*dim+1,b*dim+1) -= ( c[5][5]*dNdXa[0]*dNdXb[0] + c[1][1]*dNdXa[1]*dNdXb[1] + c[3][3]*dNdXa[2]*dNdXb[2] ) * detJq;
              dRdU(a*dim+1,b*dim+2) -= ( c[3][3]*dNdXa[2]*dNdXb[1] + c[1][2]*dNdXa[1]*dNdXb[2] ) * detJq;

              dRdU(a*dim+2,b*dim+0) -= ( c[0][2]*dNdXa[2]*dNdXb[0] + c[4][4]*dNdXa[0]*dNdXb[2] ) * detJq;
              dRdU(a*dim+2,b*dim+1) -= ( c[1][2]*dNdXa[2]*dNdXb[1] + c[3][3]*dNdXa[1]*dNdXb[2] ) * detJq;
              dRdU(a*dim+2,b*dim+2) -= ( c[4][4]*dNdXa[0]*dNdXb[0] + c[3][3]*dNdXa[1]*dNdXb[1] + c[2][2]*dNdXa[2]*dNdXb[2] ) * detJq;

              if( tiOption == timeIntegrationOption::ImplicitDynamic )
              {

                real64 integrationFactor = density[k] * N[a] * N[b] * detJq;
                real64 temp1 = ( massDamping * newmarkGamma/( newmarkBeta * dt ) + 1.0 / ( newmarkBeta * dt * dt ) )* integrationFactor;

                for( int i=0 ; i<dim ; ++i )
                {
                  realT const acc = 1.0 / ( newmarkBeta * dt * dt ) * ( uhat[b][i] - uhattilde[b][i] );
                  realT const vel = vtilde[b][i] + newmarkGamma/( newmarkBeta * dt ) *( uhat[b][i] - uhattilde[b][i] );

                  dRdU_InertiaMassDamping(a*dim+i,b*dim+i) -= temp1;
                  R_InertiaMassDamping(a*dim+i) -= ( massDamping * vel + acc ) * integrationFactor;
                }
              }
            }
          }
        }



          R1Tensor temp;
          for( integer q=0 ; q<NUM_QUADRATURE_POINTS ; ++q )
          {
            const realT detJq = detJ[k][q];
            R2SymTensor stress0 = referenceStress;
            stress0 *= detJq;
            for( integer a=0 ; a<NUM_NODES_PER_ELEM ; ++a )
            {
              dNdXa = DNDX[k][q][a];

              temp.AijBj(stress0,dNdXa);
              realT maxF = temp.MaxVal();
              maxForce.max( maxF );

              R(a*dim+0) -= temp[0];
              R(a*dim+1) -= temp[1];
              R(a*dim+2) -= temp[2];
            }
          }


      // TODO It is simpler to do this...try it.
      //  dRdU.Multiply(dof_np1,R);
        for( integer a=0 ; a<NUM_NODES_PER_ELEM ; ++a )
        {
          realT nodeForce = 0;
          for( integer b=0 ; b<NUM_NODES_PER_ELEM ; ++b )
          {
            for( int i=0 ; i<dim ; ++i )
            {
              for( int j=0 ; j<dim ; ++j )
              {
                R(a*dim+i) += dRdU(a*dim+i,b*dim+j) * u_local[b][j];
              }
            }

            if( tiOption == timeIntegrationOption::ImplicitDynamic )
            {
              for( int i=0 ; i<dim ; ++i )
              {
                for( int j=0 ; j<dim ; ++j )
                {
                  R_StiffnessDamping(a*dim+i) += stiffnessDamping * dRdU(a*dim+i,b*dim+j) * ( vtilde[b][j] + newmarkGamma/(newmarkBeta * dt)*(uhat[b][j]-uhattilde[b][j]) );
                }
              }
            }

          }

          nodeForce = std::max( std::max( R(a*dim+0), R(a*dim+1) ),  R(a*dim+2) );
          maxForce.max( fabs( nodeForce ) );
        }


        if( tiOption == timeIntegrationOption::ImplicitDynamic )
        {
          dRdU_StiffnessDamping = dRdU;
          dRdU_StiffnessDamping.Scale( stiffnessDamping * newmarkGamma / ( newmarkBeta * dt ) );

          dRdU += dRdU_InertiaMassDamping;
          dRdU += dRdU_StiffnessDamping;
          R    += R_InertiaMassDamping;
          R    += R_StiffnessDamping;
        }

        // TODO remove local epetra objects, remove use of unwrappedPointer()
        matrix->unwrappedPointer()->SumIntoGlobalValues( elementLocalDofIndex, dRdU);
        rhs->unwrappedPointer()->SumIntoGlobalValues( elementLocalDofIndex, R);
      }
    });

    return maxForce.get();
  }
};

} // namespace SolidMechanicsLagrangianSSLEKernels

} // namespace geosx
