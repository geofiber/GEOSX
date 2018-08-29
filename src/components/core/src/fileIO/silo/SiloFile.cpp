/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * @file SiloFile.cpp
 */

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <sys/stat.h>

#include "managers/DomainPartition.hpp"
#include "mesh/MeshBody.hpp"


#pragma GCC diagnostic push

#ifdef __clang__
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif

#pragma GCC diagnostic ignored "-Wold-style-cast"

#include "SiloFile.hpp"

#include "common/Logger.hpp"

#include "codingUtilities/Utilities.hpp"

#include "constitutive/ConstitutiveManager.hpp"

#include "mesh/MeshBody.hpp"

/// forward declaration of NodeManagerT for use as a template argument
class NodeManager;

namespace geosx
{


/**
 *
 * @return
 */
namespace SiloFileUtilities
{

template<> int DB_TYPE<int> ()
{
  return DB_INT;
}
template<> int DB_TYPE<unsigned int> ()
{
  return DB_INT;
}
template<> int DB_TYPE<float> ()
{
  return DB_FLOAT;
}
template<> int DB_TYPE<real64> ()
{
  return DB_DOUBLE;
}
template<> int DB_TYPE<R1Tensor> ()
{
  return DB_DOUBLE;
}
template<> int DB_TYPE<R2Tensor> ()
{
  return DB_DOUBLE;
}
template<> int DB_TYPE<R2SymTensor> ()
{
  return DB_DOUBLE;
}
template<> int DB_TYPE<unsigned long> ()
{
  return DB_LONG;
}
template<> int DB_TYPE<long> ()
{
  return DB_LONG;
}
template<> int DB_TYPE<long long> ()
{
  return DB_LONG_LONG;
}
template<> int DB_TYPE<string> ()
{
  return DB_CHAR;
}

template<> int GetNumberOfVariablesInField<int> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<unsigned int> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<unsigned long> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<long> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<float> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<real64> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<long long unsigned int> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<long long int> ()
{
  return 1;
}
template<> int GetNumberOfVariablesInField<R1Tensor> ()
{
  return R1Tensor::Length();
}
template<> int GetNumberOfVariablesInField<R2Tensor> ()
{
  return R2Tensor::Length();
}
template<> int GetNumberOfVariablesInField<R2SymTensor> ()
{
  return R2SymTensor::Length();
}
template<> int GetNumberOfVariablesInField<string> ()
{
  return 1;
}

template<typename TYPE>
void SetVariableNames(string const & fieldName, string_array& varnamestring, char const* varnames[])
{
  varnamestring.resize(GetNumberOfVariablesInField<TYPE> ());
  varnamestring[0] = fieldName;
  varnames[0] = varnamestring[0].c_str();
}
template void SetVariableNames<int> ( string const & fieldName,
                                      string_array& varnamestring,
                                      char const* varnames[]);
template void SetVariableNames<unsigned long>( string const & fieldName,
                                               string_array& varnamestring,
                                               char const* varnames[]);
template void SetVariableNames<real64>( string const & fieldName,
                                        string_array& varnamestring,
                                        char const* varnames[]);
template void SetVariableNames<long long unsigned int>( string const & fieldName,
                                                        string_array& varnamestring,
                                                        char const* varnames[]);



template<>
void SetVariableNames<R1Tensor> ( string const & fieldName,
                                  string_array& varnamestring,
                                  char const* varnames[])
{
  varnamestring.resize(GetNumberOfVariablesInField<R1Tensor> ());
  varnamestring[0] = fieldName + "_1";
  varnamestring[1] = fieldName + "_2";
  varnamestring[2] = fieldName + "_3";
  varnames[0] = const_cast<char*>( varnamestring[0].c_str() );
  varnames[1] = const_cast<char*>( varnamestring[1].c_str() );
  varnames[2] = const_cast<char*>( varnamestring[2].c_str() );
}

template<>
void SetVariableNames<R2Tensor> ( string const & fieldName,
                                  string_array& varnamestring,
                                  char const* varnames[])
{
  varnamestring.resize(GetNumberOfVariablesInField<R2Tensor> ());
  varnamestring[0] = fieldName + "_11";
  varnamestring[1] = fieldName + "_12";
  varnamestring[2] = fieldName + "_13";
  varnamestring[3] = fieldName + "_21";
  varnamestring[4] = fieldName + "_22";
  varnamestring[5] = fieldName + "_23";
  varnamestring[6] = fieldName + "_31";
  varnamestring[7] = fieldName + "_32";
  varnamestring[8] = fieldName + "_33";
  varnames[0] = const_cast<char*>( varnamestring[0].c_str() );
  varnames[1] = const_cast<char*>( varnamestring[1].c_str() );
  varnames[2] = const_cast<char*>( varnamestring[2].c_str() );
  varnames[3] = const_cast<char*>( varnamestring[3].c_str() );
  varnames[4] = const_cast<char*>( varnamestring[4].c_str() );
  varnames[5] = const_cast<char*>( varnamestring[5].c_str() );
  varnames[6] = const_cast<char*>( varnamestring[6].c_str() );
  varnames[7] = const_cast<char*>( varnamestring[7].c_str() );
  varnames[8] = const_cast<char*>( varnamestring[8].c_str() );
}

template<>
void SetVariableNames<R2SymTensor> ( string const & fieldName,
                                     string_array& varnamestring,
                                     char const * varnames[])
{
  varnamestring.resize(GetNumberOfVariablesInField<R2Tensor> ());
  varnamestring[0] = fieldName + "_11";
  varnamestring[1] = fieldName + "_21";
  varnamestring[2] = fieldName + "_22";
  varnamestring[3] = fieldName + "_31";
  varnamestring[4] = fieldName + "_32";
  varnamestring[5] = fieldName + "_33";
  varnames[0] = const_cast<char*>( varnamestring[0].c_str() );
  varnames[1] = const_cast<char*>( varnamestring[1].c_str() );
  varnames[2] = const_cast<char*>( varnamestring[2].c_str() );
  varnames[3] = const_cast<char*>( varnamestring[3].c_str() );
  varnames[4] = const_cast<char*>( varnamestring[4].c_str() );
  varnames[5] = const_cast<char*>( varnamestring[5].c_str() );
}


template<> int GetTensorRank<int> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<unsigned int> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<long> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<unsigned long> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<real32> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<real64> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<R1Tensor> ()
{
  return DB_VARTYPE_VECTOR;
}
template<> int GetTensorRank<R2Tensor> ()
{
  return DB_VARTYPE_TENSOR;
}
template<> int GetTensorRank<R2SymTensor> ()
{
  return DB_VARTYPE_SYMTENSOR;
}
template<> int GetTensorRank<long long unsigned int> ()
{
  return DB_VARTYPE_SCALAR;
}
template<> int GetTensorRank<string> ()
{
  return DB_VARTYPE_SCALAR;
}



}



using namespace constitutive;
using namespace dataRepository;

// *********************************************************************************************************************
/// Default Constructor
SiloFile::SiloFile():
  m_dbFilePtr(nullptr),
  m_dbBaseFilePtr(nullptr),
  m_numGroups(1),
  m_baton(nullptr),
  m_driver(DB_HDF5),
  m_plotFileRoot("plot"),
  m_restartFileRoot("restart"),
  m_fileName(),
  m_baseFileName()
{}

// *********************************************************************************************************************
/// Destructor
SiloFile::~SiloFile()
{}

// *********************************************************************************************************************

void SiloFile::MakeSiloDirectories()
{

  int rank=0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif

  if( rank==0 )
  {
    struct stat sb;

    if( !( stat(m_siloDirectory.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode) ) )
    {
      mode_t nMode = 0733;
      mkdir(m_siloDirectory.c_str(),nMode);
    }

    if( !( stat( (m_siloDirectory +"/"+ m_siloDataSubDirectory).c_str(), &sb) == 0 && S_ISDIR(sb.st_mode) ) )
    {
      mode_t nMode = 0733;
      mkdir((m_siloDirectory +"/"+ m_siloDataSubDirectory).c_str(),nMode);
    }
  }
}

/**
 *
 */
void SiloFile::Initialize( const PMPIO_iomode_t readwrite, int const numGroups )
{
#if USE_MPI
  // Ensure all procs agree on numGroups, driver and file_ext
  m_numGroups = numGroups;

  MPI_Bcast(&m_numGroups, 1, MPI_INT, 0, MPI_COMM_GEOSX);
  MPI_Bcast( const_cast<int*>(&m_driver), 1, MPI_INT, 0, MPI_COMM_GEOSX);
  // Initialize PMPIO, pass a pointer to the driver type as the user data.
  m_baton = PMPIO_Init( m_numGroups,
                        readwrite,
                        MPI_COMM_GEOSX,
                        1,
                        PMPIO_DefaultCreate,
                        PMPIO_DefaultOpen,
                        PMPIO_DefaultClose,
                        const_cast<int*>(&m_driver));
#else
  m_numGroups = 1;
  // Initialize PMPIO, pass a pointer to the driver type as the user data.
  m_baton = PMPIO_Init(m_numGroups, PMPIO_WRITE, nullptr, 1, PMPIO_DefaultCreate, PMPIO_DefaultOpen,
                       PMPIO_DefaultClose, &m_driver);
#endif

}

// *********************************************************************************************************************
/**
 * @author settgast
 *
 */
void SiloFile::Finish()
{
  PMPIO_Finish(m_baton);
}

// *********************************************************************************************************************
/**
 *
 * @param domainNumber
 * @param cycleNum
 * @param fileName
 * @param nsName
 */
void SiloFile::WaitForBatonWrite( int const domainNumber,
                                  int const cycleNum,
                                  bool const isRestart )
{

  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  int const groupRank = PMPIO_GroupRank(m_baton, rank);
  char fileName[200] = { 0 };
  char baseFileName[200] = { 0 };
  char dirName[200] = { 0 };

  if( isRestart )
  {

    sprintf( baseFileName, "%s_%06d", m_restartFileRoot.c_str(), cycleNum);
    sprintf( fileName, "%s%s%s_%06d.%03d",
             m_siloDataSubDirectory.c_str(), "/", m_restartFileRoot.c_str(), cycleNum, groupRank);
  }
  else
  {
    sprintf(baseFileName, "%s_%06d",
            m_plotFileRoot.c_str(),
            cycleNum);

    sprintf(fileName,
            "%s_%06d.%03d",
            m_plotFileRoot.c_str(),
            cycleNum,
            groupRank);
  }
  sprintf(dirName, "domain_%05d", domainNumber);

  string dataFilePathAndName = m_siloDirectory + "/" + m_siloDataSubDirectory + "/" + fileName;
  m_dbFilePtr = static_cast<DBfile *>( PMPIO_WaitForBaton(m_baton, dataFilePathAndName.c_str(), dirName) );

  m_fileName = fileName;
  m_baseFileName = baseFileName;

  if( rank==0 )
  {
    m_dbBaseFilePtr = DBCreate( (m_siloDirectory + "/"+ m_baseFileName).c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5 );
//    m_dbBaseFilePtr = DBOpen( m_baseFileName.c_str(), DB_HDF5, DB_APPEND );
  }
}


void SiloFile::WaitForBaton( int const domainNumber, string const & restartFileName )
{

  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  int const groupRank = PMPIO_GroupRank(m_baton, rank);
  char fileName[200] = { 0 };
  char baseFileName[200] = { 0 };
  char dirName[200] = { 0 };


  sprintf(baseFileName, "%s", restartFileName.c_str());
  if( groupRank == 0 )
    sprintf(fileName, "%s", restartFileName.c_str());
  else
  {
    if( m_siloDirectory.empty())
    {
      sprintf(fileName, "%s.%03d", restartFileName.c_str(), groupRank);
    }
    else
    {
      sprintf(fileName, "%s%s%s.%03d", m_siloDirectory.c_str(), "/", restartFileName.c_str(), groupRank);
    }

  }

  sprintf(dirName, "domain_%05d", domainNumber);

  m_dbFilePtr = (DBfile *) PMPIO_WaitForBaton(m_baton, fileName, dirName);

  m_fileName = fileName;
  m_baseFileName = baseFileName;
}
/**
 *
 */
void SiloFile::HandOffBaton()
{
  PMPIO_HandOffBaton(m_baton, m_dbFilePtr);

  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  if( rank==0 )
  {
    DBClose(m_dbBaseFilePtr);
  }
}

/**
 *
 * @param meshName
 * @param nnodes
 * @param coords
 * @param globalNodeNum
 * @param numRegions
 * @param shapecnt
 * @param meshConnectivity
 * @param globalElementNum
 * @param ghostFlag
 * @param shapetype
 * @param shapesize
 * @param cycleNumber
 * @param problemTime
 */
void SiloFile::WriteMeshObject(string const & meshName,
                               const localIndex nnodes,
                               real64* coords[3],
                               globalIndex const * const globalNodeNum,
                               char const * const ghostNodeFlag,
                               char const * const ghostZoneFlag,
                               int const numShapes,
                               int const * shapecnt,
                               const localIndex* const * const meshConnectivity,
                               globalIndex const * const * const globalElementNum,
                               int const * const shapetype,
                               int const * const shapesize,
                               int const cycleNumber,
                               real64 const problemTime)
{

  const DBdatatype datatype = DB_DOUBLE;
  int const one = 1;

  DBoptlist* optlist = DBMakeOptlist(5);
  if( globalNodeNum!=nullptr )
  {
    if( std::is_same<globalIndex,int>::value || std::is_same<globalIndex,long long>::value )
    {
      DBAddOption(optlist, DBOPT_NODENUM, const_cast<globalIndex*>(globalNodeNum));
      if( std::is_same<globalIndex,long long>::value )
      {
        DBAddOption( optlist, DBOPT_LLONGNZNUM, const_cast<int*>(&one) );
      }
    }
  }
  DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
  DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));
  DBAddOption(optlist, DBOPT_GHOST_NODE_LABELS, const_cast<char*>(ghostNodeFlag) );


  int numTotZones = 0;
  int lnodelist = 0;
  for( int i = 0 ; i < numShapes ; ++i )
  {
    numTotZones += shapecnt[i];
    //  if shapesize <= 0, that signals that we are using arbitrary polygons.
    if( shapesize[i] > 0 )
      lnodelist += shapecnt[i] * shapesize[i];
    else
      lnodelist += -shapesize[i];
  }


  if( numTotZones == 0 )
  {
    char pwd[256];
    DBGetDir(m_dbFilePtr, pwd);
    string emptyObject = pwd;
    emptyObject += "/" + meshName;
    m_emptyMeshes.push_back(emptyObject);
  }
  else
  {

    string zonelistName;
    zonelistName = meshName + "_zonelist";

    DBPutUcdmesh(m_dbFilePtr, meshName.c_str(), 3, nullptr, (float**) coords, nnodes, numTotZones,
                 zonelistName.c_str(), nullptr, datatype, optlist);

    DBClearOptlist(optlist);

    array1d<integer> nodelist(lnodelist);
    globalIndex_array globalZoneNumber(lnodelist);

    int count = 0;
    int elemCount = 0;
    for( int i = 0 ; i < numShapes ; ++i )
    {
      int n;
      if( shapesize[i] > 0 )
        n = shapecnt[i] * shapesize[i];
      else
        n = -shapesize[i];
      for( int j = 0 ; j < n ; ++j )
      {
        nodelist[count++] = meshConnectivity[i][j];
      }
    }

    if( globalElementNum != nullptr )
    {
      for( int i = 0 ; i < numShapes ; ++i )
      {
        if( std::is_same<globalIndex,int>::value || std::is_same<globalIndex,long long>::value )
        {
          for( int j = 0 ; j < shapecnt[i] ; ++j )
          {
            globalZoneNumber[elemCount++] = globalElementNum[i][j];
          }
          // write zonelist
          DBAddOption(optlist, DBOPT_ZONENUM, const_cast<globalIndex*>(globalZoneNumber.data()));
          if( std::is_same<globalIndex,long long>::value )
          {
            DBAddOption(optlist, DBOPT_LLONGNZNUM, const_cast<int*> (&one));
          }
        }
      }
    }

    integer_array shapesize2(numShapes);
    for( int i = 0 ; i < numShapes ; ++i )
    {
      if( shapesize[i] < 0 )
      {
        shapesize2[i] = 0;
      }
      else
        shapesize2[i] = shapesize[i];
    }

    int hi_offset = 0;


    DBAddOption( optlist, DBOPT_GHOST_ZONE_LABELS, const_cast<char*>( ghostZoneFlag ) );

    DBPutZonelist2( m_dbFilePtr, zonelistName.c_str(), numTotZones, 3, nodelist.data(), lnodelist, 0, 0,
                    hi_offset, const_cast<int*>(shapetype), const_cast<int*>(shapesize2.data()),
                    const_cast<int*>(shapecnt), numShapes,
                    optlist);

    DBClearOptlist(optlist);
  }

  // write multimesh object
  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  if( rank == 0 )
  {
    DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
    DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));

    WriteMultiXXXX(DB_UCDMESH, DBPutMultimesh, 0, meshName, cycleNumber, "/", optlist);
  }

  DBFreeOptlist(optlist);
}


void SiloFile::WriteBeamMesh(string const & meshName,
                             const localIndex nnodes,
                             real64* coords[3],
                             const localIndex_array& node1,
                             const localIndex_array& node2,
                             int const cycleNumber,
                             real64 const problemTime)
{
  // Connectivity.
  integer_array nodelist;
  {
    nodelist.reserve(2*node1.size());
    localIndex_array::const_iterator it2 = node2.begin();
    for( localIndex_array::const_iterator it = node1.begin() ;
         it != node1.end() ; ++it, ++it2 )
    {
      nodelist.push_back(static_cast<int>(*it));
      nodelist.push_back(static_cast<int>(*it2));
    }
  }

  WriteBeamMesh( meshName, nnodes, coords, nodelist, cycleNumber, problemTime);
}

void SiloFile::WriteBeamMesh(string const & meshName,
                             const localIndex nnodes,
                             real64* coords[3],
                             integer_array& nodelist,
                             int const cycleNumber,
                             real64 const problemTime)
{
  const DBdatatype datatype = DB_DOUBLE;

  DBoptlist* optlist = DBMakeOptlist(4);
  DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
  DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));

  int lnodelist = nodelist.size();
  int nshapetypes = 1;
  int nzones = lnodelist/2;
  int shapecnt[] = {nzones};
  int shapesize[] = {2};
  int shapetype[] = {DB_ZONETYPE_BEAM};

  // Write out connectivity information.
  int const origin = 0;
  int const lo_offset = 0;
  int const hi_offset = 0;
  int const ndims = 3;
  const char* coordnames[3]  = { "xcoords", "ycoords", "zcoords" };

  // Write out connectivity information.
  DBPutZonelist2(m_dbFilePtr, "zonelist", nzones, ndims, lnodelist > 0 ? &nodelist[0] : nullptr,
                 lnodelist, origin, lo_offset, hi_offset,
                 shapetype, shapesize, shapecnt, nshapetypes, optlist);

  // Write an unstructured mesh.
  DBPutUcdmesh(m_dbFilePtr, meshName.c_str(), ndims, const_cast<char**>(coordnames), &coords[0], nnodes, nzones,
               "zonelist", nullptr, datatype, nullptr);
  DBClearOptlist(optlist);

  //----write multimesh object
  {
    int rank = 0;
  #if USE_MPI
    MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
  #endif
    if( rank == 0 )
    {
      DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
      DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));
      WriteMultiXXXX(DB_UCDMESH, DBPutMultimesh, 0, meshName.c_str(), cycleNumber, "/", optlist);
    }
  }

  //---free the option list
  DBFreeOptlist(optlist);
}



void SiloFile::WritePointMesh( string const & meshName,
                               const localIndex numPoints,
                               real64* coords[3],
                               int const cycleNumber,
                               real64 const problemTime)
{
  const DBdatatype datatype = DB_DOUBLE;
  DBoptlist* optlist = DBMakeOptlist(2);
  DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
  DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));
  DBPutPointmesh (m_dbFilePtr, meshName.c_str(), 3, coords, numPoints, datatype, optlist);


  //----write multimesh object
  {
    int rank = 0;
  #if USE_MPI
    MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
  #endif
    if( rank == 0 )
    {
      WriteMultiXXXX(DB_POINTMESH, DBPutMultimesh, 0, meshName.c_str(), cycleNumber, "/", optlist);
    }
  }

}


/**
 *
 * @param elementManager
 * @param cycleNumber
 * @param problemTime
 */
void SiloFile::WriteMaterialMapsCompactStorage( ElementRegionManager const * const elementManager,
                                          ConstitutiveManager const * const constitutiveManager,
                                          string const & meshName,
                                          int const cycleNumber,
                                          real64 const problemTime)
{

  auto const
  constitutiveMap = elementManager->
                    ConstructViewAccessor<
    std::pair< array2d<localIndex>,array2d<localIndex> >
    >( CellBlockSubRegion::viewKeyStruct::constitutiveMapString );

  string name = "Regions";
  int const nmat = constitutiveManager->GetSubGroups().size();
  array1d<int> matnos(nmat);
  std::vector<string> materialNameStrings(nmat);
  array1d<char const*> materialNames(nmat+1);
  materialNames.back() = nullptr;

  for( int matIndex=0 ; matIndex<nmat ; ++matIndex )
  {
    matnos[matIndex] = matIndex;
    materialNameStrings[matIndex] = constitutiveManager->GetGroup(matIndex)->getName();
    materialNames[matIndex] = materialNameStrings[matIndex].c_str();
  }

  int ndims = 1;
  int dims = elementManager->getNumberOfElements();

  array1d<integer> matlist(dims * nmat);


  int elemCount = 0;
  int regionCount = 0;
  for( localIndex er=0 ; er<elementManager->numRegions() ; ++er )
  {
    ElementRegion const * const elemRegion = elementManager->GetRegion(er);
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion const * const subRegion = elemRegion->GetSubRegion(esr);
      for( localIndex k = 0 ; k < subRegion->size() ; ++k )
      {
        // matIndex1 is the index of the material contained in the element
        localIndex const matIndex1 = constitutiveMap[er][esr].get().first[k][0];
        // matIndex2 is the index of the point within material specified in matIndex1
        localIndex const matIndex2 = constitutiveMap[er][esr].get().second[k][0];

        matlist[elemCount++] = matIndex1;
      }
    }
  }

  {
    DBoptlist* optlist = DBMakeOptlist(3);
    DBAddOption(optlist, DBOPT_MATNAMES, materialNames.data());
    DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
    DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));

    DBPutMaterial( m_dbFilePtr,
                   name.c_str(),
                   meshName.c_str(),
                   nmat,
                   matnos.data(),
                   matlist.data(),
                   &dims,
                   ndims,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   0,
                   DB_DOUBLE,
                   optlist);

    DBFreeOptlist(optlist);
  }
  // write multimesh object
  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  if( rank == 0 )
  {

    int size = 1;
#if USE_MPI
    MPI_Comm_size(MPI_COMM_GEOSX, &size);
#endif

    string_array vBlockNames(size);
    std::vector<char*> BlockNames(size);
    char tempBuffer[1024];
    char currentDirectory[256];

    DBGetDir(m_dbBaseFilePtr, currentDirectory);
    DBSetDir(m_dbBaseFilePtr, "/");

    for( int i = 0 ; i < size ; ++i )
    {
      int groupRank = PMPIO_GroupRank(m_baton, i);

      /* this mesh block is another file */
      sprintf( tempBuffer,
               "%s%s%s.%03d:/domain_%05d/%s",
               m_siloDataSubDirectory.c_str(),
               "/",
               m_baseFileName.c_str(),
               groupRank,
               i,
               name.c_str() );

      vBlockNames[i] = tempBuffer;
      BlockNames[i] = const_cast<char*>( vBlockNames[i].c_str() );
    }

    {
      DBoptlist* optlist = DBMakeOptlist(5);
      DBAddOption(optlist, DBOPT_MATNAMES, materialNames.data());
      DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
      DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));
      DBAddOption(optlist, DBOPT_NMATNOS, const_cast<int*>(&nmat) );
      DBAddOption(optlist, DBOPT_MATNOS, matnos.data() );

      DBPutMultimat(m_dbBaseFilePtr, name.c_str(), size, BlockNames.data(),
                    const_cast<DBoptlist*> (optlist));
      DBFreeOptlist(optlist);

    }

    DBSetDir(m_dbBaseFilePtr, currentDirectory);

  }


}




void SiloFile::WriteMaterialMapsFullStorage( ElementRegionManager const * const elementManager,
                                             ConstitutiveManager const * const constitutiveManager,
                                             string const & meshName,
                                             int const cycleNumber,
                                             real64 const problemTime)
{


  string name = "materials";
  int const nmat = constitutiveManager->GetSubGroups().size();
  array1d<int> matnos(nmat);
  std::vector<string> materialNameStrings(nmat);
  array1d<char const*> materialNames(nmat+1);
  materialNames.back() = nullptr;

  for( int matIndex=0 ; matIndex<nmat ; ++matIndex )
  {
    matnos[matIndex] = matIndex;

    materialNameStrings[matIndex] = constitutiveManager->GetGroup(matIndex)->getName();
    materialNames[matIndex] = materialNameStrings[matIndex].c_str();
  }


  int ndims = 1;
  int dims = 0;
  int mixlen=0;

  for( localIndex er=0 ; er<elementManager->numRegions() ; ++er )
  {
    ElementRegion const * const elemRegion = elementManager->GetRegion(er);
    int const numMatInRegion = elemRegion->getMaterialList().size();

    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion const * const subRegion = elemRegion->GetSubRegion(esr);
      if( numMatInRegion > 1 )
      {
        mixlen += subRegion->size() * numMatInRegion;
      }
      dims += subRegion->size();
    }
  }

  array1d<integer> matlist( dims );
  array1d<integer> mix_zone( mixlen );
  array1d<integer> mix_mat( mixlen );
  array1d<integer> mix_next( mixlen );
  array1d<real64> mix_vf( mixlen );

  int elemCount = 0;
  int mixCount = 0;
  for( localIndex er=0 ; er<elementManager->numRegions() ; ++er )
  {
    ElementRegion const * const elemRegion = elementManager->GetRegion(er);
    int const numMatInRegion = elemRegion->getMaterialList().size();

    array1d<localIndex> matIndices(numMatInRegion);

    for( localIndex a=0 ; a<numMatInRegion ; ++a )
    {
      matIndices[a] = constitutiveManager->
                      GetConstitituveRelation( elemRegion->getMaterialList()[a] )->
                      getIndexInParent();
    }

    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion const * const subRegion = elemRegion->GetSubRegion(esr);

      if( numMatInRegion == 1 )
      {
        for( localIndex k = 0 ; k < subRegion->size() ; ++k )
        {
          matlist[elemCount++] = matIndices[0];
        }
      }
      else if( numMatInRegion > 1 )
      {
        for( localIndex k = 0 ; k < subRegion->size() ; ++k )
        {
          matlist[elemCount++] = -(mixCount+1);
          for( localIndex a=0 ; a<numMatInRegion ; ++a )
          {
            mix_zone[mixCount] = k;
            mix_mat[mixCount] = matIndices[a];
            mix_vf[mixCount] = 1.0/numMatInRegion;
            if( a == numMatInRegion-1 )
            {
              mix_next[mixCount] = 0;
            }
            else
            {
              mix_next[mixCount] = mixCount+2;
            }
            ++mixCount;
          }
        }
      }
    }
  }

  {
    DBoptlist* optlist = DBMakeOptlist(3);
    DBAddOption(optlist, DBOPT_MATNAMES, materialNames.data());
    DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
    DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));

    DBPutMaterial( m_dbFilePtr,
                   name.c_str(),
                   meshName.c_str(),
                   nmat,
                   matnos.data(),
                   matlist.data(),
                   &dims,
                   ndims,
                   mix_next.data(),
                   mix_mat.data(),
                   mix_zone.data(),
                   mix_vf.data(),
                   mixlen,
                   DB_DOUBLE,
                   optlist);

    DBFreeOptlist(optlist);
  }
  // write multimesh object
  int rank = 0;
#if USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif
  if( rank == 0 )
  {

    int size = 1;
#if USE_MPI
    MPI_Comm_size(MPI_COMM_GEOSX, &size);
#endif

    array1d<string> vBlockNames(size);
    std::vector<char*> BlockNames(size);
    char tempBuffer[1024];
    char currentDirectory[256];

    DBGetDir(m_dbBaseFilePtr, currentDirectory);
    DBSetDir(m_dbBaseFilePtr, "/");

    for( int i = 0 ; i < size ; ++i )
    {
      int groupRank = PMPIO_GroupRank(m_baton, i);

      /* this mesh block is another file */
      sprintf( tempBuffer,
               "%s%s%s.%03d:/domain_%05d/%s",
               m_siloDataSubDirectory.c_str(),
               "/",
               m_baseFileName.c_str(),
               groupRank,
               i,
               name.c_str() );

      vBlockNames[i] = tempBuffer;
      BlockNames[i] = const_cast<char*>( vBlockNames[i].c_str() );
    }

    {
      DBoptlist* optlist = DBMakeOptlist(5);
      DBAddOption(optlist, DBOPT_MATNAMES, materialNames.data());
      DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNumber));
      DBAddOption(optlist, DBOPT_DTIME, const_cast<real64*> (&problemTime));
      DBAddOption(optlist, DBOPT_NMATNOS, const_cast<int*>(&nmat) );
      DBAddOption(optlist, DBOPT_MATNOS, matnos.data() );

      DBPutMultimat(m_dbBaseFilePtr, name.c_str(), size, BlockNames.data(),
                    const_cast<DBoptlist*> (optlist));
      DBFreeOptlist(optlist);

    }

    DBSetDir(m_dbBaseFilePtr, currentDirectory);

  }


  string subDirectory = "Materials";
  string rootDirectory = "/" + subDirectory;

  {
    string shortsubdir(subDirectory);
    string::size_type pos = subDirectory.find_last_of("//");

    if( pos != shortsubdir.npos )
    {
      shortsubdir.erase(0,pos+1);
    }


    MakeSubDirectory( shortsubdir, rootDirectory );
    DBSetDir(m_dbFilePtr, shortsubdir.c_str());
  }



  set<string> fieldNames;
  for( localIndex matI=0 ; matI<nmat ; ++matI )
  {
    ConstitutiveBase const * const
    constitutiveModel = constitutiveManager->GetConstitituveRelation(matI);

    for( auto const & wrapperIter : constitutiveModel->wrappers() )
    {
      auto const & wrapper = wrapperIter.second;

      if( wrapper->getPlotLevel() < dataRepository::PlotLevel::LEVEL_1 )
      {
        std::type_info const & typeID = wrapper->get_typeid();

        if( typeID==typeid( array2d<real64> ) )
        {
          fieldNames.insert( wrapper->getName() );
        }
      }
    }
  }

  for( auto fieldName : fieldNames )
  {
    ElementRegionManager::MaterialViewAccessor< array2d<real64> const >
    field = elementManager->ConstructMaterialViewAccessor< array2d<real64>  >( fieldName,
                                                                               constitutiveManager);

    WriteMaterialDataField< real64 >( meshName,
                                      fieldName,
                                      field,
                                      elementManager,
                                      constitutiveManager,
                                      DB_ZONECENT,
                                      cycleNumber,
                                      problemTime,
                                      rootDirectory,
                                      string_array() );

  }

  DBSetDir(m_dbFilePtr, "..");

}

void SiloFile::ClearEmptiesFromMultiObjects(int const cycleNum)
{

  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_GEOSX, &size);
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);

  string sendbufferVars;
  string sendbufferMesh;

  if( rank != 0 )
  {
    for( string_array::const_iterator emptyObject=m_emptyVariables.begin() ;
         emptyObject!=m_emptyVariables.end() ; ++emptyObject )
    {
      sendbufferVars += *emptyObject + ' ';
    }

    for( string_array::const_iterator emptyObject=m_emptyMeshes.begin() ;
         emptyObject!=m_emptyMeshes.end() ; ++emptyObject )
    {
      sendbufferMesh += *emptyObject + ' ';
    }

  }

  int sizeOfSendBufferVars = sendbufferVars.size();
  int sizeOfSendBufferMesh = sendbufferMesh.size();

  integer_array rcounts(size);
  integer_array displs(size);
  MPI_Gather( &sizeOfSendBufferVars, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, MPI_COMM_GEOSX);

  int sizeOfReceiveBuffer = 0;
  displs[0] = 0;
  for( int i=1 ; i<size ; ++i )
  {
    displs[i] = displs[i-1]+rcounts[i-1];
    sizeOfReceiveBuffer += rcounts[i];
  }
  string receiveBufferVars(sizeOfReceiveBuffer,'\0');

  MPI_Gatherv ( &sendbufferVars[0], sizeOfSendBufferVars, MPI_CHAR,
                &receiveBufferVars[0], rcounts.data(), displs.data(),
                MPI_CHAR, 0, MPI_COMM_GEOSX );


  MPI_Gather( &sizeOfSendBufferMesh, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, MPI_COMM_GEOSX);

  int sizeOfReceiveBufferMesh = 0;
  displs[0] = 0;
  for( int i=1 ; i<size ; ++i )
  {
    displs[i] = displs[i-1]+rcounts[i-1];
    sizeOfReceiveBufferMesh += rcounts[i];
  }
  string receiveBufferMesh(sizeOfReceiveBufferMesh,'\0');

  MPI_Gatherv ( &sendbufferMesh[0], sizeOfSendBufferMesh, MPI_CHAR,
                &receiveBufferMesh[0], rcounts.data(), displs.data(),
                MPI_CHAR, 0, MPI_COMM_GEOSX );



  if( rank== 0 )
  {
    std::istringstream iss(receiveBufferVars);
    copy(std::istream_iterator<string>(iss),
         std::istream_iterator<string>(),
         std::back_inserter< string_array >(m_emptyVariables));

    std::istringstream issm(receiveBufferMesh);
    copy(std::istream_iterator<string>(issm),
         std::istream_iterator<string>(),
         std::back_inserter< string_array >(m_emptyMeshes));
  }

  if( rank == 0 )
  {
    string baseFilePathAndName = m_siloDirectory + "/" + m_baseFileName;
    DBfile *siloFile = DBOpen(baseFilePathAndName.c_str(), DB_UNKNOWN, DB_APPEND);
    string empty("EMPTY");

    for( string_array::iterator emptyObject = m_emptyVariables.begin() ; emptyObject
         != m_emptyVariables.end() ; ++emptyObject )
    {
      size_t pathBegin = emptyObject->find_first_of('/', 1);
      size_t pathEnd = emptyObject->find_last_of('/');
      string domainString(*emptyObject, 1, pathBegin);
      string pathToMultiObj(*emptyObject, pathBegin, pathEnd - pathBegin);
      string varName(*emptyObject, pathEnd + 1);

      DBSetDir(siloFile, pathToMultiObj.c_str());

      DBmultivar* multiVar = DBGetMultivar(siloFile, varName.c_str());

      if( multiVar != nullptr )
      {
        array1d<const char*> newvarnames(multiVar->nvars);

        for( int i = 0 ; i < multiVar->nvars ; ++i )
        {

          string path(multiVar->varnames[i]);
          if( path.find(domainString) != string::npos )
          {
            newvarnames(i) = empty.c_str();
          }
          else
          {
            newvarnames(i) = multiVar->varnames[i];
          }
        }

        DBoptlist *optlist = DBMakeOptlist(5);
        DBAddOption(optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNum));
        //      DBAddOption( optlist, DBOPT_DTIME,
        // const_cast<real64*>(&problemTime) );
        DBAddOption(optlist, DBOPT_REGION_PNAMES, multiVar->region_pnames);
        DBAddOption(optlist, DBOPT_TENSOR_RANK, &multiVar->tensor_rank);
        DBAddOption(optlist, DBOPT_MMESH_NAME, multiVar->mmesh_name);

        DBPutMultivar(siloFile, varName.c_str(), multiVar->nvars,
                      const_cast<char**> (newvarnames.data()), multiVar->vartypes, optlist);
        DBFreeOptlist(optlist);
        DBFreeMultivar(multiVar);
      }
    }


    for( string_array::iterator emptyObject = m_emptyMeshes.begin() ; emptyObject
         != m_emptyMeshes.end() ; ++emptyObject )
    {
      size_t pathBegin = emptyObject->find_first_of('/', 1);
      size_t pathEnd = emptyObject->find_last_of('/');
      string domainString(*emptyObject, 1, pathBegin);
      string pathToMultiObj(*emptyObject, pathBegin, pathEnd - pathBegin);
      string varName(*emptyObject, pathEnd + 1);

      if( !(pathToMultiObj.compare("")) )
      {
        pathToMultiObj = "/";
      }

      DBSetDir(siloFile, pathToMultiObj.c_str());

      DBmultimesh* multiMesh = DBGetMultimesh(siloFile, varName.c_str());

      if( multiMesh != nullptr )
      {
        array1d<const char*> newmeshnames(multiMesh->nblocks);

        for( int i = 0 ; i < multiMesh->nblocks ; ++i )
        {

          string path(multiMesh->meshnames[i]);
          if( path.find(domainString) != string::npos )
          {
            newmeshnames(i) = empty.c_str();
          }
          else
          {
            newmeshnames(i) = multiMesh->meshnames[i];
          }
        }

        DBoptlist *optlist = DBMakeOptlist(2);
        DBAddOption( optlist, DBOPT_CYCLE, const_cast<int*> (&cycleNum));
//        DBAddOption( optlist, DBOPT_DTIME, const_cast<real64*>(&problemTime) );

        DBPutMultimesh( siloFile, varName.c_str(), multiMesh->nblocks,
                        const_cast<char**> (newmeshnames.data()), multiMesh->meshtypes, optlist);
        DBFreeOptlist(optlist);
        DBFreeMultimesh(multiMesh);
      }
    }
    DBClose(siloFile);
  }
  m_emptyVariables.clear();
  m_emptyMeshes.clear();

}





integer_array SiloFile::SiloNodeOrdering()
{

  integer_array nodeOrdering;

//  if( !m_elementGeometryID.compare(0, 4, "CPE2") )
//  {
//    nodeOrdering.resize(2);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//  }
//  else if( !m_elementGeometryID.compare(0, 4, "CPE3") )
//  {
//    nodeOrdering.resize(3);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//    nodeOrdering[2] = 2;
//    //    throw GPException("ElementRegionT::AllocateElementLibrary(): CPE3
// unimplemented");
//  }
//  else if (!m_elementGeometryID.compare(0, 4, "CPE4"))
//  {
//    nodeOrdering.resize(4);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//    nodeOrdering[2] = 3;
//    nodeOrdering[3] = 2;
//  }
//  else if (!m_elementGeometryID.compare(0, 4, "C3D4"))
//  {
//    nodeOrdering.resize(4);
//    nodeOrdering[0] = 1;
//    nodeOrdering[1] = 0;
//    nodeOrdering[2] = 2;
//    nodeOrdering[3] = 3;
//  }
//  else if (!m_elementGeometryID.compare(0, 4, "C3D8") ||
// !m_elementGeometryID.compare(0, 4, "C3D6"))
//  {
  nodeOrdering.resize(8);
  nodeOrdering[0] = 0;
  nodeOrdering[1] = 1;
  nodeOrdering[2] = 3;
  nodeOrdering[3] = 2;
  nodeOrdering[4] = 4;
  nodeOrdering[5] = 5;
  nodeOrdering[6] = 7;
  nodeOrdering[7] = 6;
//  }
//  else if (!m_elementGeometryID.compare(0, 4, "STRI"))
//  {
//    nodeOrdering.resize(3);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//    nodeOrdering[2] = 2;
//  }
//  else if (!m_elementGeometryID.compare(0, 3, "S4R"))
//  {
//    nodeOrdering.resize(4);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//    nodeOrdering[2] = 2;
//    nodeOrdering[3] = 3;
//  }
//  else if (!m_elementGeometryID.compare(0, 4, "TRSH"))
//  {
//    nodeOrdering.resize(4);
//    nodeOrdering[0] = 0;
//    nodeOrdering[1] = 1;
//    nodeOrdering[2] = 2;
//  }
  return nodeOrdering;
}



void SiloFile::WriteManagedGroupSilo( ManagedGroup const * group,
                                      string const & siloDirName,
                                      string const & meshname,
                                      int const centering,
                                      int const cycleNum,
                                      real64 const problemTime,
                                      bool const isRestart,
                                      const localIndex_array& mask )
{

  string subDirectory = siloDirName;
  string rootDirectory = "/" + siloDirName;

  {
    string shortsubdir(siloDirName);
    string::size_type pos = siloDirName.find_last_of("//");

    if( pos != shortsubdir.npos )
    {
      shortsubdir.erase(0,pos+1);
    }

    MakeSubDirectory( shortsubdir, rootDirectory );
    DBSetDir(m_dbFilePtr, shortsubdir.c_str());
  }

  WriteViewWrappersToSilo<real64>( meshname,
                                   group->wrappers(),
                                   centering,
                                   cycleNum,
                                   problemTime,
                                   isRestart,
                                   rootDirectory,
                                   mask);


  DBSetDir(m_dbFilePtr, "..");

}



void SiloFile::WriteDomainPartition( DomainPartition const & domain,
                                     int const cycleNum,
                                     real64 const problemTime,
                                     bool const isRestart )
{

  MeshLevel const * const mesh = domain.getMeshBody(0)->getMeshLevel(0);
  ConstitutiveManager const * const
  constitutiveManager = domain.GetGroup<ConstitutiveManager>(keys::ConstitutiveManager);

  WriteMeshLevel( mesh, constitutiveManager, cycleNum, problemTime, isRestart );

  if( isRestart )
  {
//    siloFile.DBWriteWrapper("m_globalDomainNumber",m_globalDomainNumber);
  }

}

void SiloFile::WriteMeshLevel( MeshLevel const * const meshLevel,
                               ConstitutiveManager const * const constitutiveManager,
                               int const cycleNum,
                               real64 const problemTime,
                               bool const isRestart )
{
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_GEOSX, &rank);
#endif

  //--------------WRITE FE DATA-----------------
//  if (m_feElementManager->m_numElems > 0)
  {

    NodeManager const * const nodeManager = meshLevel->getNodeManager();
    localIndex const numNodes = nodeManager->size();


    string const ghostNodeName = "ghostNodeFlag";
    string const ghostZoneName = "ghostZoneFlag";

    integer_array const & nodeGhostRank = nodeManager->GhostRank();
    array1d<char> ghostNodeFlag( nodeGhostRank.size() );
    array1d<char> ghostZoneFlag;





    r1_array const & referencePosition = nodeManager->getReference<r1_array>(keys::referencePositionString);

    bool writeArbitraryPolygon(false);
    string const meshName("MeshLevel");

    //set the nodal coordinate data structure
    real64* coords[3];
    array1d<real64> xcoords(numNodes);
    array1d<real64> ycoords(numNodes);
    array1d<real64> zcoords(numNodes);
    for( localIndex a = 0 ; a < numNodes ; ++a )
    {
      R1Tensor nodePosition;
      nodePosition = referencePosition[a];

      xcoords[a] = nodePosition(0);
      ycoords[a] = nodePosition(1);
      zcoords[a] = nodePosition(2);

      if( nodeGhostRank[a] >=0 )
      {
        ghostNodeFlag[a] = 1;
      }
    }

    coords[0] = xcoords.data();
    coords[1] = ycoords.data();
    coords[2] = zcoords.data();

    ElementRegionManager const * const elementManager = meshLevel->getElemManager();
    const localIndex numElementRegions = elementManager->GetGroup(keys::elementRegions)->GetSubGroups().size();
    array1d<localIndex*> meshConnectivity(numElementRegions);
    array1d<globalIndex*> globalElementNumbers(numElementRegions);
    array1d<integer> shapecnt(numElementRegions);
    array1d<integer> shapetype(numElementRegions);
    array1d<integer> shapesize(numElementRegions);

    array1d<FixedOneToManyRelation> elementToNodeMap;
    elementToNodeMap.resize( numElementRegions );

    int count = 0;

    ManagedGroup const * elementRegions = elementManager->GetGroup(dataRepository::keys::elementRegions);

    for( localIndex er=0 ; er<elementManager->numCellBlocks() ; ++er )
    {
      ElementRegion const * const region = elementManager->GetRegion(er);

      for( localIndex esr=0 ; esr<region->numSubRegions() ; ++esr )
      {
        CellBlockSubRegion const * cellBlock = region->GetSubRegion(esr);

        array2d<localIndex> const & elemsToNodes = cellBlock->getWrapper<FixedOneToManyRelation>(cellBlock->viewKeys().nodeList)->reference();// getData<array2d<localIndex>>(keys::nodeList);

        // The following line seems to be redundant. It's actual function is to
        // size this temp array.(pfu)
        elementToNodeMap[count].resize(elemsToNodes.size(0),elemsToNodes.size(1));

        integer_array const & elemGhostRank = cellBlock->GhostRank();



        for( localIndex k = 0 ; k < cellBlock->size() ; ++k )
        {
          arrayView1d<localIndex const> const elemToNodeMap = elemsToNodes[k];

          const integer_array nodeOrdering = SiloNodeOrdering();
          integer numNodesPerElement = integer_conversion<int>(elemsToNodes.size(1));
          for( localIndex a = 0 ; a < numNodesPerElement ; ++a )
          {
            elementToNodeMap[count](k, a) = elemToNodeMap[nodeOrdering[a]];
          }

          if( elemGhostRank[k] >= 0 )
          {
            ghostZoneFlag.push_back( 1 );
          }
        }


        meshConnectivity[count] = elementToNodeMap[count].data();


//        globalElementNumbers[count] = elementRegion.m_localToGlobalMap.data();
        shapecnt[count] = static_cast<int>(cellBlock->size());

//      if ( !elementRegion.m_elementGeometryID.compare(0, 4, "C3D8") )
//      {
        shapetype[count] = DB_ZONETYPE_HEX;
//      }
//      else if ( !elementRegion.m_elementGeometryID.compare(0, 4, "C3D6") )
//      {
//        shapetype[count] = DB_ZONETYPE_HEX;
//        writeArbitraryPolygon = true;
//      }
//      else if ( !elementRegion.m_elementGeometryID.compare(0, 4, "C3D4") )
//      {
//        shapetype[count] = DB_ZONETYPE_TET;
//      }
//      else if ( !elementRegion.m_elementGeometryID.compare(0, 4, "CPE4") ||
// !elementRegion.m_elementGeometryID.compare(0, 3, "S4R") )
//      {
//        shapetype[count] = DB_ZONETYPE_QUAD;
//      }
//      else if ( !elementRegion.m_elementGeometryID.compare(0, 4, "STRI") ||
// !elementRegion.m_elementGeometryID.compare(0, 4, "TRSH") ||
// !elementRegion.m_elementGeometryID.compare(0, 4, "CPE3"))
//      {
//        shapetype[count] = DB_ZONETYPE_TRIANGLE;
//      }
//      else if ( !elementRegion.m_elementGeometryID.compare(0, 4, "CPE2") )
//      {
//        shapetype[count] = DB_ZONETYPE_TRIANGLE;
//      }
//      else
//      {
//        GEOS_ERROR("PhysicalDomainT::WriteFiniteElementMesh: Do not recognize
// geometry type " + elementRegion.m_elementGeometryID + " \n");
//      }

        shapesize[count] = integer_conversion<int>(elemsToNodes.size(1));
        count++;
      }
    }



    WriteMeshObject(meshName,
                    numNodes,
                    coords,
                    nodeManager->m_localToGlobalMap.data(),
                    ghostNodeFlag.data(),
                    ghostZoneFlag.data(),
                    integer_conversion<int>(numElementRegions),
                    shapecnt.data(),
                    meshConnectivity.data(),
                    nullptr /*globalElementNumbers.data()*/,
                    shapetype.data(),
                    shapesize.data(),
                    cycleNum,
                    problemTime);


    // write node fields in silo mesh, and all restart data as unassociated
    // variables.
    WriteManagedGroupSilo( nodeManager,
                           "NodalFields",
                           meshName,
                           DB_NODECENT,
                           cycleNum,
                           problemTime,
                           isRestart,
                           localIndex_array());



//    DBSetDir(m_dbFilePtr, "NodalFields" );
//    WriteDataField<int>( meshName,
//                         ghostNodeName,
//                         ghostNodeFlag,
//                         DB_NODECENT,
//                         cycleNum,
//                         problemTime,
//                         "/NodalFields" );
//    DBSetDir(m_dbFilePtr, "..");



    {
//      array1d<array1d<localIndex> > materialOrder;
//      array1d<localIndex> materialOrderCounter;
//      materialOrder.resize( constitutiveManager->GetSubGroups().size() );
//      materialOrderCounter.resize( constitutiveManager->GetSubGroups().size() );
//      for( localIndex matIndex=0 ; matIndex<constitutiveManager->GetSubGroups().size() ; ++matIndex )
//      {
//        ConstitutiveBase const * const
//        constitutiveModel = constitutiveManager->GetGroup<ConstitutiveBase>(matIndex);
//
//        materialOrder[matIndex].resize( constitutiveModel->size() );
//        materialOrder[matIndex] = -1;
//        materialOrderCounter[matIndex] = 0;
//      }
      ElementRegionManager const * const elemManager = meshLevel->getElemManager();


      this->WriteMaterialMapsFullStorage( elemManager,
                                          constitutiveManager,
                                          meshName,
                                          cycleNum,
                                          problemTime );

      for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
      {
        ElementRegion const * const elemRegion = elemManager->GetRegion(er);
        for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
        {
          CellBlockSubRegion const * const subRegion = elemRegion->GetSubRegion(esr);

          string regionName = elemRegion->getName() + "_" + subRegion->getName();

          WriteManagedGroupSilo( subRegion,
                                 regionName,
                                 meshName,
                                 DB_ZONECENT,
                                 cycleNum,
                                 problemTime,
                                 isRestart,
                                 localIndex_array());

//          DBSetDir(m_dbFilePtr, regionName.c_str() );
//          string temp = "/" + regionName;
//          WriteDataField<int>( meshName,
//                               ghostZoneName,
//                               ghostZoneFlag,
//                               DB_ZONECENT,
//                               cycleNum,
//                               problemTime,
//                               temp.c_str() );
//          DBSetDir(m_dbFilePtr, "..");


        }
      }
    }
//    m_feElementManager->WriteSilo( siloFile, meshName, cycleNum, problemTime,
// isRestart );


  }//end FE write
}

}
#pragma GCC diagnostic pop
