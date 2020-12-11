set( fileIO_vtk_headers
     VTKPVDWriter.hpp
     VTKVTMWriter.hpp
     VTKGEOSXData.hpp
     VTKPolyDataWriterInterface.hpp )

set( fileIO_vtk_sources
     VTKGEOSXData.cpp
     VTKPVDWriter.cpp
     VTKVTMWriter.cpp
     VTKPolyDataWriterInterface.cpp )

set( dependencyList lapack dataRepository mesh meshUtilities managers vtk )

blt_add_library( NAME                  fileIO_vtk
                 SOURCES               ${fileIO_vtk_sources}
                 HEADERS               ${fileIO_vtk_headers}
                 DEPENDS_ON            ${dependencyList}
                 OBJECT                ${buildAsObj} )

add_subdirectory( unitTests )
