# Input:
# TENSORFLOW_ROOT 
# BUILD_CPP_IF
#
# Output:
# TensorFlow_FOUND        
# TensorFlow_INCLUDE_DIRS 
# TensorFlow_LIBRARY    
# TensorFlow_LIBRARY_PATH
# TensorFlowFramework_LIBRARY    
# TensorFlowFramework_LIBRARY_PATH


if (BUILD_CPP_IF AND INSTALL_TENSORFLOW)
  # Here we try to install libtensorflow_cc using conda install.

  if (USE_CUDA_TOOLKIT)
    set (VARIANT cuda)
  else ()
    set (VARIANT cpu)
  endif ()

  if (NOT DEFINED TENSORFLOW_ROOT)
    set (TENSORFLOW_ROOT ${CMAKE_INSTALL_PREFIX})
  endif ()
  # execute conda install
  execute_process(
	  COMMAND conda install libtensorflow_cc=*=${VARIANT}* -c deepmodeling -y -p ${TENSORFLOW_ROOT}
	  )
endif ()

if (BUILD_CPP_IF AND USE_TF_PYTHON_LIBS)
  # Here we try to install libtensorflow_cc.so as well as libtensorflow_framework.so using libs within the python site-package tensorflow folder.

  if (NOT DEFINED TENSORFLOW_ROOT)
    set (TENSORFLOW_ROOT ${CMAKE_INSTALL_PREFIX})
  endif ()
  # execute install script
  execute_process(
    COMMAND sh ${DEEPMD_SOURCE_DIR}/source/install/install_tf.sh ${Python_SITELIB} ${TENSORFLOW_ROOT}
    )
endif ()

if(DEFINED TENSORFLOW_ROOT)
  string(REPLACE "lib64" "lib" TENSORFLOW_ROOT_NO64 ${TENSORFLOW_ROOT})
endif(DEFINED TENSORFLOW_ROOT)

# define the search path
list(APPEND TensorFlow_search_PATHS ${TENSORFLOW_ROOT})
list(APPEND TensorFlow_search_PATHS "${TENSORFLOW_ROOT}/../tensorflow_core")
list(APPEND TensorFlow_search_PATHS ${TENSORFLOW_ROOT_NO64})
list(APPEND TensorFlow_search_PATHS "${TENSORFLOW_ROOT_NO64}/../tensorflow_core")
list(APPEND TensorFlow_search_PATHS "/usr/")
list(APPEND TensorFlow_search_PATHS "/usr/local/")

# includes
find_path(TensorFlow_INCLUDE_DIRS
  NAMES 
  tensorflow/core/public/session.h
  tensorflow/core/platform/env.h
  tensorflow/core/framework/op.h
  tensorflow/core/framework/op_kernel.h
  tensorflow/core/framework/shape_inference.h
  PATHS ${TensorFlow_search_PATHS} 
  PATH_SUFFIXES "/include"
  NO_DEFAULT_PATH
  )
if (BUILD_CPP_IF)
find_path(TensorFlow_INCLUDE_DIRS_GOOGLE
  NAMES 
  google/protobuf/type.pb.h
  PATHS ${TensorFlow_search_PATHS} 
  PATH_SUFFIXES "/include"
  NO_DEFAULT_PATH
  )
list(APPEND TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS_GOOGLE})
endif ()
  
if (NOT TensorFlow_INCLUDE_DIRS AND tensorflow_FIND_REQUIRED)
  message(FATAL_ERROR 
    "Not found 'tensorflow/core/public/session.h' directory in path '${TensorFlow_search_PATHS}' "
    "You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
endif ()

if (BUILD_CPP_IF)
  message (STATUS "Enabled cpp interface build, looking for tensorflow_cc and tensorflow_framework")
  # tensorflow_cc and tensorflow_framework
  if (NOT TensorFlow_FIND_COMPONENTS)
    set(TensorFlow_FIND_COMPONENTS tensorflow_cc tensorflow_framework)
  endif ()
  # the lib
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
  list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
  set (TensorFlow_LIBRARY_PATH "")
  foreach (module ${TensorFlow_FIND_COMPONENTS})
    find_library(TensorFlow_LIBRARY_${module}
      NAMES ${module}
      PATHS ${TensorFlow_search_PATHS} PATH_SUFFIXES lib NO_DEFAULT_PATH
      )
    if (TensorFlow_LIBRARY_${module})
      list(APPEND TensorFlow_LIBRARY ${TensorFlow_LIBRARY_${module}})
      get_filename_component(TensorFlow_LIBRARY_PATH_${module} ${TensorFlow_LIBRARY_${module}} PATH)
      list(APPEND TensorFlow_LIBRARY_PATH ${TensorFlow_LIBRARY_PATH_${module}})
    elseif (tensorflow_FIND_REQUIRED)
      message(FATAL_ERROR 
	"Not found lib/'${module}' in '${TensorFlow_search_PATHS}' "
	"You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
    endif ()
  endforeach ()
else (BUILD_CPP_IF)
  message (STATUS "Disabled cpp interface build, looking for tensorflow_framework")
endif (BUILD_CPP_IF)


# tensorflow_framework
if (NOT TensorFlowFramework_FIND_COMPONENTS)
  if (WIN32)
    set(TensorFlowFramework_FIND_COMPONENTS _pywrap_tensorflow_internal)
    set(TF_SUFFIX "")
  else ()
  set(TensorFlowFramework_FIND_COMPONENTS tensorflow_framework)
    set(TF_SUFFIX lib)
  endif ()
endif ()
# the lib
if (WIN32)
  list(APPEND TensorFlow_search_PATHS ${TENSORFLOW_ROOT}/python)
else ()
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.1)
list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES .so.2)
endif()
set (TensorFlowFramework_LIBRARY_PATH "")
foreach (module ${TensorFlowFramework_FIND_COMPONENTS})
  find_library(TensorFlowFramework_LIBRARY_${module}
    NAMES ${module}
    PATHS ${TensorFlow_search_PATHS} PATH_SUFFIXES ${TF_SUFFIX} NO_DEFAULT_PATH
    )
  if (TensorFlowFramework_LIBRARY_${module})
    list(APPEND TensorFlowFramework_LIBRARY ${TensorFlowFramework_LIBRARY_${module}})
    get_filename_component(TensorFlowFramework_LIBRARY_PATH_${module} ${TensorFlowFramework_LIBRARY_${module}} PATH)
    list(APPEND TensorFlowFramework_LIBRARY_PATH ${TensorFlowFramework_LIBRARY_PATH_${module}})
  elseif (tensorflow_FIND_REQUIRED)
    message(FATAL_ERROR 
      "Not found lib/'${module}' in '${TensorFlow_search_PATHS}' "
      "You can manually set the tensorflow install path by -DTENSORFLOW_ROOT ")
  endif ()
endforeach ()

if (BUILD_CPP_IF)
  # define the output variable
  if (TensorFlow_INCLUDE_DIRS AND TensorFlow_LIBRARY AND TensorFlowFramework_LIBRARY)
    set(TensorFlow_FOUND TRUE)
  else ()
    set(TensorFlow_FOUND FALSE)
  endif ()
else (BUILD_CPP_IF)
  if (TensorFlow_INCLUDE_DIRS AND TensorFlowFramework_LIBRARY)
    set(TensorFlow_FOUND TRUE)
  else ()
    set(TensorFlow_FOUND FALSE)
  endif ()
endif (BUILD_CPP_IF)

# detect TensorFlow version
try_run(
  TENSORFLOW_VERSION_RUN_RESULT_VAR TENSORFLOW_VERSION_COMPILE_RESULT_VAR
  ${CMAKE_CURRENT_BINARY_DIR}/tf_version
  "${CMAKE_CURRENT_LIST_DIR}/tf_version.cpp"
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${TensorFlow_INCLUDE_DIRS}"
  RUN_OUTPUT_VARIABLE TENSORFLOW_VERSION
  COMPILE_OUTPUT_VARIABLE TENSORFLOW_VERSION_COMPILE_OUTPUT_VAR
)
if (NOT ${TENSORFLOW_VERSION_COMPILE_RESULT_VAR})
  message(FATAL_ERROR "Failed to compile: \n ${TENSORFLOW_VERSION_COMPILE_OUTPUT_VAR}" )
endif()
if (NOT ${TENSORFLOW_VERSION_RUN_RESULT_VAR} EQUAL "0")
  message(FATAL_ERROR "Failed to run, return code: ${TENSORFLOW_VERSION}" )
endif()

# print message
if (NOT TensorFlow_FIND_QUIETLY)
  message(STATUS "Found TensorFlow: ${TensorFlow_INCLUDE_DIRS}, ${TensorFlow_LIBRARY}, ${TensorFlowFramework_LIBRARY} "
    " in ${TensorFlow_search_PATHS} (found version \"${TENSORFLOW_VERSION}\")")
endif ()

unset(TensorFlow_search_PATHS)
