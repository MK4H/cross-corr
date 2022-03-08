include (ExternalProject)

set_property (DIRECTORY PROPERTY EP_BASE Dependencies)

set (DEPENDENCIES)
set (EXTRA_CMAKE_ARGS)

# Use static linking to avoid issues with system-wide installations of Boost.
list (APPEND DEPENDENCIES ep_boost)
ExternalProject_Add (ep_boost
        URL https://sourceforge.net/projects/boost/files/boost/1.78.0/boost_1_78_0.tar.bz2/download
        URL_MD5 db0112a3a37a3742326471d20f1a186a
        CONFIGURE_COMMAND ./bootstrap.sh --with-libraries=program_options
        BUILD_COMMAND ./b2 link=static
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
        )
list (APPEND EXTRA_CMAKE_ARGS
        -DBOOST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Source/ep_boost
        -DBoost_NO_SYSTEM_PATHS=ON)

list (APPEND DEPENDENCIES ep_nlohmann)
ExternalProject_Add(ep_nlohmann
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.7.3
        CMAKE_ARGS -DJSON_BuildTests=OFF -DJSON_Install=ON -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        LOG_DOWNLOAD ON
        LOG_INSTALL ON
)
list (APPEND EXTRA_CMAKE_ARGS
        -Dnlohmann_json_ROOT=${CMAKE_CURRENT_BINARY_DIR}/Dependencies/Install/ep_nlohmann)

ExternalProject_Add (cross
        DEPENDS ${DEPENDENCIES}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}
        CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS}
        INSTALL_COMMAND ""
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
