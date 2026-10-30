# Run by the fklb_build_info target at build time (cmake -P): the commits are read now, the rest comes from configure time.
# configure_file() does not touch the header when the content is the same, so nothing is rebuilt needlessly.
include("${FKLB_BIN}/generated/fklb_build_info_vars.cmake")

# "-dirty" means: a tracked file the binaries are built from has uncommitted changes. For this repository that is the sources
# (results/, logs and scripts change while experiments run and do not affect the binaries); for a submodule, any tracked file.
function(fklb_git_describe OUT DIR)
    if(NOT "${DIR}" STREQUAL "${FKLB_SRC}" AND NOT EXISTS "${DIR}/.git")
        # A submodule that is not checked out: git -C <dir> would walk up to this repository and report ITS commit. Report the state, and the
        # commit this repository pins (the gitlink), which needs no checkout.
        file(RELATIVE_PATH REL "${FKLB_SRC}" "${DIR}")
        execute_process(COMMAND git -C "${FKLB_SRC}" ls-tree HEAD "${REL}" OUTPUT_VARIABLE LINK OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REGEX MATCH "commit ([0-9a-f]+)" LINK_MATCH "${LINK}")
        if(CMAKE_MATCH_1)
            set(${OUT} "not checked out (pinned ${CMAKE_MATCH_1})" PARENT_SCOPE)
        else()
            set(${OUT} "not checked out" PARENT_SCOPE)
        endif()
        return()
    endif()
    set(PATHS "")
    if("${DIR}" STREQUAL "${FKLB_SRC}")
        set(PATHS -- common cmake fkl-self opencv-cuda npp external CMakeLists.txt CMakePresets.json)
    endif()
    execute_process(COMMAND git -C "${DIR}" rev-parse HEAD OUTPUT_VARIABLE H OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND git -C "${DIR}" status --porcelain --untracked-files=no ${PATHS} OUTPUT_VARIABLE D
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if("${H}" STREQUAL "")
        set(H "unknown")
    elseif(NOT "${D}" STREQUAL "")
        set(H "${H}-dirty")
    endif()
    set(${OUT} "${H}" PARENT_SCOPE)
endfunction()

fklb_git_describe(FKLB_COMMIT_REPO "${FKLB_SRC}")
fklb_git_describe(FKLB_COMMIT_FKL "${FKLB_SRC}/external/fkl")
fklb_git_describe(FKLB_COMMIT_CVGS "${FKLB_SRC}/external/cvGPUSpeedup")
fklb_git_describe(FKLB_COMMIT_FASTNPP "${FKLB_SRC}/external/FastNPP")

configure_file("${FKLB_SRC}/cmake/fklb_build_info.h.in" "${FKLB_BIN}/generated/fklb_build_info.h" @ONLY)
