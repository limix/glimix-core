add_library (liknorm combine.c expfam.c integral.c interval.c liknorm.c
normal.c special.c bracket.c brent.c zero.c shrink.c likelihood.c)
target_include_directories (liknorm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET liknorm PROPERTY C_STANDARD 11)
