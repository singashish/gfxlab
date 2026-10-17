function(gfx_compile_shaders target output_dir)
    set(shader_outputs)

    foreach(shader IN LISTS ARGN)
        get_filename_component(
            shader_abs
            "${shader}"
            ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        get_filename_component(shader_name "${shader}" NAME)
        set(output "${output_dir}/${shader_name}.spv")

        add_custom_command(
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
            COMMAND Vulkan::glslc
                    --target-env=vulkan1.3
                    -o "${output}"
                    "${shader_abs}"
            DEPENDS "${shader_abs}"
            VERBATIM
            COMMENT "Compiling shader ${shader_name}"
        )

        list(APPEND shader_outputs "${output}")
    endforeach()

    add_custom_target(${target} DEPENDS ${shader_outputs})
endfunction()
