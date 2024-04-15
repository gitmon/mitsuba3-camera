set(MI_PLUGIN_PREFIX "textures")

add_plugin(bitmap         bitmap.cpp)
add_plugin(checkerboard   checkerboard.cpp)
add_plugin(checkerlines   checkerlines.cpp)
add_plugin(polarlines     polarlines.cpp)
add_plugin(comb2d         comb2d.cpp)
add_plugin(star           star.cpp)
add_plugin(mesh_attribute mesh_attribute.cpp)
add_plugin(volume         volume.cpp)

set(MI_PLUGIN_TARGETS "${MI_PLUGIN_TARGETS}" PARENT_SCOPE)
