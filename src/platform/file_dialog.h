#pragma once

#include <string>
#include <vector>

namespace gfx {

// Shows the system's "Open" dialog for model files (.obj, .gltf, .glb)
// with multiple selection and waits until it is closed. `owner` is the
// window's native handle (Window::native_handle()). Returns the chosen
// files as full UTF-8 paths in the order the dialog gives them, or an
// empty list when the user cancelled (or the dialog failed, which is
// logged). The program's current directory is not changed
std::vector<std::string> open_model_files_dialog(void* owner);

} // namespace gfx
