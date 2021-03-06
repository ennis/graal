#include <graal/buffer.hpp>
#include <graal/detail/resource.hpp>
#include <graal/detail/swapchain_impl.hpp>
#include <graal/image.hpp>

namespace graal::detail {

void resource::set_name(std::string name) {
    name_ = std::move(name);
}

image_resource* resource::as_image() {
    if (!(type_ == resource_type::image || type_ == resource_type::swapchain_image)) {
        return nullptr;
    }
    return static_cast<image_resource*>(this);
}

buffer_resource* resource::as_buffer() {
    if (type_ != resource_type::buffer) { return nullptr; }
    return static_cast<buffer_resource*>(this);
}

virtual_resource& resource::as_virtual_resource() {
    assert(type_ == resource_type::buffer || type_ == resource_type::image);

    if (type_ == resource_type::buffer) {
        return static_cast<virtual_resource&>(static_cast<buffer_impl&>(*this));
    } else {
        return static_cast<virtual_resource&>(static_cast<image_impl&>(*this));
    }
}

}  // namespace graal::detail