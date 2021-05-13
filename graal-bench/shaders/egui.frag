#version 450

layout(set=0,binding=1) uniform texture2D u_texture;
layout(set=0,binding=2) uniform sampler u_sampler;

layout(location=0) in vec4 v_rgba;
layout(location=1) in vec2 v_tc;
layout(location=0) out vec4 f_color;

void main() {
    // The texture sampler is sRGB aware, and glium already expects linear rgba output
    // so no need for any sRGB conversions here:
    f_color = v_rgba * texture(sampler2D(u_texture, u_sampler), v_tc);
}