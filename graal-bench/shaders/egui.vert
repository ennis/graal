#version 450

layout(set=0,binding=0,std140) uniform Globals {
    vec2 u_screen_size;
};

layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_tc;
layout(location=2) in uvec4 a_srgba; // 0-255 sRGB

layout(location=0) out vec4 v_rgba;
layout(location=1) out vec2 v_tc;

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb), srgba.a / 255.0);
}

void main() {
    gl_Position = vec4(
    2.0 * a_pos.x / u_screen_size.x - 1.0,
    2.0 * a_pos.y / u_screen_size.y - 1.0,
    0.0,
    1.0);
    // egui encodes vertex colors in gamma spaces, so we must decode the colors here:
    v_rgba = linear_from_srgba(a_srgba);
    v_tc = a_tc;
}