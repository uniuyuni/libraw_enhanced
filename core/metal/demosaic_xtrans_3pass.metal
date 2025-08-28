//
// demosaic_xtrans_3pass.metal
// X-Trans 3-pass demosaic - CPU identical implementation
//

#include "shader_types.h"
#include "shader_common.h"

using namespace metal;

// Tile processing constants
constant int TS = 114;  // Match CPU tile size

// RGB to LAB conversion (CPU identical)
float3 rgb_to_lab(float3 rgb, constant float* cbrt_lut, constant float* xyz_cam_flat) {
    float3 xyz = float3(0.0f);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            xyz[i] += xyz_cam_flat[i * 3 + j] * rgb[j];
        }
    }
    
    // Clamp to valid range for LUT access
    int x_idx = clamp(int(xyz.x), 0, 0x13fff);
    int y_idx = clamp(int(xyz.y), 0, 0x13fff);
    int z_idx = clamp(int(xyz.z), 0, 0x13fff);
    
    // Correctly apply gamma LUT
    xyz.x = cbrt_lut[x_idx];
    xyz.y = cbrt_lut[y_idx];
    xyz.z = cbrt_lut[z_idx];
    
    float L = 116.0f * xyz.y - 16.0f;
    float a = 500.0f * (xyz.x - xyz.y);
    float b = 200.0f * (xyz.y - xyz.z);
    
    return float3(L, a, b);
}

// Pass 1: Green interpolation
kernel void demosaic_xtrans_3pass_pass1(
    const device ushort* raw_data [[buffer(0)]],
    device float* cfa_data [[buffer(1)]],
    device float* rgb_green [[buffer(2)]],
    constant XTransParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    
    if (gid.x >= width || gid.y >= height) return;
    
    uint idx = gid.y * width + gid.x;
    uint color = fcol_xtrans(gid.y, gid.x, params.xtrans);
    
    float raw_val = float(raw_data[idx]) / params.maximum_value;
    
    cfa_data[idx] = raw_val;
    rgb_green[idx] = (color == 1) ? raw_val : 0.0f;
    
    // Green interpolation for non-green pixels
    if (color != 1 && gid.x > 1 && gid.x < width - 2 && gid.y > 1 && gid.y < height - 2) {
        float sum = 0.0f;
        int count = 0;
        
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                uint ny = gid.y + dy;
                uint nx = gid.x + dx;
                
                if (fcol_xtrans(ny, nx, params.xtrans) == 1) {
                    sum += raw_data[ny * width + nx];
                    count++;
                }
            }
        }
        
        if (count > 0) {
            rgb_green[idx] = (sum / float(count)) / params.maximum_value;
        }
    }
}

// Pass 2: LAB-based color interpolation
kernel void demosaic_xtrans_3pass_pass2(
    const device float* cfa_data [[buffer(0)]],
    const device float* rgb_green [[buffer(1)]],
    device float* lab_data [[buffer(2)]],
    device float* rgb_data [[buffer(3)]],
    constant XTransParams& params [[buffer(4)]],
    constant float* cbrt_lut [[buffer(5)]],
    constant float* xyz_cam_flat [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    
    if (gid.x >= width || gid.y >= height) return;
    
    uint idx = gid.y * width + gid.x;
    uint color = fcol_xtrans(gid.y, gid.x, params.xtrans);
    
    float3 rgb = {0.0f};
    rgb.g = rgb_green[idx];
    rgb.r = (color == 0) ? cfa_data[idx] : 0.0f;
    rgb.b = (color == 2) ? cfa_data[idx] : 0.0f;
    
    // Interpolate missing channels
    if (gid.x > 1 && gid.x < width - 2 && gid.y > 1 && gid.y < height - 2) {
        for (int c = 0; c < 3; c += 2) {
            if (color == (uint)c) continue;
            
            float sum = 0.0f;
            int count = 0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    uint ny = gid.y + dy;
                    uint nx = gid.x + dx;
                    uint n_idx = ny * width + nx;
                    
                    if (fcol_xtrans(ny, nx, params.xtrans) == (uint)c) {
                        sum += cfa_data[n_idx];
                        count++;
                    }
                }
            }
            if (count > 0) {
                if (c == 0) rgb.r = sum / float(count);
                else rgb.b = sum / float(count);
            }
        }
    }
    
    // Store preliminary RGB
    rgb_data[idx * 3 + 0] = rgb.r;
    rgb_data[idx * 3 + 1] = rgb.g;
    rgb_data[idx * 3 + 2] = rgb.b;
    
    // Convert to LAB and store
    float3 lab = rgb_to_lab(rgb, cbrt_lut, xyz_cam_flat);
    lab_data[idx * 3 + 0] = lab.x;
    lab_data[idx * 3 + 1] = lab.y;
    lab_data[idx * 3 + 2] = lab.z;
}

// Pass 3: Final refinement
kernel void demosaic_xtrans_3pass_pass3(
    device float* rgb_data [[buffer(0)]],
    const device float* lab_data [[buffer(1)]],
    constant XTransParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = params.width;
    uint height = params.height;
    
    if (gid.x < params.border_width || gid.x >= width - params.border_width ||
        gid.y < params.border_width || gid.y >= height - params.border_width) {
        return;
    }
    
    uint idx = gid.y * width + gid.x;
    float3 center_lab = {lab_data[idx * 3 + 0], lab_data[idx * 3 + 1], lab_data[idx * 3 + 2]};
    
    float3 sum_rgb = {0.0f};
    float total_weight = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint ny = gid.y + dy;
            uint nx = gid.x + dx;
            
            uint n_idx = ny * width + nx;
            float3 neighbor_lab = {lab_data[n_idx * 3 + 0], lab_data[n_idx * 3 + 1], lab_data[n_idx * 3 + 2]};
            
            float lab_dist = distance(center_lab, neighbor_lab);
            float spatial_dist = sqrt(float(dx * dx + dy * dy));
            float weight = exp(-(lab_dist * lab_dist) / 0.1f - (spatial_dist * spatial_dist) / 1.0f);
            
            sum_rgb.r += rgb_data[n_idx * 3 + 0] * weight;
            sum_rgb.g += rgb_data[n_idx * 3 + 1] * weight;
            sum_rgb.b += rgb_data[n_idx * 3 + 2] * weight;
            total_weight += weight;
        }
    }
    
    if (total_weight > 0.0f) {
        rgb_data[idx * 3 + 0] = sum_rgb.r / total_weight;
        rgb_data[idx * 3 + 1] = sum_rgb.g / total_weight;
        rgb_data[idx * 3 + 2] = sum_rgb.b / total_weight;
    }
}