#include "groot-preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace groot {

uint16_t bf16_round_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);  // round to bf16, ties to even
    bits &= 0xffff0000u;
    float rounded;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    // Reinterpret the half's bits: a numeric cast to uint16_t would truncate to 0.
    const _Float16 half = static_cast<_Float16>(rounded);
    uint16_t out;
    std::memcpy(&out, &half, sizeof(out));
    return out;
}

namespace {

struct Taps {
    int index[4];
    float weight[4];
    int count;
};

std::vector<Taps> make_taps(int src, int dst) {
    const double scale = double(src) / double(dst);
    std::vector<Taps> out(dst);
    for (int d = 0; d < dst; ++d) {
        const double a = d * scale;
        const double b = (d + 1) * scale;
        Taps & t = out[d];
        t.count = 0;
        const int first = int(std::floor(a));
        const int last  = std::min(int(std::ceil(b)), src);
        for (int i = first; i < last && t.count < 4; ++i) {
            const double lo = std::max(a, double(i));
            const double hi = std::min(b, double(i + 1));
            if (hi <= lo) {
                continue;
            }
            t.index[t.count]  = i;
            t.weight[t.count] = float((hi - lo) / scale);
            ++t.count;
        }
    }
    return out;
}

uint8_t round_pixel(float v) {
    if (!(v > 0.0f)) {
        return 0;  // also maps NaN to 0
    }
    if (v >= 255.0f) {
        return 255;
    }
    return static_cast<uint8_t>(std::floor(v + 0.5f));
}

uint16_t lut_value(int byte) {
    const float x = (byte * (1.0f / 255.0f) - 0.5f) / 0.5f;
    return bf16_round_to_half(x);
}

}  // namespace

void resize_area_rgb(const uint8_t * src, int sw, int sh, int dw, int dh, uint8_t * dst) {
    const std::vector<Taps> xt = make_taps(sw, dw);
    const std::vector<Taps> yt = make_taps(sh, dh);
    for (int y = 0; y < dh; ++y) {
        const Taps & ty = yt[y];
        for (int x = 0; x < dw; ++x) {
            const Taps & tx = xt[x];
            float acc[3] = { 0.0f, 0.0f, 0.0f };
            for (int j = 0; j < ty.count; ++j) {
                const uint8_t * row = src + size_t(ty.index[j]) * sw * 3u;
                for (int i = 0; i < tx.count; ++i) {
                    const float w = ty.weight[j] * tx.weight[i];
                    const uint8_t * p = row + size_t(tx.index[i]) * 3u;
                    acc[0] += w * p[0];
                    acc[1] += w * p[1];
                    acc[2] += w * p[2];
                }
            }
            uint8_t * o = dst + (size_t(y) * dw + x) * 3u;
            o[0] = round_pixel(acc[0]);
            o[1] = round_pixel(acc[1]);
            o[2] = round_pixel(acc[2]);
        }
    }
}

void pad_image_to_patches(const uint8_t * image, uint16_t * out, int view) {
    static const std::vector<uint16_t> lut = [] {
        std::vector<uint16_t> table(256);
        for (int i = 0; i < 256; ++i) {
            table[i] = lut_value(i);
        }
        return table;
    }();

    // offsets[i] selects pixel (i/16, i%16) inside a 16x16 patch of the 256-wide image.
    uint16_t offsets[256];
    for (int i = 0; i < 256; ++i) {
        offsets[i] = uint16_t((i / 16) * 256 * 3 + (i % 16) * 3);
    }

    for (int gh = 0; gh < 8; ++gh) {
        for (int gw = 0; gw < 8; ++gw) {
            for (int mh = 0; mh < 2; ++mh) {
                for (int mw = 0; mw < 2; ++mw) {
                    const int patch = (gh * 8 + gw) * 4 + mh * 2 + mw;
                    const uint8_t * src =
                        image + ((gh * 2 + mh) * 16 * 256 + (gw * 2 + mw) * 16) * 3;
                    for (int channel = 0; channel < 3; ++channel) {
                        uint16_t values[256];
                        for (int i = 0; i < 256; ++i) {
                            values[i] = lut[src[offsets[i] + channel]];
                        }
                        uint16_t * dst =
                            out + (size_t(view) * 256 + patch) * kPatchWidth + channel * 512;
                        std::memcpy(dst, values, sizeof(values));
                        std::memcpy(dst + 256, values, sizeof(values));  // temporal repeat
                    }
                }
            }
        }
    }
}

void images_to_patches(const uint8_t * front, const uint8_t * wrist, uint16_t * out) {
    constexpr int crop  = 243;  // int(256 * 0.95)
    constexpr int first = (kImageSide - crop) / 2;
    uint8_t contiguous[crop * crop * 3];
    uint8_t resized[kImageSide * kImageSide * 3];
    const uint8_t * views[2] = { front, wrist };
    for (int view = 0; view < 2; ++view) {
        const uint8_t * source = views[view];
        // The crop lives inside a 256-wide image: repack it so the resizer can use
        // a 243-wide row stride.
        for (int y = 0; y < crop; ++y) {
            std::memcpy(contiguous + size_t(y) * crop * 3,
                        source + (size_t(first + y) * kImageSide + first) * 3u, size_t(crop) * 3);
        }
        resize_area_rgb(contiguous, crop, crop, kImageSide, kImageSide, resized);
        pad_image_to_patches(resized, out, view);
    }
}

void normalize_state(const float * state, const float * lo, const float * hi, uint16_t * out) {
    std::memset(out, 0, sizeof(uint16_t) * 132);
    for (int i = 0; i < kStateWidth; ++i) {
        float v = 2.0f * (state[i] - lo[i]) / (hi[i] - lo[i]) - 1.0f;
        v = std::min(1.0f, std::max(-1.0f, v));
        out[i] = bf16_round_to_half(v);
    }
}

std::string clean_instruction(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z') {
            out.push_back(char(c - 'A' + 'a'));
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                   c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
            out.push_back(char(c));
        }
    }
    return out;
}

std::string build_prompt(const std::string & cleaned) {
    std::string image = "<|vision_start|>";
    for (int i = 0; i < 64; ++i) {
        image += "<|image_pad|>";
    }
    image += "<|vision_end|>";
    return "<|im_start|>user\n" + image + image + cleaned + "<|im_end|>\n";
}

}  // namespace groot
