// GR00T LIBERO observation preprocessing for the llama.cpp VLA pipeline.
//
// The exported ViT consumes raw camera pixels, not normalized CLIP tensors:
//   256x256 RGB -> 95% centre crop -> area resize back to 256x256
//   -> 16x16 patches, 2x2 merge, temporal repeat 2 -> [512, 1536] fp16
// The arithmetic mirrors the baseline runtime (groot_patch_pixels + the Python
// Preprocessor) so both paths see the same numbers.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace groot {

constexpr int kImageSide    = 256;
constexpr int kPatchTokens  = 512;   // two cameras x 256 patches
constexpr int kPatchWidth   = 1536;  // 3 channels x 16x16 pixels x 2 temporal
constexpr int kImageTokens  = 128;   // 2 x (8x8 patches / 2x2 merge)
constexpr int kActionWidth  = 132;
constexpr int kActionLength = 40;
constexpr int kStateWidth   = 8;     // x y z roll pitch yaw gripper[2]

// Round-to-nearest bf16 followed by an fp16 conversion, matching the baseline's
// bf16_as_half helper (used for the pixel LUT and the state vector).
uint16_t bf16_round_to_half(float value);

// Box-overlap resize (OpenCV INTER_AREA) for the 95% crop back to 256x256.
// Returns round-nearest uint8, which is within one LSB of OpenCV on the tested
// inputs; see tools/mtmd/groot/README.md.
void resize_area_rgb(const uint8_t * src, int sw, int sh, int dw, int dh, uint8_t * dst);

// Crop + resize one camera and then pack both views into the ViT input.
void images_to_patches(const uint8_t * front, const uint8_t * wrist, uint16_t * out);

// 2x2 merge ordering and the [256, 3] -> [3, 512] transposition of groot_patch_pixels.
void pad_image_to_patches(const uint8_t * image, uint16_t * out, int view);

// Normalize the raw state with the LIBERO min/max contract and bf16-round it.
void normalize_state(const float * state, const float * lo, const float * hi, uint16_t * out);

// Instruction cleaning: lowercase, drop everything except word characters and
// whitespace (the baseline's re.sub(r'[^\w\s]', '', text.lower())).
std::string clean_instruction(const std::string & text);

// The GR00T prompt: two 64-token image spans in a single user turn.
std::string build_prompt(const std::string & cleaned);

// Token layout: positions are mrope (t, h, w) with an 8x8 grid per image.
struct TokenLayout {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;  // 3 x S, row-major [axis][token]
    std::vector<uint8_t> visual_mask;
};

}  // namespace groot
