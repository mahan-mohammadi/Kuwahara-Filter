#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_SECTORS 8
#define LUT_RES 65

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} Pixel;

typedef struct {
    float e;
    float f;
    float g;
} Tensor;

typedef struct {
    int dx;
    int dy;
    double weight;
    uint8_t q_mask;
} ClassicSample;

static inline int clamp(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static int parse_next_token(const uint8_t **cursor, const uint8_t *end, char *token, size_t max_len) {
    const uint8_t *p = *cursor;

    while (p < end) {
        if (isspace(*p)) {
            p++;
            continue;
        }
        if (*p == '#') {
            while (p < end && *p != '\n' && *p != '\r') {
                p++;
            }
            continue;
        }
        break;
    }

    if (p >= end) {
        *cursor = p;
        return 0;
    }

    size_t i = 0;
    while (p < end && !isspace(*p) && *p != '#') {
        if (i + 1 < max_len) {
            token[i++] = (char)*p;
        }
        p++;
    }
    token[i] = '\0';
    *cursor = p;
    return i > 0;
}

Pixel *load_ppm(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open input file '%s'.\n", filename);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error: Cannot seek input file.\n");
        fclose(file);
        return NULL;
    }

    long file_size = ftell(file);
    if (file_size <= 0) {
        fprintf(stderr, "Error: Invalid or empty file.\n");
        fclose(file);
        return NULL;
    }
    rewind(file);

    uint8_t *raw_buffer = malloc((size_t)file_size);
    if (!raw_buffer) {
        fprintf(stderr, "Error: Memory allocation failed for file buffer.\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(raw_buffer, 1, (size_t)file_size, file);
    fclose(file);

    if (bytes_read != (size_t)file_size) {
        fprintf(stderr, "Error: Failed to read entire file.\n");
        free(raw_buffer);
        return NULL;
    }

    const uint8_t *cursor = raw_buffer;
    const uint8_t *end = raw_buffer + bytes_read;
    char token[64];

    if (!parse_next_token(&cursor, end, token, sizeof(token)) || strcmp(token, "P6") != 0) {
        fprintf(stderr, "Error: Only binary P6 PPM format is supported.\n");
        free(raw_buffer);
        return NULL;
    }

    if (!parse_next_token(&cursor, end, token, sizeof(token))) {
        fprintf(stderr, "Error: Missing width in PPM header.\n");
        free(raw_buffer);
        return NULL;
    }
    long w = strtol(token, NULL, 10);

    if (!parse_next_token(&cursor, end, token, sizeof(token))) {
        fprintf(stderr, "Error: Missing height in PPM header.\n");
        free(raw_buffer);
        return NULL;
    }
    long h = strtol(token, NULL, 10);

    if (!parse_next_token(&cursor, end, token, sizeof(token))) {
        fprintf(stderr, "Error: Missing maxval in PPM header.\n");
        free(raw_buffer);
        return NULL;
    }
    long max_val = strtol(token, NULL, 10);

    if (w <= 0 || h <= 0 || w > INT_MAX || h > INT_MAX || max_val != 255) {
        fprintf(stderr, "Error: Invalid dimensions (%ldx%ld) or maxval (%ld).\n", w, h, max_val);
        free(raw_buffer);
        return NULL;
    }

    while (cursor < end && *cursor == '#') {
        while (cursor < end && *cursor != '\n' && *cursor != '\r') {
            cursor++;
        }
    }

    if (cursor >= end || !isspace(*cursor)) {
        fprintf(stderr, "Error: Malformed header delimiter before pixel data.\n");
        free(raw_buffer);
        return NULL;
    }

    if (*cursor == '\r' && (cursor + 1) < end && *(cursor + 1) == '\n') {
        cursor += 2;
    } else {
        cursor += 1;
    }

    size_t pixel_count = (size_t)w * (size_t)h;
    if (pixel_count > SIZE_MAX / sizeof(Pixel)) {
        fprintf(stderr, "Error: Image dimensions cause integer overflow.\n");
        free(raw_buffer);
        return NULL;
    }

    size_t required_bytes = pixel_count * sizeof(Pixel);
    size_t remaining_bytes = (size_t)(end - cursor);

    if (remaining_bytes < required_bytes) {
        fprintf(stderr, "Error: Incomplete PPM pixel data (expected %zu bytes, got %zu).\n",
                required_bytes, remaining_bytes);
        free(raw_buffer);
        return NULL;
    }

    Pixel *image = malloc(required_bytes);
    if (!image) {
        fprintf(stderr, "Error: Memory allocation failed for image.\n");
        free(raw_buffer);
        return NULL;
    }

    memcpy(image, cursor, required_bytes);
    free(raw_buffer);

    *width = (int)w;
    *height = (int)h;
    return image;
}

int save_ppm(const char *filename, const Pixel *image, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create output file '%s'.\n", filename);
        return 0;
    }

    size_t io_buf_size = 1024 * 1024;
    char *io_buffer = malloc(io_buf_size);
    if (io_buffer) {
        setvbuf(file, io_buffer, _IOFBF, io_buf_size);
    }

    int header_ok = fprintf(file, "P6\n%d %d\n255\n", width, height) > 0;
    size_t pixel_count = (size_t)width * (size_t)height;
    size_t written = 0;

    if (header_ok) {
        written = fwrite(image, sizeof(Pixel), pixel_count, file);
    }

    int flush_ok = (fflush(file) == 0);
    fclose(file);
    free(io_buffer);

    return header_ok && (written == pixel_count) && flush_ok;
}

int kuwahara_filter_classic_circular_gaussian(const Pixel * restrict input, Pixel * restrict output, int width, int height, int radius) {
    int side = 2 * radius + 1;
    size_t max_samples = (size_t)side * (size_t)side;

    ClassicSample *samples = malloc(max_samples * sizeof(ClassicSample));
    if (!samples) {
        return 0;
    }

    double sigma = (double)radius * 0.5;
    double inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    int r_sq = radius * radius;
    size_t sample_count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int d2 = dx * dx + dy * dy;
            if (d2 > r_sq) {
                continue;
            }

            uint8_t mask = 0;
            if (dx <= 0 && dy <= 0) mask |= (1 << 0);
            if (dx >= 0 && dy <= 0) mask |= (1 << 1);
            if (dx <= 0 && dy >= 0) mask |= (1 << 2);
            if (dx >= 0 && dy >= 0) mask |= (1 << 3);

            samples[sample_count].dx = dx;
            samples[sample_count].dy = dy;
            samples[sample_count].weight = exp(-(double)d2 * inv_two_sigma_sq);
            samples[sample_count].q_mask = mask;
            sample_count++;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum_r[4] = {0}, sum_g[4] = {0}, sum_b[4] = {0};
            double sq_r[4]  = {0}, sq_g[4]  = {0}, sq_b[4]  = {0};
            double w_sum[4] = {0};

            for (size_t i = 0; i < sample_count; i++) {
                int px = clamp(x + samples[i].dx, 0, width - 1);
                int py = clamp(y + samples[i].dy, 0, height - 1);
                double w = samples[i].weight;
                uint8_t mask = samples[i].q_mask;

                Pixel p = input[(size_t)py * width + px];
                double pr = p.r, pg = p.g, pb = p.b;
                double wr = w * pr, wg = w * pg, wb = w * pb;
                double wr2 = wr * pr, wg2 = wg * pg, wb2 = wb * pb;

                for (int q = 0; q < 4; q++) {
                    if (mask & (1 << q)) {
                        sum_r[q] += wr;
                        sum_g[q] += wg;
                        sum_b[q] += wb;
                        sq_r[q]  += wr2;
                        sq_g[q]  += wg2;
                        sq_b[q]  += wb2;
                        w_sum[q] += w;
                    }
                }
            }

            double best_var = 1e30;
            double best_mean[3] = {0.0, 0.0, 0.0};

            for (int q = 0; q < 4; q++) {
                double inv_w = 1.0 / w_sum[q];
                double mr = sum_r[q] * inv_w;
                double mg = sum_g[q] * inv_w;
                double mb = sum_b[q] * inv_w;

                double vr = sq_r[q] * inv_w - mr * mr;
                double vg = sq_g[q] * inv_w - mg * mg;
                double vb = sq_b[q] * inv_w - mb * mb;

                double total_var = fmax(0.0, vr) + fmax(0.0, vg) + fmax(0.0, vb);
                if (total_var < best_var) {
                    best_var = total_var;
                    best_mean[0] = mr;
                    best_mean[1] = mg;
                    best_mean[2] = mb;
                }
            }

            Pixel result;
            result.r = (uint8_t)clamp((int)(best_mean[0] + 0.5), 0, 255);
            result.g = (uint8_t)clamp((int)(best_mean[1] + 0.5), 0, 255);
            result.b = (uint8_t)clamp((int)(best_mean[2] + 0.5), 0, 255);
            output[(size_t)y * width + x] = result;
        }
    }

    free(samples);
    return 1;
}

static void build_gaussian_circle_sector_lut(float lut[NUM_SECTORS][LUT_RES][LUT_RES]) {
    const int half = LUT_RES / 2;
    const float sigma_r = 0.5f;
    const float inv_two_sigma_r_sq = 1.0f / (2.0f * sigma_r * sigma_r);
    const float sigma_a = (float)M_PI / (float)NUM_SECTORS * 0.6f;
    const float inv_two_sigma_a_sq = 1.0f / (2.0f * sigma_a * sigma_a);

    for (int s = 0; s < NUM_SECTORS; s++) {
        float sector_angle = (float)s * (2.0f * (float)M_PI / (float)NUM_SECTORS);

        for (int iy = 0; iy < LUT_RES; iy++) {
            float v = (float)(iy - half) / (float)half;
            for (int ix = 0; ix < LUT_RES; ix++) {
                float u = (float)(ix - half) / (float)half;
                float r2 = u * u + v * v;

                if (r2 > 1.0f) {
                    lut[s][iy][ix] = 0.0f;
                    continue;
                }

                float radial_gaussian = expf(-r2 * inv_two_sigma_r_sq);

                if (r2 < 1e-6f) {
                    lut[s][iy][ix] = radial_gaussian;
                    continue;
                }

                float angle = atan2f(v, u);
                float diff = angle - sector_angle;
                while (diff > (float)M_PI)  diff -= 2.0f * (float)M_PI;
                while (diff < -(float)M_PI) diff += 2.0f * (float)M_PI;

                float angular_gaussian = expf(-(diff * diff) * inv_two_sigma_a_sq);
                lut[s][iy][ix] = radial_gaussian * angular_gaussian;
            }
        }
    }
}

static void compute_structure_tensor(const Pixel * restrict input, Tensor * restrict tensor, int width, int height) {
    static const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    static const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float gx_r = 0.0f, gx_g = 0.0f, gx_b = 0.0f;
            float gy_r = 0.0f, gy_g = 0.0f, gy_b = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                int py = clamp(y + ky, 0, height - 1);
                for (int kx = -1; kx <= 1; kx++) {
                    int px = clamp(x + kx, 0, width - 1);
                    Pixel p = input[(size_t)py * width + px];

                    float wx = (float)sobel_x[ky + 1][kx + 1] / 8.0f;
                    float wy = (float)sobel_y[ky + 1][kx + 1] / 8.0f;

                    gx_r += p.r * wx;
                    gx_g += p.g * wx;
                    gx_b += p.b * wx;

                    gy_r += p.r * wy;
                    gy_g += p.g * wy;
                    gy_b += p.b * wy;
                }
            }

            size_t idx = (size_t)y * width + x;
            tensor[idx].e = gx_r * gx_r + gx_g * gx_g + gx_b * gx_b;
            tensor[idx].f = gx_r * gy_r + gx_g * gy_g + gx_b * gy_b;
            tensor[idx].g = gy_r * gy_r + gy_g * gy_g + gy_b * gy_b;
        }
    }
}

static int smooth_structure_tensor(Tensor * restrict tensor, int width, int height, float sigma) {
    int k_radius = (int)ceilf(3.0f * sigma);
    int k_size = 2 * k_radius + 1;

    float *kernel = malloc((size_t)k_size * sizeof(float));
    Tensor *temp = malloc((size_t)width * (size_t)height * sizeof(Tensor));
    if (!kernel || !temp) {
        free(kernel);
        free(temp);
        return 0;
    }

    float sum = 0.0f;
    float inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    for (int i = -k_radius; i <= k_radius; i++) {
        float w = expf(-(float)(i * i) * inv_two_sigma_sq);
        kernel[i + k_radius] = w;
        sum += w;
    }
    for (int i = 0; i < k_size; i++) {
        kernel[i] /= sum;
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float e = 0.0f, f = 0.0f, g = 0.0f;
            for (int k = -k_radius; k <= k_radius; k++) {
                int px = clamp(x + k, 0, width - 1);
                float w = kernel[k + k_radius];
                Tensor t = tensor[(size_t)y * width + px];
                e += t.e * w;
                f += t.f * w;
                g += t.g * w;
            }
            size_t idx = (size_t)y * width + x;
            temp[idx].e = e;
            temp[idx].f = f;
            temp[idx].g = g;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float e = 0.0f, f = 0.0f, g = 0.0f;
            for (int k = -k_radius; k <= k_radius; k++) {
                int py = clamp(y + k, 0, height - 1);
                float w = kernel[k + k_radius];
                Tensor t = temp[(size_t)py * width + x];
                e += t.e * w;
                f += t.f * w;
                g += t.g * w;
            }
            size_t idx = (size_t)y * width + x;
            tensor[idx].e = e;
            tensor[idx].f = f;
            tensor[idx].g = g;
        }
    }

    free(kernel);
    free(temp);
    return 1;
}

int kuwahara_filter_anisotropic(const Pixel * restrict input, Pixel * restrict output, int width, int height, int radius) {
    size_t pixel_count = (size_t)width * (size_t)height;
    Tensor *tensor = malloc(pixel_count * sizeof(Tensor));
    if (!tensor) {
        return 0;
    }

    compute_structure_tensor(input, tensor, width, height);

    if (!smooth_structure_tensor(tensor, width, height, 2.0f)) {
        free(tensor);
        return 0;
    }

    float (*sector_lut)[LUT_RES][LUT_RES] = malloc(sizeof(float[NUM_SECTORS][LUT_RES][LUT_RES]));
    if (!sector_lut) {
        free(tensor);
        return 0;
    }
    build_gaussian_circle_sector_lut(sector_lut);

    const float alpha = 1.0f;
    const float inv_radius = 1.0f / (float)radius;
    const float lut_half = (float)(LUT_RES / 2);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Tensor t = tensor[(size_t)y * width + x];

            float diff = t.e - t.g;
            float disc = sqrtf(diff * diff + 4.0f * t.f * t.f);
            float lambda1 = 0.5f * (t.e + t.g + disc);
            float lambda2 = 0.5f * (t.e + t.g - disc);

            float anisotropy = 0.0f;
            if (lambda1 + lambda2 > 1e-6f) {
                anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2);
                anisotropy = fminf(fmaxf(anisotropy, 0.0f), 1.0f);
            }

            float phi = 0.5f * atan2f(2.0f * t.f, t.e - t.g);
            float cos_phi = cosf(phi);
            float sin_phi = sinf(phi);

            float scale_u = (alpha + anisotropy) / alpha * inv_radius;
            float scale_v = alpha / (alpha + anisotropy) * inv_radius;

            double sum_r[NUM_SECTORS] = {0}, sum_g[NUM_SECTORS] = {0}, sum_b[NUM_SECTORS] = {0};
            double sq_r[NUM_SECTORS]  = {0}, sq_g[NUM_SECTORS]  = {0}, sq_b[NUM_SECTORS]  = {0};
            double weight_sum[NUM_SECTORS] = {0};

            int max_extent = (int)ceilf((float)radius * (alpha + anisotropy) / alpha);
            int y_min = clamp(y - max_extent, 0, height - 1);
            int y_max = clamp(y + max_extent, 0, height - 1);
            int x_min = clamp(x - max_extent, 0, width - 1);
            int x_max = clamp(x + max_extent, 0, width - 1);

            for (int ny = y_min; ny <= y_max; ny++) {
                float dy = (float)(ny - y);
                for (int nx = x_min; nx <= x_max; nx++) {
                    float dx = (float)(nx - x);

                    float u = ( dx * cos_phi + dy * sin_phi) * scale_u;
                    float v = (-dx * sin_phi + dy * cos_phi) * scale_v;

                    float dist_sq = u * u + v * v;
                    if (dist_sq > 1.0f) {
                        continue;
                    }

                    int lut_x = clamp((int)roundf(u * lut_half + lut_half), 0, LUT_RES - 1);
                    int lut_y = clamp((int)roundf(v * lut_half + lut_half), 0, LUT_RES - 1);

                    Pixel p = input[(size_t)ny * width + nx];
                    double pr = p.r, pg = p.g, pb = p.b;
                    double pr2 = pr * pr, pg2 = pg * pg, pb2 = pb * pb;

                    for (int s = 0; s < NUM_SECTORS; s++) {
                        double w = (double)sector_lut[s][lut_y][lut_x];
                        if (w <= 1e-6) {
                            continue;
                        }
                        sum_r[s] += w * pr;
                        sum_g[s] += w * pg;
                        sum_b[s] += w * pb;
                        sq_r[s]  += w * pr2;
                        sq_g[s]  += w * pg2;
                        sq_b[s]  += w * pb2;
                        weight_sum[s] += w;
                    }
                }
            }

            double final_r = 0.0, final_g = 0.0, final_b = 0.0;
            double total_alpha = 0.0;

            for (int s = 0; s < NUM_SECTORS; s++) {
                if (weight_sum[s] <= 1e-12) {
                    continue;
                }

                double inv_w = 1.0 / weight_sum[s];
                double mr = sum_r[s] * inv_w;
                double mg = sum_g[s] * inv_w;
                double mb = sum_b[s] * inv_w;

                double vr = sq_r[s] * inv_w - mr * mr;
                double vg = sq_g[s] * inv_w - mg * mg;
                double vb = sq_b[s] * inv_w - mb * mb;

                double sigma_sq = fmax(0.0, vr) + fmax(0.0, vg) + fmax(0.0, vb);
                double sigma2 = 1.0 + sigma_sq;
                double sigma4 = sigma2 * sigma2;
                double sector_w = 1.0 / (sigma4 * sigma4);

                final_r += mr * sector_w;
                final_g += mg * sector_w;
                final_b += mb * sector_w;
                total_alpha += sector_w;
            }

            Pixel result;
            if (total_alpha > 0.0) {
                double inv_alpha = 1.0 / total_alpha;
                result.r = (uint8_t)clamp((int)(final_r * inv_alpha + 0.5), 0, 255);
                result.g = (uint8_t)clamp((int)(final_g * inv_alpha + 0.5), 0, 255);
                result.b = (uint8_t)clamp((int)(final_b * inv_alpha + 0.5), 0, 255);
            } else {
                result = input[(size_t)y * width + x];
            }

            output[(size_t)y * width + x] = result;
        }
    }

    free(sector_lut);
    free(tensor);
    return 1;
}

static int parse_positive_int(const char *str, int *out_val) {
    char *endptr = NULL;
    errno = 0;
    long val = strtol(str, &endptr, 10);
    if (errno != 0 || endptr == str || *endptr != '\0' || val < 1 || val > INT_MAX) {
        return 0;
    }
    *out_val = (int)val;
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s input.ppm output.ppm [radius] [threads] [mode: classic|anisotropic]\n", argv[0]);
        return 1;
    }

    int radius = 4;
    int threads = omp_get_max_threads();
    int use_anisotropic = 1;

    if (argc >= 4 && !parse_positive_int(argv[3], &radius)) {
        fprintf(stderr, "Error: Radius must be a valid positive integer.\n");
        return 1;
    }

    if (argc >= 5 && !parse_positive_int(argv[4], &threads)) {
        fprintf(stderr, "Error: Threads must be a valid positive integer.\n");
        return 1;
    }

    if (argc >= 6) {
        if (strcmp(argv[5], "classic") == 0) {
            use_anisotropic = 0;
        } else if (strcmp(argv[5], "anisotropic") == 0) {
            use_anisotropic = 1;
        } else {
            fprintf(stderr, "Error: Mode must be 'classic' or 'anisotropic'.\n");
            return 1;
        }
    }

    omp_set_num_threads(threads);

    int width = 0;
    int height = 0;

    Pixel *input = load_ppm(argv[1], &width, &height);
    if (!input) {
        return 1;
    }

    size_t pixel_count = (size_t)width * (size_t)height;
    Pixel *output = malloc(pixel_count * sizeof(Pixel));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed for output image.\n");
        free(input);
        return 1;
    }

    printf("Image size: %d x %d\n", width, height);
    printf("Filter mode: %s (Circular + Gaussian)\n", use_anisotropic ? "anisotropic" : "classic");
    printf("Radius: %d\n", radius);
    printf("Threads: %d\n", threads);

    double start = omp_get_wtime();

    int status = 0;
    if (use_anisotropic) {
        status = kuwahara_filter_anisotropic(input, output, width, height, radius);
    } else {
        status = kuwahara_filter_classic_circular_gaussian(input, output, width, height, radius);
    }

    if (!status) {
        fprintf(stderr, "Error: Filter execution failed due to memory allocation.\n");
        free(input);
        free(output);
        return 1;
    }

    double end = omp_get_wtime();
    printf("Processing time: %.6f seconds\n", end - start);

    if (!save_ppm(argv[2], output, width, height)) {
        fprintf(stderr, "Error: Could not save output image.\n");
        free(input);
        free(output);
        return 1;
    }

    printf("Output saved: %s\n", argv[2]);

    free(input);
    free(output);
    return 0;
}
