# Kuwahara Filter in C (Classic & Anisotropic)

A fast, zero-dependency C implementation of the **Classic Circular Gaussian Kuwahara** and **Anisotropic Kuwahara** non-photorealistic rendering (NPR) filters. It reads and writes binary `P6` PPM images using a single-pass in-memory I/O pipeline and scales across CPU cores with OpenMP.


| Original | Filtered |
|:---:|:---:|
| ![Original](pexels-fernando-silva-2157007356-35587125.jpg) | ![Filtered](output.png) |

## Features

- **Zero External Image Libraries:** Custom, spec-compliant `P6` PPM parser that loads the file in a single `fread` call, handles arbitrary header comments, and guards against integer overflows and pixel-alignment shifts.
- **Circular + Gaussian Kernel Weighting:** Eliminates the blocky, axis-aligned artifacts of square-window Kuwahara filters by strictly bounding samples to a circle ($x^2 + y^2 \le r^2$) with a 2D Gaussian radial decay ($\sigma_r = r/2$).
- **Anisotropic Mode (Kyprianidis et al., 2009):** Adapts kernel shape and orientation to local image edges using the 2D Structure Tensor, preserving sharp boundaries and producing smooth, painterly brushstrokes.
- **Classic Mode:** Divides the circular Gaussian disk into 4 overlapping quadrants using a precomputed lookup table (LUT) to avoid per-pixel trigonometric or exponential overhead.
- **Multithreaded Execution:** Parallelized across image rows and structure tensor passes using OpenMP.

---

## Algorithmic Pipeline

### 1. Classic Circular Gaussian Mode
1. Precomputes a circular lookup table (`ClassicSample`) of offsets $(dx, dy)$ within $dx^2 + dy^2 \le r^2$, paired with radial Gaussian weights $w = \exp\left(-\frac{dx^2 + dy^2}{2\sigma_r^2}\right)$ and quadrant bitmasks.
2. Evaluates weighted RGB means and variances across the 4 overlapping circular quadrants per pixel.
3. Outputs the mean color of the quadrant with the minimum total RGB variance.

### 2. Anisotropic Mode
1. **Gradient Estimation:** Applies $3 \times 3$ Sobel filters across RGB channels to compute directional derivatives $G_x$ and $G_y$.
2. **Structure Tensor Construction & Smoothing:** Builds the tensor components $(E, F, G) = (G_x^2, G_x G_y, G_y^2)$ and smooths them with a separable 1D Gaussian pass ($\sigma = 2.0$).
3. **Local Orientation & Anisotropy:** Extracts eigenvalues $\lambda_1, \lambda_2$ from the smoothed tensor to determine the dominant edge orientation $\phi = \frac{1}{2}\text{atan2}(2F, E - G)$ and local anisotropy $A = \frac{\lambda_1 - \lambda_2}{\lambda_1 + \lambda_2} \in [0, 1]$.
4. **Warped Elliptical Sector Integration:** Rotates each neighbor by $-\phi$ and scales the axes by $(1 + A)$ along the edge and $\frac{1}{1 + A}$ perpendicular to the edge. Samples a precomputed 8-sector circular Gaussian LUT ($65 \times 65$ resolution) and blends the 8 sector means using polynomial variance weights $w_k = (1 + \sigma_k^2)^{-4}$.

---

## Requirements

- A C99-compliant compiler (`gcc` or `clang`)
- OpenMP support (`libomp` / `libgomp`)
- Standard C Math library (`-lm`)
- *Optional:* ImageMagick, FFmpeg, or Netpbm for converting PNG/JPEG files to and from `.ppm`

---


## Usage

### 1. Compile

Build the project using GCC with OpenMP support:

```bash
gcc -O3 -march=native -fopenmp kuwa4.c -o kuwahara -lm
```

### 2. Prepare your image

The program accepts binary PPM images (`P6` format).

If your image is in PNG or JPEG format, convert it using ImageMagick:

```bash
magick input.jpg input.ppm
```

### 3. Run the filter

Basic usage:

```bash
./kuwahara input.ppm output.ppm
```

This runs the Anisotropic Kuwahara filter with the default radius and maximum available OpenMP threads.

### 4. Configure the filter

You can optionally customize the radius, number of threads, and filter mode:

```bash
./kuwahara input.ppm output.ppm [radius] [threads] [mode]
```

**Examples:**

Run the Classic Kuwahara filter:

```bash
./kuwahara input.ppm output.ppm 4 8 classic
```

Run the Anisotropic Kuwahara filter:

```bash
./kuwahara input.ppm output.ppm 6 8 anisotropic
```

Use the default anisotropic mode with a larger radius:

```bash
./kuwahara input.ppm output.ppm 8
```

### 5. Convert the result

The output is saved as a PPM image. You can convert it back to PNG:

```bash
magick output.ppm output.png
```

### Command-line options

| Parameter | Description | Default |
|---|---|---|
| `radius` | Controls the neighborhood size of the filter | `4` |
| `threads` | Number of CPU threads used by OpenMP | Maximum available |
| `mode` | Filter algorithm: `classic` or `anisotropic` | `anisotropic` |

**Note:** Arguments must be provided in order. To specify the mode, you must also provide the radius and thread count.
