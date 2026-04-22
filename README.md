# MLIR Showcase — MobileNetV2 with IREE

This project demonstrates an end-to-end workflow for exporting a TensorFlow model (MobileNetV2) to MLIR, compiling it with [IREE](https://iree.dev/), and running inference on the CPU backend. It also shows how to inspect the generated MLIR at various stages of the compilation pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Step 1 — Export the TensorFlow Model](#step-1--export-the-tensorflow-model)
4. [Step 2 — Import into MLIR](#step-2--import-into-mlir)
5. [Step 3 — Compile to a VM FlatBuffer](#step-3--compile-to-a-vm-flatbuffer)
6. [Step 4 — Generate Sample Input](#step-4--generate-sample-input)
7. [Step 5 — Run Inference](#step-5--run-inference)
   - [Option A — `iree-run-module` CLI](#option-a--iree-run-module-cli)
   - [Option B — Python script (`inference.py`)](#option-b--python-script-inferencepy)
   - [Option C — Native C runner (IREE C API)](#option-c--native-c-runner-iree-c-api)
8. [Bonus — Inspecting MLIR at Different Levels](#bonus--inspecting-mlir-at-different-levels)
9. [File Reference](#file-reference)

---

## Prerequisites

- **Python 3.12** with a virtual environment
- **TensorFlow 2.x**
- **IREE packages** — compiler, runtime, and TensorFlow import tool:
  - `iree-base-compiler` — provides `iree-compile`, `iree-opt`
  - `iree-base-runtime` — provides `iree-run-module`
  - `iree-tools-tf` — provides `iree-import-tf`
- **NumPy**

All of the above are included in `requirements.txt`, so setup is just:

```bash
# Create and activate the virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install all Python dependencies (TensorFlow, IREE, NumPy, etc.)
pip install -r requirements.txt
```

---

## Project Structure

```
mlir-study/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── export_model.py                     # Exports MobileNetV2 as a TF SavedModel
├── signatures.py                       # Utility to inspect SavedModel signatures
├── sample.py                           # Generates a random input tensor (.npy)
├── mobilenet_v2_saved_model/           # Exported TF SavedModel directory
│   ├── saved_model.pb                  #   Graph definition + signatures (Protocol Buffer)
│   ├── fingerprint.pb                  #   Model integrity hash (Protocol Buffer)
│   ├── assets/                         #   Extra assets (empty for this model)
│   └── variables/                      #   Model weights
│       ├── variables.data-00000-of-00001  # Actual weight values (binary)
│       └── variables.index                # Index/lookup for the weight shards
├── mobilenet_v2.mlirbc                   # MLIR imported from the SavedModel
├── mobilenet_v2_readable.mlir          # Cleaned-up / canonicalized MLIR
├── mobilenet_v2_llvm_dialect.mlir      # MLIR lowered to LLVM dialect (executable sources)
├── mobilenet_v2_llvm.ll                # LLVM IR extracted from the compilation
├── inference.py                        # Python inference script (uses iree-base-runtime)
├── dog.jpg                             # Sample input image (dog photo for testing) (optional; see below)
├── imagenet_classes.txt                # ImageNet 1000 class names (optional; see below)
├── mobilenet_v2.vmfb                   # Compiled VM FlatBuffer (ready to run)
└── venv/                               # Python virtual environment
```

### About the `.pb` files

The `.pb` files inside `mobilenet_v2_saved_model/` use Google's
[Protocol Buffer](https://protobuf.dev/) binary serialization format:

- **`saved_model.pb`** — contains the **computational graph definition**
  (operations, node connections, signatures). It does **not** store the model
  weights, only the structure.
- **`fingerprint.pb`** — a hash/fingerprint of the model added by recent
  TensorFlow versions for integrity verification and unique identification
  (useful for caching, tracking, etc.).

The actual weight values live in `variables/`. Because `.pb` files are binary
and not human-readable, you can inspect them with TensorFlow's CLI tool:

```bash
# Show all signatures and tensor info in the SavedModel
saved_model_cli show --dir mobilenet_v2_saved_model --all
```

---

## Step 1 — Export the TensorFlow Model

Download MobileNetV2 with ImageNet weights from Keras and export it as a
TF SavedModel with a fixed input signature `[1, 224, 224, 3]` (batch=1, 224×224 RGB).

```bash
python export_model.py
```

> **What this does:** loads the Keras MobileNetV2, re-exports it with a concrete
> `serve` signature so that IREE can import it without dynamic shape issues.

You can verify the exported signatures at any time with:

```bash
python signatures.py
# Expected output: Signatures found: ['serve']
```

---

## Step 2 — Import into MLIR

Convert the TF SavedModel into MLIR using IREE's TensorFlow importer.

```bash
iree-import-tf \
  mobilenet_v2_saved_model \
  --tf-import-type=savedmodel_v1 \
  --tf-savedmodel-exported-names=serve \
  -o mobilenet_v2.mlirbc
```

| Flag | Purpose |
|---|---|
| `--tf-import-type=savedmodel_v1` | Tells the importer the format is a SavedModel (v1-style directory layout). |
| `--tf-savedmodel-exported-names=serve` | Only imports the `serve` function (our inference entry point). |
| `-o mobilenet_v2.mlirbc` | Output file in MLIR bytecode format. |

> **Result:** `mobilenet_v2.mlirbc` — the full model expressed as high-level MLIR
> dialects (e.g., `mhlo`, `stablehlo`, `tf`).

---

## Step 3 — Compile to a VM FlatBuffer

Compile the MLIR to a binary that IREE's runtime can execute on the CPU.

```bash
iree-compile \
  mobilenet_v2.mlirbc \
  --iree-hal-target-backends=llvm-cpu \
  -o mobilenet_v2.vmfb
```

| Flag | Purpose |
|---|---|
| `--iree-hal-target-backends=llvm-cpu` | Targets the LLVM-based CPU backend (generates native code via LLVM). |
| `-o mobilenet_v2.vmfb` | Output as a VM FlatBuffer, IREE's portable deployment format. |

> **Result:** `mobilenet_v2.vmfb` — a self-contained artifact ready for inference.

---

## Step 4 — Generate Sample Input

Preprocess a real image and save the resulting tensor as a `.npy` file.

### Download optional assets (`dog.jpg` and `imagenet_classes.txt`)

If you don't have them locally (or you don't want to version them in git), you can download them with:

```bash
curl -L -o dog.jpg "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
curl -L -o imagenet_classes.txt "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
```

Then generate `input.npy`:

```bash
python sample.py                # uses dog.jpg by default
python sample.py photo.png      # or pass any other image
```

> **What this does:** loads the image, resizes it to 224×224, applies the
> MobileNetV2 preprocessing (pixels scaled from `[0, 255]` to `[-1, 1]`),
> and writes the result as a `(1, 224, 224, 3)` float32 tensor to `input.npy`.

---

## Step 5 — Run Inference

There are two ways to run inference — pick whichever suits you best.

### Option A — `iree-run-module` CLI

Execute the compiled model using IREE's command-line runner.

```bash
iree-run-module \
  --module=mobilenet_v2.vmfb \
  --function=serve \
  --input=@input.npy
```

| Flag | Purpose |
|---|---|
| `--module=mobilenet_v2.vmfb` | Path to the compiled VM FlatBuffer. |
| `--function=serve` | Entry-point function to invoke. |
| `--input=@input.npy` | Load the input tensor from the `.npy` file (`@` prefix = read from file). |

> **Result:** prints the output tensor (1000 class logits for ImageNet) to stdout.

### Option B — Python script (`inference.py`)

If you prefer a self-contained script (no `.npy` file needed), use
`inference.py`.  It loads a local image (default: `dog.jpg`), applies
MobileNetV2 preprocessing, invokes the model through the IREE Python runtime
bindings, and prints the top-5 predicted ImageNet classes with probabilities.

```bash
python inference.py                              # dog.jpg + mobilenet_v2.vmfb
python inference.py --image cat.png              # custom image
python inference.py --model other.vmfb           # custom .vmfb
python inference.py --image cat.png --top 10     # show top-10 predictions
```

Example output (using `dog.jpg`):

```
Loaded image: dog.jpg  (original size: 1546×1213)
Running inference...

Output shape: [1, 1000]

Top-5 predictions:
Rank   Class                           Probability
--------------------------------------------------
1      Samoyed                            0.1990%
2      Arctic fox                         0.1081%
3      Pomeranian                         0.1070%
4      keeshond                           0.1015%
5      Persian cat                        0.1009%

Done.
```

> The top-1 prediction is **Samoyed** — a white fluffy dog breed that matches
> the photo.  This confirms the full pipeline (export → MLIR → compile →
> inference) works end-to-end.

#### CLI vs Python — when to use which

| | `iree-run-module` | `inference.py` |
|---|---|---|
| **Dependencies** | Only the `iree-base-runtime` pip package | Same, plus NumPy and Pillow |
| **Input** | Requires a pre-generated `.npy` file | Loads and preprocesses an image directly |
| **Output** | Raw tensor dump to stdout | Formatted: top-N class names with probabilities |
| **Best for** | Quick one-off checks, scripting in shell pipelines | Prototyping, integration into Python workflows |

### Option C — Native C runner (IREE C API)

If you want to execute inference from native code (without `iree-run-module`),
this repo includes a minimal C runner in `native_runner/runner.c`.

It uses the IREE runtime C API to:
- create runtime instance/device/session
- load `module.vmfb`
- read `input.npy` (`float32`, little-endian, C-order)
- invoke `module.serve`

#### 1) Add IREE as a submodule (runtime sources for C build)

```bash
git submodule add https://github.com/iree-org/iree.git third_party/iree

# IMPORTANT: pin IREE to the same revision as your local iree-compile tool
# to avoid VM bytecode mismatch (for this repo: module 16.0 vs runtime 17.0).
git -C third_party/iree checkout ae97779f59a81bc56f804927be57749bb22548fa

# Runtime-only build does not need heavy compiler submodules.
git -C third_party/iree submodule update --init third_party/benchmark
```

Quick sanity check (must report a compatible compiler revision):

```bash
source .venv/bin/activate
iree-compile --version
git -C third_party/iree rev-parse --short HEAD
```

#### 2) Build the native runner

```bash
cmake -S native_runner -B native_runner/build -G Ninja
cmake --build native_runner/build --target mobilenet_runner
```

#### 2b) Build a statically linked runner

```bash
cmake -S native_runner -B native_runner/build-static -G Ninja \
  -DMOBILENET_RUNNER_STATIC_LINK=ON
cmake --build native_runner/build-static --target mobilenet_runner
```

Quick check:

```bash
ldd native_runner/build-static/mobilenet_runner
# expected: "not a dynamic executable"
```
or on macOS:
```bash
otool -L native_runner/build-static/mobilenet_runner
# expected: no linked dylibs (other than system ones, which are unavoidable)
```

> macOS note: fully static executables are not supported by the Apple toolchain.
> On macOS, `MOBILENET_RUNNER_STATIC_LINK=ON` is ignored and the runner is built
> with normal system dynamic libraries.

#### 3) Compile a VMFB compatible with system dylib loading (macOS)

Run this block as a standalone command (do not append the next block).

```bash
# Command A: build VMFB for system dylib loading
iree-compile mobilenet_v2.mlirbc \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-link-embedded=false \
  --iree-llvmcpu-target-triple=arm64-apple-macosx14.0.0 \
  -o mobilenet_v2_plugin.vmfb
```

#### 4) (Optional) Dump system-dylib executable artifacts to `./dump_plugin`

This is the command used to create the `dump_plugin/` folder with
`*.ll`, `*.bc`, `*.o`, `*.s`, and `*.dylib` artifacts:

```bash
# Command B: dump executable artifacts + emit VMFB
iree-compile mobilenet_v2.mlirbc \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-link-embedded=false \
  --iree-llvmcpu-target-triple=arm64-apple-macosx14.0.0 \
  --iree-hal-dump-executable-files-to=./dump_plugin \
  -o mobilenet_v2_plugin.vmfb
```

#### 5) Run inference from C

```bash
./native_runner/build/mobilenet_runner \
  local-task \
  mobilenet_v2_plugin.vmfb \
  input.npy \
  module.serve
```

> Note: default function name is `module.serve`, so the last argument is optional.
> If you see `bytecode version mismatch`, your VMFB was compiled with a different
> IREE revision than the runtime linked in `mobilenet_runner`. Rebuild after
> pinning `third_party/iree` to `ae97779f59a81bc56f804927be57749bb22548fa`.


---

## Bonus — Inspecting MLIR at Different Levels

### Canonicalized / Readable MLIR

Run IREE's optimizer without lowering to get a cleaned-up version of the MLIR:

```bash
iree-opt \
  mobilenet_v2.mlirbc \
  -o mobilenet_v2_readable.mlir
```

> Useful for studying the high-level operations (convolutions, batch norms, etc.)
> in a more readable form.

#### `iree-ir-tool copy` vs `iree-opt` (no passes)

You might also encounter the following alternative command:

```bash
iree-ir-tool copy mobilenet_v2.mlirbc -o mobilenet_v2_readable.mlir
```

While the output looks very similar, the two tools have **different purposes**:

| | `iree-ir-tool copy` | `iree-opt` (no passes) |
|---|---|---|
| **Primary purpose** | Format conversion / round-trip | Optimization pipeline driver |
| **Transformations applied** | None (parse → print only) | None explicit, but may apply minimal normalizations |
| **Format conversion** (e.g. `.mlirbc` ↔ `.mlir`) | ✅ designed for this | Possible, but not its main goal |
| **Typical result** (textual → textual) | Faithful reproduction of the input | Practically identical, with possible minor formatting differences |

**Rule of thumb:** use `iree-ir-tool copy` when you need to convert between MLIR
serialization formats (binary bytecode ↔ textual), and `iree-opt` when you want
to (potentially) run optimization passes on the IR.

### Lowered to LLVM Dialect (Executable Sources)

Stop the compilation pipeline right before final code generation to inspect the
LLVM dialect MLIR:

```bash
iree-compile \
  mobilenet_v2.mlirbc \
  --iree-hal-target-backends=llvm-cpu \
  --compile-to=executable-sources \
  -o mobilenet_v2_llvm_dialect.mlir
```

| Flag | Purpose |
|---|---|
| `--compile-to=executable-sources` | Stops compilation after lowering to executable sources (LLVM dialect), before final binary emission. |

> This is extremely helpful for understanding how high-level tensor ops are
> decomposed into loops, memory allocations, and low-level LLVM operations.

### Dump LLVM IR Files (`.ll`) to `./dump`

To emit LLVM IR files (not the MLIR wrapper), dump executable compilation
artifacts to a directory:

```bash
# Standalone command: do not concatenate with other iree-compile invocations.
iree-compile mobilenet_v2.mlirbc \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-hal-dump-executable-files-to=./dump \
  -o /dev/null
```

### Compile Dumped LLVM IR with `llc` and `clang`

Example using one dumped file:

```bash
llc ./dump/module___linked_embedded_elf_arm_64.linked.ll -o linked.s
```

On macOS, force a Mach-O target before linking:

```bash
llc ./dump/module___linked_embedded_elf_arm_64.linked.ll \
  -mtriple=arm64-apple-macosx14.0.0 \
  -filetype=obj \
  -o linked_macho.o

clang linked_macho.o -shared -o linked.so
```

> Note: without `-mtriple=arm64-apple-macosx14.0.0`, `llc` may emit an ELF
> object (`embedded_elf_arm_64`) that Apple `ld` cannot link.

---

## File Reference

| File | Description |
|---|---|
| `export_model.py` | Downloads MobileNetV2 from Keras and exports it as a TF SavedModel with a fixed `serve` signature. |
| `signatures.py` | Loads the SavedModel and prints available signature names (sanity check). |
| `sample.py` | Preprocesses an image (default: `dog.jpg`) and saves the tensor as `input.npy`. |
| `inference.py` | Runs inference via the IREE Python runtime with top-N class name output. |
| `dog.jpg` | Sample input image (dog photo) for testing the inference pipeline. |
| `imagenet_classes.txt` | ImageNet 1000 class names used by `inference.py` for human-readable output. |
| `mobilenet_v2.mlirbc` | MLIR bytecode produced by `iree-import-tf`. |
| `mobilenet_v2_readable.mlir` | Canonicalized MLIR (via `iree-opt`). |
| `mobilenet_v2_llvm_dialect.mlir` | MLIR lowered to LLVM dialect / executable sources. |
| `mobilenet_v2_llvm.ll` | LLVM IR (if extracted manually from the pipeline). |
| `mobilenet_v2.vmfb` | Final compiled artifact for IREE runtime. |
| `native_runner/runner.c` | Minimal native IREE C API runner that loads `input.npy` and calls `module.serve`. |
| `native_runner/CMakeLists.txt` | Runtime-only CMake build for the native C runner. |
