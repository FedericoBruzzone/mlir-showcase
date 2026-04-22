// Minimal native IREE runner for MobileNetV2.
// Loads input from a .npy float32 tensor and invokes module.serve.

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/runtime/api.h"

typedef struct npy_tensor_t {
  float* data;
  size_t byte_length;
  iree_hal_dim_t dims[8];
  iree_host_size_t rank;
} npy_tensor_t;

static void npy_tensor_deinitialize(npy_tensor_t* tensor) {
  if (!tensor) return;
  free(tensor->data);
  memset(tensor, 0, sizeof(*tensor));
}

static iree_status_t npy_parse_shape(const char* header, npy_tensor_t* out_tensor) {
  const char* shape_key = strstr(header, "'shape':");
  if (!shape_key) shape_key = strstr(header, "\"shape\":");
  if (!shape_key) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY header has no shape field");
  }

  const char* lparen = strchr(shape_key, '(');
  const char* rparen = lparen ? strchr(lparen, ')') : NULL;
  if (!lparen || !rparen || rparen <= lparen) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NPY shape tuple malformed");
  }

  out_tensor->rank = 0;
  const char* p = lparen + 1;
  while (p < rparen) {
    while (p < rparen && (isspace((unsigned char)*p) || *p == ',')) ++p;
    if (p >= rparen) break;

    char* end = NULL;
    long long dim = strtoll(p, &end, 10);
    if (end == p || dim <= 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid shape dimension in NPY header");
    }
    if (out_tensor->rank >= IREE_ARRAYSIZE(out_tensor->dims)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "rank too large for demo runner");
    }
    out_tensor->dims[out_tensor->rank++] = (iree_hal_dim_t)dim;
    p = end;
  }

  if (out_tensor->rank == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty NPY shape is not supported");
  }

  return iree_ok_status();
}

static iree_status_t load_npy_f32(const char* path, npy_tensor_t* out_tensor) {
  memset(out_tensor, 0, sizeof(*out_tensor));

  FILE* f = fopen(path, "rb");
  if (!f) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "failed to open %s", path);
  }

  iree_status_t status = iree_ok_status();
  uint8_t preamble[12] = {0};
  if (fread(preamble, 1, 10, f) != 10) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS, "short NPY preamble");
    goto done;
  }

  if (memcmp(preamble, "\x93NUMPY", 6) != 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "not a NPY file: %s", path);
    goto done;
  }

  uint8_t major = preamble[6];
  uint8_t minor = preamble[7];
  (void)minor;

  uint32_t header_len = 0;
  if (major == 1) {
    header_len = (uint32_t)preamble[8] | ((uint32_t)preamble[9] << 8);
  } else {
    if (fread(preamble + 10, 1, 2, f) != 2) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS, "short NPY header len");
      goto done;
    }
    header_len = (uint32_t)preamble[8] | ((uint32_t)preamble[9] << 8) |
                 ((uint32_t)preamble[10] << 16) | ((uint32_t)preamble[11] << 24);
  }

  char* header = (char*)malloc((size_t)header_len + 1);
  if (!header) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "alloc failed for NPY header");
    goto done;
  }
  if (fread(header, 1, header_len, f) != header_len) {
    free(header);
    status = iree_make_status(IREE_STATUS_DATA_LOSS, "short NPY header data");
    goto done;
  }
  header[header_len] = '\0';

  if (!strstr(header, "'descr': '<f4'") && !strstr(header, "\"descr\": \"<f4\"")) {
    free(header);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "only little-endian float32 NPY is supported");
    goto done;
  }
  if (strstr(header, "'fortran_order': True") ||
      strstr(header, "\"fortran_order\": true")) {
    free(header);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Fortran-ordered NPY is not supported");
    goto done;
  }

  status = npy_parse_shape(header, out_tensor);
  free(header);
  if (!iree_status_is_ok(status)) goto done;

  size_t element_count = 1;
  for (iree_host_size_t i = 0; i < out_tensor->rank; ++i) {
    element_count *= (size_t)out_tensor->dims[i];
  }
  out_tensor->byte_length = element_count * sizeof(float);
  out_tensor->data = (float*)malloc(out_tensor->byte_length);
  if (!out_tensor->data) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "alloc failed for NPY tensor data");
    goto done;
  }

  if (fread(out_tensor->data, 1, out_tensor->byte_length, f) !=
      out_tensor->byte_length) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "short NPY tensor payload");
    goto done;
  }

done:
  fclose(f);
  if (!iree_status_is_ok(status)) {
    npy_tensor_deinitialize(out_tensor);
  }
  return status;
}

static iree_status_t run_serve(iree_runtime_session_t* session,
                               iree_string_view_t function_name,
                               const npy_tensor_t* input_tensor) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, function_name, &call));

  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator = iree_runtime_session_host_allocator(session);

  iree_status_t status = iree_ok_status();

  iree_hal_buffer_view_t* input = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_allocate_buffer_copy(
        device, device_allocator, input_tensor->rank, input_tensor->dims,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .access = IREE_HAL_MEMORY_ACCESS_ALL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(input_tensor->data, input_tensor->byte_length),
        &input);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input);
  }
  iree_hal_buffer_view_release(input);

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  iree_hal_buffer_view_t* result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Inference output:\n");
    status = iree_hal_buffer_view_fprint(stdout, result,
                                         /*max_element_count=*/1000,
                                         host_allocator);
    fprintf(stdout, "\n");
  }

  iree_hal_buffer_view_release(result);
  iree_runtime_call_deinitialize(&call);
  return status;
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    fprintf(stderr,
            "usage: %s <device_uri> <module.vmfb> <input.npy> [function]\n",
            argv[0]);
    fprintf(stderr,
            "example: %s local-task mobilenet_v2_plugin.vmfb input.npy module.serve\n",
            argv[0]);
    return 1;
  }

  const char* device_uri = argv[1];
  const char* module_path = argv[2];
  const char* input_path = argv[3];
  const char* function_name_arg = (argc >= 5) ? argv[4] : "module.serve";

  npy_tensor_t input_tensor;
  iree_status_t status = load_npy_f32(input_path, &input_tensor);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  iree_runtime_instance_t* instance = NULL;
  status = iree_runtime_instance_create(&instance_options, iree_allocator_system(),
                                        &instance);

  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(device_uri), &device);
  }

  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(session,
                                                                    module_path);
  }

  if (iree_status_is_ok(status)) {
    status = run_serve(session, iree_make_cstring_view(function_name_arg),
                       &input_tensor);
  }

  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  npy_tensor_deinitialize(&input_tensor);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  return ret;
}
