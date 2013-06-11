#include <OpenCL/cl.h>
#include <stdio.h>

const char* kernel_source = 
  "__kernel void copy_array(__global float* input, "
  "                         __global float* output) {"
  "  int i = get_global_id(0);"
  "  output[i] = input[i];"
  "}";

#define SIZE (1024 * 1000)

#define CHECK_STATUS(s) do { \
    cl_int ss = (s); \
    if (ss != CL_SUCCESS) { \
      fprintf(stderr, "Error %d at line %d\n", ss, __LINE__); \
      exit(1); \
    } \
  } while (0)

void random_fill(float array[], size_t size) {
  for (int i = 0; i < size; ++i)
    array[i] = (float)rand() / RAND_MAX;
}

int main() {
  cl_int status;

  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  CHECK_STATUS(status);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  CHECK_STATUS(status);

  cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &status);
  CHECK_STATUS(status);

  CHECK_STATUS(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));

  cl_kernel kernel = clCreateKernel(program, "copy_array", &status);
  CHECK_STATUS(status);

  float from[SIZE];
  random_fill(from, SIZE);

  cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * SIZE, from, &status);
  CHECK_STATUS(status);

  cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    sizeof(float) * SIZE, NULL, &status);
  CHECK_STATUS(status);

  CHECK_STATUS(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input));
  CHECK_STATUS(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output));

  size_t work_units = SIZE;
  CHECK_STATUS(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units, NULL, 0, NULL, NULL));

  float to[SIZE];
  CHECK_STATUS(clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(float) * SIZE, to, 0, NULL, NULL));

  CHECK_STATUS(clReleaseMemObject(input));
  CHECK_STATUS(clReleaseMemObject(output));
  CHECK_STATUS(clReleaseKernel(kernel));
  CHECK_STATUS(clReleaseProgram(program));
  CHECK_STATUS(clReleaseCommandQueue(queue));
  CHECK_STATUS(clReleaseContext(context));

  for (int i = 0; i < SIZE; ++i)
    if (from[i] != to[i])
      fprintf(stderr, "Mismatch at %d (%f != %f)", i, from[i], to[i]);

  return 0;
}
