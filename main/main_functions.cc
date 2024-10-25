#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "uart_communication.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
#define BUF_SIZE 1024  // Tamaño del buffer

static int kTensorArenaSize = 176 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;
}  // namespace

void processImage() {
  // Get image from provider
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.f)) {
    MicroPrintf("Image capture failed.");
    uart_send_data("ERROR: Image capture failed");
    return;
  }

  // Run inference
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
    uart_send_data("ERROR: Inference failed");
    return;
  }

  // Get output results
  TfLiteTensor* output = interpreter->output(0);
  
  float max_score = 0.0f;
  int max_index = 0;

  // Find highest prediction
  for (int i = 0; i < kCategoryCount; ++i) {
    if (output->data.f[i] > max_score) {
      max_score = output->data.f[i];
      max_index = i;
    }
  }

  // Send prediction through UART
  char result[50];
  snprintf(result, sizeof(result), "PRED:%s:%.2f", kCategoryLabels[max_index], max_score);
  uart_send_data(result);
}

void setup() {
  // Initialize UART
  uart_init();

  // Initialize PSRAM
  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Initialize model
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Allocate tensor arena from PSRAM
  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  printf("Free heap size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Free PSRAM size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Create op resolver
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  
  micro_op_resolver.AddDequantize();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif

  uart_send_data("ESP32-CAM Ready!");
}

#ifndef CLI_ONLY_INFERENCE
void loop() {
    char received_command[BUF_SIZE] = {0};  // Declaración del buffer

    // Llama a uart_receive_data() y pasa el buffer
    uart_receive_data(received_command);

    // Verifica si se recibió un comando y si es "Sacar Foto"
    if (strlen(received_command) > 0) {
        // Limpia el buffer de entrada de UART
        received_command[strcspn(received_command, "\r\n")] = 0;  // Elimina caracteres de nueva línea

        // Si el comando es "Sacar Foto", llama a processImage()
        if (strcmp(received_command, "Sacar Foto") == 0) {
            processImage();
        } else {
            // En caso de que se reciba otro comando
            uart_send_data("Comando desconocido.");
        }
    }

    // Pausa entre chequeos para no saturar el CPU
    vTaskDelay(pdMS_TO_TICKS(2000));  // 2 segundos entre capturas
}
#endif
