/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "esp_main.h"

void RespondToDetection(float* sign_score, const char* kCategoryLabels[]) {
  // Find the sign with the highest score.
  float max_score = 0;
  int max_score_index = 0;
  for (int i = 0; i < 6; ++i) {
    if (sign_score[i] > max_score) {
      max_score = sign_score[i];
      max_score_index = i;
    }
  }

  // Log the detected sign.
  if (max_score > 0.5) {
    MicroPrintf("Detected sign: %s", kCategoryLabels[max_score_index]);
  } else {
    MicroPrintf("No sign detected");
  }
  MicroPrintf("abierta: %f, apuntar: %f, cero: %f, chill: %f, perro: %f, rock: %f", sign_score[0], sign_score[1], sign_score[2], sign_score[3], sign_score[4], sign_score[5]);
}
