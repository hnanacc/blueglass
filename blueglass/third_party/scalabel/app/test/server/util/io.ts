// Copyright 2025 Intel Corporation
// SPDX: Apache-2.0

import * as fs from "fs-extra"
import { State } from "../../../src/types/state"

/**
 * helper function to read sample state
 *
 * @param fileName
 */
export function readSampleState(fileName: string): State {
  return JSON.parse(fs.readFileSync(fileName, "utf8"))
}
