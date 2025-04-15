// Copyright 2025 Intel Corporation
// SPDX: Apache-2.0

import { MaybeError } from "../types/common"

/**
 * Add callback for the main window visibility change
 *
 * @param callback
 */
export function addVisibilityListener(
  callback: (error: MaybeError, visible: boolean) => void
): void {
  window.addEventListener("blur", () => {
    callback(undefined, false)
  })
  window.addEventListener("focus", () => {
    callback(undefined, true)
  })
  window.addEventListener("visibilitychange", () => {
    callback(undefined, !document.hidden)
  })
}
