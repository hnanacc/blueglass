// Copyright 2025 Intel Corporation
// SPDX: Apache-2.0

/**
 * Defining the types of some general callback functions
 */
export type MaybeError = Error | undefined

/**
 * Severity of alert
 */
export enum Severity {
  SUCCESS = "success",
  ERROR = "error",
  WARNING = "warning",
  INFO = "info"
}
