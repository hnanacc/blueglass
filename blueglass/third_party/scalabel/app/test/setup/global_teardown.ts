// Copyright 2025 Intel Corporation
// SPDX: Apache-2.0

import * as child from "child_process"

module.exports = async () => {
  child.spawn("pkill", ["redis-server"])
}
