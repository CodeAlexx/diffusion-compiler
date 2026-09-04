// The machine-readable record of a training step.
//
// Two claims are checked. The record has to carry the fields the brief asks
// for, so a reader does not have to parse a log. And a phase time has to be
// ABSENT when nothing measured it -- the reference trainer's own record
// reports every phase as 0.0 except one, which names a split it never
// measured, and a zero that means "not measured" is worse than no field.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/json.hpp"
#include "dif/training/report.hpp"
#include "dif/training/session.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

Program two_layer() {
  Program program;
  const std::uint64_t rows = 4U, width = 6U;
  program.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, width}},
      {2U, DType::F32, TensorRole::Constant, {width, width}},
      {3U, DType::F32, TensorRole::Internal, {rows, width}},
      {4U, DType::F32, TensorRole::Internal, {rows, width}},
      {5U, DType::F32, TensorRole::Input, {rows, width}},
      {6U, DType::F32, TensorRole::Output, {1U}}};
  program.operations = {{1U, Opcode::Linear, {1U, 2U}, {3U}, {}},
                        {2U, Opcode::SiLU, {3U}, {4U}, {}},
                        {3U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  dif::ir::verify(program);
  return program;
}

void the_record_carries_what_a_reader_needs() {
  const auto forward = two_layer();
  const auto plan = dif::training::compile(
      forward, 6U, std::vector<std::uint32_t>{2U}, {});
  dif::training::TrainingStepResult result;
  result.loss = 0.25F;
  result.completed_steps = 7U;
  result.step_milliseconds = 1.5;
  result.telemetry.kernel_launches = 42U;
  result.telemetry.d2h_copies = 1U;
  result.telemetry.host_stream_synchronizes = 2U;

  dif::training::TrainingStepReport report;
  report.model = "krea2";
  report.backend = "cuda";
  report.plan_fingerprint = "abc123";
  report.loss = result.loss;
  report.completed_steps = result.completed_steps;
  report.step_milliseconds = result.step_milliseconds;
  report.telemetry = result.telemetry;
  report.persistent_state_bytes = 4096U;

  const auto text = report.json();
  // Every field the brief names, present and findable by a machine.
  for (const auto *key :
       {"model", "backend", "plan_fingerprint", "loss", "completed_steps",
        "step_milliseconds", "nonfinites", "accumulation_index",
        "optimizer_applied", "phases", "submitted", "memory"})
    expect(text.find(std::string("\"") + key + "\"") != std::string::npos,
           std::string("the record carries ") + key);
  // And it is real JSON, not a string that looks like it.
  const auto parsed = dif::json::parse(text);
  expect(parsed.find("model") != nullptr, "the record parses as JSON");
  const auto *submitted = parsed.find("submitted");
  expect(submitted != nullptr && submitted->find("kernel_launches") != nullptr,
         "what the runtime submitted is nested where a reader expects it");
}

// The same rule the phase times follow, applied to identity: a caller that
// did not supply a checkpoint path leaves no "checkpoint": "" behind for a
// reader to interpret.
void an_unset_identity_is_absent_not_empty() {
  dif::training::TrainingStepReport report;
  const auto bare = report.json();
  for (const auto *key : {"model", "backend", "device", "target",
                          "runtime_budget", "plan_fingerprint", "checkpoint"})
    expect(bare.find(std::string("\"") + key + "\"") == std::string::npos,
           std::string("an unsupplied ") + key + " does not appear");
  // What a step always knows is still there, so the record is never empty.
  expect(bare.find("\"loss\"") != std::string::npos,
         "what the step measured is always reported");
  report.model = "krea2";
  expect(dif::json::parse(report.json()).find("model") != nullptr,
         "and a supplied identity does appear");
}

void an_unmeasured_phase_is_absent_not_zero() {
  dif::training::TrainingStepReport report;
  report.step_milliseconds = 1.0;
  const auto without = report.json();
  expect(without.find("forward_milliseconds") == std::string::npos,
         "a phase nothing measured does not appear at all");
  expect(without.find("transfer_milliseconds") != std::string::npos,
         "transfer time is always known, so it always appears");

  report.forward_milliseconds = 0.4;
  report.backward_milliseconds = 0.5;
  const auto with = report.json();
  expect(with.find("forward_milliseconds") != std::string::npos,
         "a measured phase appears");
  expect(with.find("optimizer_milliseconds") == std::string::npos,
         "and an unmeasured one still does not");
}

// Phase attribution places an event by the operation it belongs to, so it
// cannot be fooled by an opcode that appears in more than one phase -- a
// Linear runs in the forward pass AND in the backward one.
void phases_are_attributed_by_position_not_by_opcode() {
  Program program;
  program.tensors = {{1U, DType::F32, TensorRole::Input, {2U, 2U}}};
  for (std::uint32_t id = 1U; id <= 6U; ++id)
    program.operations.push_back({id, Opcode::Linear, {1U}, {1U}, {}});

  std::vector<dif::runtime::TraceEvent> events;
  const auto add = [&](std::uint32_t operation, double ms) {
    dif::runtime::TraceEvent event;
    event.operation_id = operation;
    event.host_start_ms = 0.0;
    event.host_end_ms = ms;
    events.push_back(std::move(event));
  };
  add(1U, 1.0);   // forward   (positions 0,1)
  add(2U, 2.0);
  add(3U, 4.0);   // backward  (positions 2,3)
  add(4U, 8.0);
  add(5U, 16.0);  // optimizer (positions 4,5)
  add(6U, 32.0);
  dif::runtime::TraceEvent transfer;  // belongs to no operation
  transfer.operation_id = 0U;
  transfer.host_start_ms = 0.0;
  transfer.host_end_ms = 64.0;
  events.push_back(transfer);

  const auto times = dif::training::attribute_phases(events, program, 2U, 4U);
  expect(times.has_value(), "a traced run attributes its phases");
  if (!times)
    return;
  expect(times->forward_milliseconds == 3.0,
         "forward is the operations before the boundary");
  expect(times->backward_milliseconds == 12.0,
         "backward is the operations between the boundaries");
  expect(times->optimizer_milliseconds == 48.0,
         "the optimizer is everything after");
  expect(times->transfer_milliseconds == 64.0,
         "events belonging to no operation are transfer cost");

  // An untraced run reports nothing rather than zeros.
  expect(!dif::training::attribute_phases({}, program, 2U, 4U).has_value(),
         "an untraced run attributes nothing at all");
}

} // namespace

int main() {
  the_record_carries_what_a_reader_needs();
  an_unset_identity_is_absent_not_empty();
  an_unmeasured_phase_is_absent_not_zero();
  phases_are_attributed_by_position_not_by_opcode();
  if (failures != 0) {
    std::cerr << failures << " training report failure(s)\n";
    return 1;
  }
  std::cout << "training report tests passed\n";
  return 0;
}
