#include "dif/training/autodiff.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <set>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace dif::training {

namespace {

// Attribute sets a gradient operation has to carry to reproduce the
// forward's geometry exactly.
std::vector<ir::Attribute>
group_norm_attributes(const ir::Operation &operation) {
  return {ir::Attribute::u64(ir::AttrKey::Groups,
                             operation.u64(ir::AttrKey::Groups, 1U)),
          ir::Attribute::f64(ir::AttrKey::Epsilon,
                             operation.f64(ir::AttrKey::Epsilon, 1.0e-5))};
}

std::vector<ir::Attribute> conv2d_attributes(const ir::Operation &operation) {
  const std::array<ir::AttrKey, 9> keys{
      ir::AttrKey::StrideH,   ir::AttrKey::StrideW, ir::AttrKey::DilationH,
      ir::AttrKey::DilationW, ir::AttrKey::PadTop,  ir::AttrKey::PadBottom,
      ir::AttrKey::PadWest,   ir::AttrKey::PadEast, ir::AttrKey::Groups};
  const std::array<std::uint64_t, 9> defaults{1U, 1U, 1U, 1U, 0U,
                                              0U, 0U, 0U, 1U};
  std::vector<ir::Attribute> attributes;
  for (std::size_t index = 0U; index < keys.size(); ++index)
    attributes.push_back(ir::Attribute::u64(
        keys[index], operation.u64(keys[index], defaults[index])));
  return attributes;
}

std::vector<ir::Attribute> conv3d_attributes(const ir::Operation &operation) {
  const std::array<ir::AttrKey, 13> keys{
      ir::AttrKey::StrideT,    ir::AttrKey::StrideH,   ir::AttrKey::StrideW,
      ir::AttrKey::DilationT,  ir::AttrKey::DilationH, ir::AttrKey::DilationW,
      ir::AttrKey::PadFront,   ir::AttrKey::PadBack,   ir::AttrKey::PadTop,
      ir::AttrKey::PadBottom,  ir::AttrKey::PadWest,   ir::AttrKey::PadEast,
      ir::AttrKey::Groups};
  const std::array<std::uint64_t, 13> defaults{1U, 1U, 1U, 1U, 1U, 1U, 0U,
                                               0U, 0U, 0U, 0U, 0U, 1U};
  std::vector<ir::Attribute> attributes;
  for (std::size_t index = 0U; index < keys.size(); ++index)
    attributes.push_back(ir::Attribute::u64(
        keys[index], operation.u64(keys[index], defaults[index])));
  return attributes;
}

} // namespace


AutodiffResult differentiate(const ir::Program &forward,
                             std::uint32_t loss_tensor,
                             std::span<const std::uint32_t> with_respect_to) {
  ir::verify(forward);
  const auto *loss = forward.tensor(loss_tensor);
  if (!loss || loss->dtype != ir::DType::F32 ||
      loss->dims != std::vector<std::uint64_t>{1U})
    fail("autodiff loss must be an available F32[1] tensor");
  if (with_respect_to.empty())
    fail("autodiff requires at least one differentiation target");
  for (const auto tensor : with_respect_to) {
    const auto *description = forward.tensor(tensor);
    // Targets may live in any supported float storage dtype (flame
    // BF16_GRAD_DECISION Option A: a gradient tensor carries the dtype of
    // its forward tensor; kernels accumulate in F32 internally).  The loss
    // itself stays F32[1].
    if (!description || (description->dtype != ir::DType::F32 &&
                         description->dtype != ir::DType::BF16 &&
                         description->dtype != ir::DType::F16))
      fail("autodiff target must be a floating tensor");
  }

  const std::unordered_set<std::uint32_t> requested(with_respect_to.begin(),
                                                    with_respect_to.end());
  std::unordered_set<std::uint32_t> produced;
  for (const auto &operation : forward.operations)
    for (const auto output : operation.outputs)
      produced.insert(output);

  // Every active opcode the reverse sweep has no rule for, collected so a
  // port learns the whole list at once.
  std::set<std::string> missing;

  AutodiffResult result;
  result.program = forward;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : result.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : result.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  auto add_tensor = [&](const ir::TensorDesc &primal) {
    const auto id = next_tensor++;
    result.program.tensors.push_back(
        {id, primal.dtype, ir::TensorRole::Internal, primal.dims});
    return id;
  };
  auto add_operation = [&](ir::Opcode opcode,
                           std::vector<std::uint32_t> inputs,
                           std::vector<std::uint32_t> outputs,
                           std::vector<ir::Attribute> attributes = {}) {
    result.program.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs),
         std::move(attributes)});
  };

  // Multi-use gradients: every consumer's contribution is recorded with the
  // consumer's operation id and folded only when the gradient is read, in
  // ascending consumer-id order. The fold order is then a property of the
  // graph, not of the position of the consumers in the operation list, so
  // two topological orderings of the same forward program produce
  // bit-identical gradients (a Flame-era lesson: arrival-order folds
  // mismatched 1.3M of 4.8M bf16 elements across rewrites).
  struct Contribution {
    std::uint32_t consumer;
    std::uint32_t tensor;
  };
  std::unordered_map<std::uint32_t, std::vector<Contribution>> pending;
  std::unordered_map<std::uint32_t, std::uint32_t> gradients;
  const auto seed = add_tensor(*loss);
  add_operation(ir::Opcode::Fill, {}, {seed},
                {ir::Attribute::f64(ir::AttrKey::Value, 1.0)});
  gradients.emplace(loss_tensor, seed);

  std::uint32_t current_consumer = 0U;
  auto accumulate = [&](std::uint32_t primal, std::uint32_t contribution) {
    pending[primal].push_back({current_consumer, contribution});
  };
  // Folds a tensor's pending contributions (sorted by consumer id) into its
  // gradient; returns 0 when nothing contributes to it.
  auto resolve = [&](std::uint32_t primal) -> std::uint32_t {
    auto waiting = pending.find(primal);
    if (waiting != pending.end() && !waiting->second.empty()) {
      auto contributions = std::move(waiting->second);
      pending.erase(waiting);
      std::sort(contributions.begin(), contributions.end(),
                [](const Contribution &a, const Contribution &b) {
                  return a.consumer < b.consumer ||
                         (a.consumer == b.consumer && a.tensor < b.tensor);
                });
      auto running = gradients.find(primal);
      std::uint32_t total =
          running == gradients.end() ? 0U : running->second;
      // add_tensor can reallocate the program's tensor vector during the
      // fold. Keep the descriptor independent of that storage.
      const auto description = *result.program.tensor(primal);
      for (const auto &contribution : contributions) {
        if (total == 0U) {
          total = contribution.tensor;
          continue;
        }
        const auto sum = add_tensor(description);
        add_operation(ir::Opcode::Add, {total, contribution.tensor}, {sum});
        total = sum;
      }
      gradients[primal] = total;
      return total;
    }
    const auto found = gradients.find(primal);
    return found == gradients.end() ? 0U : found->second;
  };

  for (auto iterator = forward.operations.rbegin();
       iterator != forward.operations.rend(); ++iterator) {
    const auto &operation = *iterator;
    current_consumer = operation.id;
    // Every output's gradient, in output order, zero where that output is
    // not on a path to the loss. A single-output rule reads the first entry
    // and cannot see the difference; an operation that produces several
    // values gets all of them, which is the only way to differentiate one.
    std::vector<std::uint32_t> grad_outputs;
    grad_outputs.reserve(operation.outputs.size());
    bool active = false;
    for (const auto output : operation.outputs) {
      const auto resolved = resolve(output);
      grad_outputs.push_back(resolved);
      active = active || resolved != 0U;
    }
    if (!active)
      continue;
    const auto grad_output = grad_outputs.front();

    switch (operation.opcode) {
    case ir::Opcode::MseLoss: {
      const auto grad_prediction =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::MseLossBackward,
                    {operation.inputs[0], operation.inputs[1], grad_output},
                    {grad_prediction});
      accumulate(operation.inputs[0], grad_prediction);
      break;
    }
    case ir::Opcode::BiasAdd: {
      accumulate(operation.inputs[0], grad_output);
      const auto grad_bias =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      add_operation(ir::Opcode::BiasBackward, {grad_output}, {grad_bias});
      accumulate(operation.inputs[1], grad_bias);
      break;
    }
    case ir::Opcode::Linear: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::LinearBackwardInput,
                    {grad_output, operation.inputs[1]}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      // Frozen-weight economy (flame lesson): the gradient of a leaf weight
      // is a reverse-mode sink — nothing else consumes it.  Emit
      // LinearBackwardWeight only when the weight is a differentiation
      // target, or when the weight is produced by another operation (then
      // its gradient is the path to earlier primals).  Graphs that request
      // every parameter gradient are emitted unchanged.
      const auto weight = operation.inputs[1];
      if (requested.contains(weight) || produced.contains(weight)) {
        const auto grad_weight =
            add_tensor(*result.program.tensor(weight));
        add_operation(ir::Opcode::LinearBackwardWeight,
                      {grad_output, operation.inputs[0]}, {grad_weight});
        accumulate(weight, grad_weight);
      }
      if (operation.inputs.size() == 3U) {
        const auto grad_bias =
            add_tensor(*result.program.tensor(operation.inputs[2]));
        add_operation(ir::Opcode::BiasBackward, {grad_output}, {grad_bias});
        accumulate(operation.inputs[2], grad_bias);
      }
      break;
    }
    case ir::Opcode::LinearInt8WeightScaled: {
      // A weight-only-quantized linear is frozen by construction: the
      // verifier requires its weight and scales to be constants. So there is
      // exactly one gradient to produce, and it reads the INT8 weight
      // directly. Dequantizing it here would put back, once per linear per
      // step, precisely the cost the resident format removed.
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::LinearInt8WeightScaledBackwardInput,
                    {grad_output, operation.inputs[1], operation.inputs[2]},
                    {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::SiLU: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::SiLUBackward,
                    {operation.inputs[0], grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Add:
      accumulate(operation.inputs[0], grad_output);
      accumulate(operation.inputs[1], grad_output);
      break;
    case ir::Opcode::Multiply: {
      const auto grad_a =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      const auto grad_b =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      add_operation(ir::Opcode::Multiply,
                    {grad_output, operation.inputs[1]}, {grad_a});
      add_operation(ir::Opcode::Multiply,
                    {grad_output, operation.inputs[0]}, {grad_b});
      accumulate(operation.inputs[0], grad_a);
      accumulate(operation.inputs[1], grad_b);
      break;
    }
    case ir::Opcode::RmsNorm: {
      const auto weight = operation.inputs[1];
      // Frozen-weight economy (mirrors Linear): the weight gradient is
      // emitted only when the weight is a differentiation target or is
      // produced by another operation.
      const bool needs_weight =
          requested.contains(weight) || produced.contains(weight);
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      std::vector<std::uint32_t> outputs{grad_input};
      std::uint32_t grad_weight = 0U;
      if (needs_weight) {
        grad_weight = add_tensor(*result.program.tensor(weight));
        outputs.push_back(grad_weight);
      }
      add_operation(ir::Opcode::RmsNormBackward,
                    {grad_output, operation.inputs[0], weight},
                    std::move(outputs),
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(operation.inputs[0], grad_input);
      if (needs_weight)
        accumulate(weight, grad_weight);
      break;
    }
    case ir::Opcode::RmsNormModulate: {
      const auto layout = static_cast<ir::ModulationLayout>(operation.u64(
          ir::AttrKey::ModulationLayout,
          static_cast<std::uint64_t>(
              ir::ModulationLayout::ExplicitScaleShift)));
      if (layout == ir::ModulationLayout::SharedVectorDelta) {
        // x, weight, vector, delta -- and all four learn. The vector feeds
        // both the scale and the shift, so one backward operation produces
        // every gradient rather than the reductions each recomputing the row
        // statistic they share.
        const auto x = operation.inputs[0];
        const auto weight = operation.inputs[1];
        const auto vector = operation.inputs[2];
        const auto delta = operation.inputs[3];
        const auto grad_input = add_tensor(*result.program.tensor(x));
        const auto grad_weight = add_tensor(*result.program.tensor(weight));
        const auto grad_vector = add_tensor(*result.program.tensor(vector));
        const auto grad_delta = add_tensor(*result.program.tensor(delta));
        add_operation(
            ir::Opcode::RmsNormModulateBackward,
            {grad_output, x, weight, vector, delta},
            {grad_input, grad_weight, grad_vector, grad_delta},
            {ir::Attribute::u64(
                 ir::AttrKey::ModulationLayout,
                 static_cast<std::uint64_t>(
                     ir::ModulationLayout::SharedVectorDelta)),
             ir::Attribute::f64(ir::AttrKey::Epsilon,
                                operation.f64(ir::AttrKey::Epsilon, 1.0e-5)),
             ir::Attribute::f64(
                 ir::AttrKey::WeightOffset,
                 operation.f64(ir::AttrKey::WeightOffset, 0.0))});
        accumulate(x, grad_input);
        accumulate(weight, grad_weight);
        accumulate(vector, grad_vector);
        accumulate(delta, grad_delta);
        break;
      }
      const bool weighted = operation.inputs.size() == 4U;
      const auto x = operation.inputs[0];
      const auto scale = operation.inputs[weighted ? 2U : 1U];
      const auto shift = operation.inputs[weighted ? 3U : 2U];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      const auto grad_scale = add_tensor(*result.program.tensor(scale));
      const auto grad_shift = add_tensor(*result.program.tensor(shift));
      std::vector<std::uint32_t> inputs{grad_output, x};
      std::vector<std::uint32_t> outputs{grad_input, grad_scale, grad_shift};
      std::uint32_t grad_weight = 0U;
      if (weighted) {
        inputs.push_back(operation.inputs[1]);
        grad_weight =
            add_tensor(*result.program.tensor(operation.inputs[1]));
      }
      inputs.push_back(scale);
      if (weighted)
        outputs.push_back(grad_weight);
      add_operation(ir::Opcode::RmsNormModulateBackward, std::move(inputs),
                    std::move(outputs),
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(x, grad_input);
      accumulate(scale, grad_scale);
      accumulate(shift, grad_shift);
      if (weighted)
        accumulate(operation.inputs[1], grad_weight);
      break;
    }
    case ir::Opcode::Attention: {
      const auto q = operation.inputs[0];
      const auto k = operation.inputs[1];
      const auto v = operation.inputs[2];
      const auto *q_description = result.program.tensor(q);
      // [S,H,D] and [B,S,H,D] differ only in a leading axis, so every axis is
      // named from the end and one rule serves both.
      const auto head_axis = q_description->dims.size() - 2U;
      const auto head_dim = q_description->dims.back();
      const auto scale = operation.f64(
          ir::AttrKey::AttentionScale,
          1.0 / std::sqrt(static_cast<double>(head_dim)));
      const auto causal = operation.boolean(ir::AttrKey::Causal, false);
      // GQA: mirror the forward op's KvHeads onto the backward chain so
      // forward and backward can never resolve different groupings.  The
      // attribute is attached ONLY when the forward op carries it: a
      // KvHeads-absent program must differentiate to byte-identical IR
      // (fingerprint stability for every pre-GQA program).
      const bool grouped = operation.find(ir::AttrKey::KvHeads) != nullptr;
      const auto kv_heads = operation.u64(ir::AttrKey::KvHeads,
                                          q_description->dims[head_axis]);
      // Saved-stats recompute path: one AttentionLse op recomputes the
      // per-(query,head) F32 logsumexp, then AttentionBackward recomputes P
      // from Q,K,lse and consumes the forward output BY DIRECT TENSOR ID
      // (operation.outputs[0]) for delta = rowsum(dO*O) — flame's saved-O
      // identity lesson.  AttentionScale and Causal are stamped explicitly
      // so forward and backward can never resolve different defaults.
      const auto lse = next_tensor++;
      auto lse_dims = q_description->dims;
      lse_dims.pop_back();
      result.program.tensors.push_back(
          {lse, ir::DType::F32, ir::TensorRole::Internal, std::move(lse_dims)});
      std::vector<ir::Attribute> lse_attributes{
          ir::Attribute::f64(ir::AttrKey::AttentionScale, scale),
          ir::Attribute::boolean(ir::AttrKey::Causal, causal)};
      if (grouped)
        lse_attributes.push_back(
            ir::Attribute::u64(ir::AttrKey::KvHeads, kv_heads));
      add_operation(ir::Opcode::AttentionLse, {q, k}, {lse},
                    lse_attributes);
      const auto grad_q = add_tensor(*result.program.tensor(q));
      const auto grad_k = add_tensor(*result.program.tensor(k));
      const auto grad_v = add_tensor(*result.program.tensor(v));
      // Carry the forward's chosen implementation onto the backward, the way
      // the rope backward below already does. Without this the backward
      // resolves the default and runs the generated kernel while the forward
      // runs cuDNN -- the same mathematics through two backends, one of them
      // recomputing every score once per head-dimension element. The
      // attribute is attached ONLY when the forward carries it, so a program
      // that never chose an implementation still differentiates to
      // byte-identical IR.
      auto backward_attributes = lse_attributes;
      if (const auto *carried = operation.find(ir::AttrKey::Implementation))
        backward_attributes.push_back(*carried);
      add_operation(ir::Opcode::AttentionBackward,
                    {grad_output, q, k, v, operation.outputs[0], lse},
                    {grad_q, grad_k, grad_v},
                    std::move(backward_attributes));
      accumulate(q, grad_q);
      accumulate(k, grad_k);
      accumulate(v, grad_v);
      break;
    }
    case ir::Opcode::QkNormPartialRope: {
      const auto x = operation.inputs[0];
      const auto weight = operation.inputs[1];
      // cos/sin are precomputed non-differentiable tables; the rotation
      // layout travels on the op as an explicit RotaryDim (stamped with the
      // executor's default so forward and backward can never disagree).
      const bool needs_weight =
          requested.contains(weight) || produced.contains(weight);
      const auto grad_input = add_tensor(*result.program.tensor(x));
      std::vector<std::uint32_t> outputs{grad_input};
      std::uint32_t grad_weight = 0U;
      if (needs_weight) {
        grad_weight = add_tensor(*result.program.tensor(weight));
        outputs.push_back(grad_weight);
      }
      const auto head_dim = result.program.tensor(x)->dims.back();
      // Everything that decides HOW the forward rotated travels onto the
      // backward: the layout, the implementation that selects it, and the
      // offset into the rotation table.  A backward that falls back to a half
      // split after an interleaved forward is wrong in a way only a gradient
      // check catches.  Each one is attached ONLY when the forward carries
      // it, so a program that never mentioned them differentiates to
      // byte-identical IR -- the same fingerprint-stability rule the GQA
      // attribute follows.
      std::vector<ir::Attribute> rope_attributes{
          ir::Attribute::f64(ir::AttrKey::Epsilon,
                             operation.f64(ir::AttrKey::Epsilon, 1.0e-5)),
          ir::Attribute::u64(ir::AttrKey::RotaryDim,
                             operation.u64(ir::AttrKey::RotaryDim, head_dim))};
      for (const auto key : {ir::AttrKey::RotaryLayout,
                             ir::AttrKey::Implementation, ir::AttrKey::Start})
        if (const auto *carried = operation.find(key))
          rope_attributes.push_back(*carried);
      add_operation(
          ir::Opcode::QkNormPartialRopeBackward,
          {grad_output, x, weight, operation.inputs[2],
           operation.inputs[3]},
          std::move(outputs),
          std::move(rope_attributes));
      accumulate(x, grad_input);
      if (needs_weight)
        accumulate(weight, grad_weight);
      break;
    }
    case ir::Opcode::LayerNorm: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      const auto grad_weight =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      const auto grad_bias =
          add_tensor(*result.program.tensor(operation.inputs[2]));
      add_operation(ir::Opcode::LayerNormBackward,
                    {grad_output, operation.inputs[0], operation.inputs[1]},
                    {grad_input, grad_weight, grad_bias},
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(operation.inputs[0], grad_input);
      accumulate(operation.inputs[1], grad_weight);
      accumulate(operation.inputs[2], grad_bias);
      break;
    }
    case ir::Opcode::SwiGlu: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      std::vector<ir::Attribute> attributes{ir::Attribute::boolean(
          ir::AttrKey::GateFirst,
          operation.boolean(ir::AttrKey::GateFirst, false))};
      if (operation.find(ir::AttrKey::Start))
        attributes.push_back(ir::Attribute::u64(
            ir::AttrKey::Start, operation.u64(ir::AttrKey::Start, 0U)));
      add_operation(ir::Opcode::SwiGluBackward,
                    {grad_output, operation.inputs[0]}, {grad_input},
                    std::move(attributes));
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::ResidualGate: {
      // d_residual = g (direct accumulation); one kernel produces the branch
      // and gate gradients.  DiffIR's ResidualGate is fully elementwise
      // (gate has the residual's shape), so d_gate = g*branch elementwise —
      // flame's sum-over-sequence applies only to its broadcast [B,1,C]
      // gate, which DiffIR expresses with explicitly expanded tensors.
      accumulate(operation.inputs[0], grad_output);
      const auto grad_branch =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      const auto grad_gate =
          add_tensor(*result.program.tensor(operation.inputs[2]));
      add_operation(ir::Opcode::ResidualGateBackward,
                    {grad_output, operation.inputs[1], operation.inputs[2]},
                    {grad_branch, grad_gate});
      accumulate(operation.inputs[1], grad_branch);
      accumulate(operation.inputs[2], grad_gate);
      break;
    }
    case ir::Opcode::Gelu: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      // The backward carries the same approximation attribute, so it
      // differentiates the closed form the forward actually evaluated.
      add_operation(ir::Opcode::GeluBackward,
                    {operation.inputs[0], grad_output}, {grad_input},
                    {ir::Attribute::u64(
                        ir::AttrKey::Approximation,
                        operation.u64(ir::AttrKey::Approximation, 0U))});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::UpsampleNearest2d: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::UpsampleNearest2dBackward, {grad_output},
                    {grad_input},
                    {ir::Attribute::u64(ir::AttrKey::ScaleH,
                                        operation.u64(ir::AttrKey::ScaleH, 1U)),
                     ir::Attribute::u64(ir::AttrKey::ScaleW,
                                        operation.u64(ir::AttrKey::ScaleW, 1U))});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Slice: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::SliceBackward, {grad_output}, {grad_input},
                    {ir::Attribute::u64(ir::AttrKey::Axis,
                                        operation.u64(ir::AttrKey::Axis, 0U)),
                     ir::Attribute::u64(ir::AttrKey::Start,
                                        operation.u64(ir::AttrKey::Start, 0U))});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::BroadcastTo: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::BroadcastToBackward, {grad_output},
                    {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Reshape: {
      // A reshape moves no values, so its gradient is the same gradient in
      // the source shape.
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::Reshape, {grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Permute: {
      const auto &source = *result.program.tensor(operation.inputs[0]);
      const auto grad_input = add_tensor(source);
      // The gradient permutes back: position i of the inverse is where the
      // forward sent axis i.
      constexpr std::array<ir::AttrKey, 8> keys{
          ir::AttrKey::Permutation0, ir::AttrKey::Permutation1,
          ir::AttrKey::Permutation2, ir::AttrKey::Permutation3,
          ir::AttrKey::Permutation4, ir::AttrKey::Permutation5,
          ir::AttrKey::Permutation6, ir::AttrKey::Permutation7};
      const auto rank = source.dims.size();
      std::vector<std::uint64_t> inverse(rank, 0U);
      for (std::size_t axis = 0U; axis < rank; ++axis)
        inverse[operation.u64(keys[axis], axis)] = axis;
      std::vector<ir::Attribute> attributes;
      for (std::size_t axis = 0U; axis < rank; ++axis)
        attributes.push_back(ir::Attribute::u64(keys[axis], inverse[axis]));
      add_operation(ir::Opcode::Permute, {grad_output}, {grad_input},
                    std::move(attributes));
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Concat: {
      // Each input takes the slice of the gradient it contributed.
      const auto axis =
          static_cast<std::size_t>(operation.u64(ir::AttrKey::Axis, 0U));
      std::uint64_t offset = 0U;
      for (const auto input : operation.inputs) {
        const auto &description = *result.program.tensor(input);
        const auto grad_input = add_tensor(description);
        add_operation(ir::Opcode::Slice, {grad_output}, {grad_input},
                      {ir::Attribute::u64(ir::AttrKey::Axis, axis),
                       ir::Attribute::u64(ir::AttrKey::Start, offset)});
        accumulate(input, grad_input);
        offset += description.dims[axis];
      }
      break;
    }
    case ir::Opcode::GroupNorm: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::GroupNormBackward,
                    {operation.inputs[0], operation.inputs[1], grad_output},
                    {grad_input}, group_norm_attributes(operation));
      accumulate(operation.inputs[0], grad_input);
      // The affine gradients reduce per channel across the batch, so they
      // are their own operation, and like every leaf weight they are only
      // emitted when something asks for them.
      const auto weight = operation.inputs[1];
      const auto bias = operation.inputs[2];
      const bool wanted = requested.contains(weight) ||
                          produced.contains(weight) ||
                          requested.contains(bias) || produced.contains(bias);
      if (wanted) {
        const auto grad_weight = add_tensor(*result.program.tensor(weight));
        const auto grad_bias = add_tensor(*result.program.tensor(bias));
        add_operation(ir::Opcode::GroupNormBackwardAffine,
                      {operation.inputs[0], grad_output},
                      {grad_weight, grad_bias},
                      group_norm_attributes(operation));
        accumulate(weight, grad_weight);
        accumulate(bias, grad_bias);
      }
      break;
    }
    case ir::Opcode::Conv2d: {
      auto attributes = conv2d_attributes(operation);
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::Conv2dBackwardInput,
                    {grad_output, operation.inputs[1]}, {grad_input},
                    attributes);
      accumulate(operation.inputs[0], grad_input);
      const auto weight = operation.inputs[1];
      if (requested.contains(weight) || produced.contains(weight)) {
        const auto grad_weight = add_tensor(*result.program.tensor(weight));
        add_operation(ir::Opcode::Conv2dBackwardWeight,
                      {grad_output, operation.inputs[0]}, {grad_weight},
                      attributes);
        accumulate(weight, grad_weight);
      }
      if (operation.inputs.size() == 3U) {
        const auto grad_bias =
            add_tensor(*result.program.tensor(operation.inputs[2]));
        add_operation(ir::Opcode::Conv2dBackwardBias, {grad_output},
                      {grad_bias});
        accumulate(operation.inputs[2], grad_bias);
      }
      break;
    }
    case ir::Opcode::RotaryApply: {
      // The cos/sin tables are position tables, not learnable values; only
      // the rotated activation carries a gradient. The pairing convention is
      // stamped from the forward so the two cannot unrotate different ways.
      const auto x = operation.inputs[0];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      std::vector<ir::Attribute> attributes;
      if (const auto *layout = operation.find(ir::AttrKey::RotaryLayout))
        attributes.push_back(*layout);
      add_operation(ir::Opcode::RotaryApplyBackward,
                    {grad_output, operation.inputs[1], operation.inputs[2]},
                    {grad_input}, std::move(attributes));
      accumulate(x, grad_input);
      break;
    }
    case ir::Opcode::SelectRowChunks: {
      // The gradients arrive one per chunk, in the forward's output order.
      // A chunk whose output never reached the loss contributes nothing, and
      // the backward needs a tensor for it anyway, so an explicit zero is
      // filled rather than the shapes being allowed to disagree.
      const auto values = operation.inputs[0];
      std::vector<std::uint32_t> inputs{operation.inputs[1]};
      for (std::size_t index = 0U; index < operation.outputs.size(); ++index) {
        auto gradient = grad_outputs[index];
        if (gradient == 0U) {
          gradient = add_tensor(*result.program.tensor(operation.outputs[index]));
          add_operation(ir::Opcode::Fill, {}, {gradient},
                        {ir::Attribute::f64(ir::AttrKey::Value, 0.0)});
        }
        inputs.push_back(gradient);
      }
      const auto grad_values = add_tensor(*result.program.tensor(values));
      add_operation(ir::Opcode::SelectRowChunksBackward, std::move(inputs),
                    {grad_values});
      accumulate(values, grad_values);
      break;
    }
    case ir::Opcode::IndexedUpdateRows: {
      const auto base = operation.inputs[0];
      const auto updates = operation.inputs[1];
      const auto grad_base = add_tensor(*result.program.tensor(base));
      std::vector<std::uint32_t> outputs{grad_base};
      std::uint32_t grad_updates = 0U;
      const bool needs_updates =
          requested.contains(updates) || produced.contains(updates);
      if (needs_updates) {
        grad_updates = add_tensor(*result.program.tensor(updates));
        outputs.push_back(grad_updates);
      }
      add_operation(ir::Opcode::IndexedUpdateRowsBackward,
                    {grad_output, operation.inputs[2]}, std::move(outputs));
      accumulate(base, grad_base);
      if (needs_updates)
        accumulate(updates, grad_updates);
      break;
    }
    case ir::Opcode::H3DeinterleaveQkvWeight: {
      // The forward is a permutation, so the gradient is its inverse. Any
      // component whose output did not reach the loss contributes zeros.
      const auto packed = operation.inputs[0];
      std::vector<std::uint32_t> inputs;
      for (std::size_t index = 0U; index < 3U; ++index) {
        auto gradient = grad_outputs[index];
        if (gradient == 0U) {
          gradient = add_tensor(*result.program.tensor(operation.outputs[index]));
          add_operation(ir::Opcode::Fill, {}, {gradient},
                        {ir::Attribute::f64(ir::AttrKey::Value, 0.0)});
        }
        inputs.push_back(gradient);
      }
      const auto grad_packed = add_tensor(*result.program.tensor(packed));
      add_operation(ir::Opcode::H3InterleaveQkvWeight, std::move(inputs),
                    {grad_packed},
                    {ir::Attribute::u64(ir::AttrKey::Heads,
                                        operation.u64(ir::AttrKey::Heads, 0U)),
                     ir::Attribute::u64(
                         ir::AttrKey::HeadDim,
                         operation.u64(ir::AttrKey::HeadDim, 0U))});
      accumulate(packed, grad_packed);
      break;
    }
    case ir::Opcode::H3AdaLNSelect: {
      const auto projected = operation.inputs[0];
      std::vector<std::uint32_t> inputs{operation.inputs[1]};
      for (std::size_t index = 0U; index < 6U; ++index) {
        auto gradient = grad_outputs[index];
        if (gradient == 0U) {
          gradient = add_tensor(*result.program.tensor(operation.outputs[index]));
          add_operation(ir::Opcode::Fill, {}, {gradient},
                        {ir::Attribute::f64(ir::AttrKey::Value, 0.0)});
        }
        inputs.push_back(gradient);
      }
      const auto grad_projected = add_tensor(*result.program.tensor(projected));
      add_operation(ir::Opcode::H3AdaLNSelectBackward, std::move(inputs),
                    {grad_projected});
      accumulate(projected, grad_projected);
      break;
    }
    case ir::Opcode::PadReflect: {
      // Every pad extent is stamped from the forward, because the gradient
      // has to fold back exactly the reflections the forward made.
      const auto x = operation.inputs[0];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      std::vector<ir::Attribute> attributes;
      for (const auto key :
           {ir::AttrKey::PadFront, ir::AttrKey::PadBack, ir::AttrKey::PadTop,
            ir::AttrKey::PadBottom, ir::AttrKey::PadWest,
            ir::AttrKey::PadEast})
        attributes.push_back(
            ir::Attribute::u64(key, operation.u64(key, 0U)));
      add_operation(ir::Opcode::PadReflectBackward, {grad_output},
                    {grad_input}, std::move(attributes));
      accumulate(x, grad_input);
      break;
    }
    case ir::Opcode::ChannelRmsNorm: {
      const auto x = operation.inputs[0];
      const auto gamma = operation.inputs[1];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      std::vector<std::uint32_t> outputs{grad_input};
      std::uint32_t grad_gamma = 0U;
      const bool needs_gamma =
          requested.contains(gamma) || produced.contains(gamma);
      if (needs_gamma) {
        grad_gamma = add_tensor(*result.program.tensor(gamma));
        outputs.push_back(grad_gamma);
      }
      // Axis and epsilon are stamped from the forward's own values: the
      // gradient's clamp branch has to be the branch the forward took.
      add_operation(ir::Opcode::ChannelRmsNormBackward,
                    {grad_output, x, gamma}, std::move(outputs),
                    {ir::Attribute::u64(ir::AttrKey::Axis,
                                        operation.u64(ir::AttrKey::Axis, 1U)),
                     ir::Attribute::f64(
                         ir::AttrKey::Epsilon,
                         operation.f64(ir::AttrKey::Epsilon, 1.0e-12))});
      accumulate(x, grad_input);
      if (needs_gamma)
        accumulate(gamma, grad_gamma);
      break;
    }
    case ir::Opcode::PadConstant: {
      // Padding a tensor with a constant adds values the input never
      // influenced, so its gradient is the crop back to the input's own
      // region -- and a crop is a Slice, which the IR already has. One Slice
      // per padded axis, and none at all for an axis that was not padded, so
      // a pad that touches one axis costs one operation.
      const auto x = operation.inputs[0];
      const auto *description = result.program.tensor(x);
      const auto rank = description->dims.size();
      const std::array<std::pair<std::size_t, ir::AttrKey>, 3> axes{
          std::pair{rank - 3U, ir::AttrKey::PadFront},
          std::pair{rank - 2U, ir::AttrKey::PadTop},
          std::pair{rank - 1U, ir::AttrKey::PadWest}};
      auto cropped = grad_output;
      auto dims = result.program.tensor(grad_output)->dims;
      for (const auto &[axis, key] : axes) {
        // A rank-4 pad has no depth axis to crop.
        if (rank < 5U && key == ir::AttrKey::PadFront)
          continue;
        const auto low = operation.u64(key, 0U);
        if (low == 0U && dims[axis] == description->dims[axis])
          continue;
        dims[axis] = description->dims[axis];
        const auto next = next_tensor++;
        result.program.tensors.push_back(
            {next, description->dtype, ir::TensorRole::Internal, dims});
        add_operation(ir::Opcode::Slice, {cropped}, {next},
                      {ir::Attribute::u64(ir::AttrKey::Axis, axis),
                       ir::Attribute::u64(ir::AttrKey::Start, low)});
        cropped = next;
      }
      accumulate(x, cropped);
      break;
    }
    case ir::Opcode::Conv3d: {
      // The same three gradients as in two dimensions, over the geometry the
      // forward carried; every attribute is stamped explicitly so the forward
      // and its gradients can never resolve a different convolution.
      auto attributes = conv3d_attributes(operation);
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::Conv3dBackwardInput,
                    {grad_output, operation.inputs[1]}, {grad_input},
                    attributes);
      accumulate(operation.inputs[0], grad_input);
      const auto weight = operation.inputs[1];
      if (requested.contains(weight) || produced.contains(weight)) {
        const auto grad_weight = add_tensor(*result.program.tensor(weight));
        add_operation(ir::Opcode::Conv3dBackwardWeight,
                      {grad_output, operation.inputs[0]}, {grad_weight},
                      attributes);
        accumulate(weight, grad_weight);
      }
      if (operation.inputs.size() == 3U) {
        const auto grad_bias =
            add_tensor(*result.program.tensor(operation.inputs[2]));
        add_operation(ir::Opcode::Conv3dBackwardBias, {grad_output},
                      {grad_bias});
        accumulate(operation.inputs[2], grad_bias);
      }
      break;
    }
    case ir::Opcode::SinusoidalTimestep:
    case ir::Opcode::RotaryPosition:
      // A timestep and a rotary position are schedule coordinates, not
      // learnable values: reverse mode terminates here the way it does at a
      // Fill. RotaryPosition produces two tables and both terminate, which is
      // why this needs the multi-output sweep above rather than a special
      // case.
      break;
    case ir::Opcode::LayerNormModulate: {
      // One backward operation produces all five gradients, because the three
      // reductions inside it share the row statistics.  Epsilon is stamped
      // from the forward's own value so the two cannot disagree.
      const auto x = operation.inputs[0];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      const auto grad_weight =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      const auto grad_bias =
          add_tensor(*result.program.tensor(operation.inputs[2]));
      const auto grad_scale =
          add_tensor(*result.program.tensor(operation.inputs[3]));
      const auto grad_shift =
          add_tensor(*result.program.tensor(operation.inputs[4]));
      add_operation(
          ir::Opcode::LayerNormModulateBackward,
          {grad_output, x, operation.inputs[1], operation.inputs[2],
           operation.inputs[3]},
          {grad_input, grad_weight, grad_bias, grad_scale, grad_shift},
          {ir::Attribute::f64(ir::AttrKey::Epsilon,
                              operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(x, grad_input);
      accumulate(operation.inputs[1], grad_weight);
      accumulate(operation.inputs[2], grad_bias);
      accumulate(operation.inputs[3], grad_scale);
      accumulate(operation.inputs[4], grad_shift);
      break;
    }
    case ir::Opcode::GatherRows: {
      // The index vector is a selection, not a value: reverse mode passes
      // through it untouched and scatters the gathered rows' gradients back
      // into the table they came from.
      const auto table = operation.inputs[0];
      const auto grad_input = add_tensor(*result.program.tensor(table));
      add_operation(ir::Opcode::GatherRowsBackward,
                    {grad_output, operation.inputs[1]}, {grad_input});
      accumulate(table, grad_input);
      break;
    }
    case ir::Opcode::Sigmoid: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::SigmoidBackward,
                    {operation.inputs[0], grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::AffineLastDim: {
      // y = x * scale[c] + bias[c].  Every gradient here is something the IR
      // can already say, so this rule adds no opcode:
      //   dx    = g * scale[c]        -- an AffineLastDim with no bias
      //   dbias = column sum of g     -- exactly BiasBackward
      //   dscale = column sum of g*x  -- a Multiply, then BiasBackward
      const auto x = operation.inputs[0];
      const auto scale = operation.inputs[1];
      if (requested.contains(x) || produced.contains(x)) {
        const auto grad_input = add_tensor(*result.program.tensor(x));
        add_operation(ir::Opcode::AffineLastDim, {grad_output, scale},
                      {grad_input});
        accumulate(x, grad_input);
      }
      if (requested.contains(scale) || produced.contains(scale)) {
        const auto product = add_tensor(*result.program.tensor(x));
        add_operation(ir::Opcode::Multiply, {grad_output, x}, {product});
        const auto grad_scale = add_tensor(*result.program.tensor(scale));
        add_operation(ir::Opcode::BiasBackward, {product}, {grad_scale});
        accumulate(scale, grad_scale);
      }
      if (operation.inputs.size() == 3U) {
        const auto bias = operation.inputs[2];
        if (requested.contains(bias) || produced.contains(bias)) {
          const auto grad_bias = add_tensor(*result.program.tensor(bias));
          add_operation(ir::Opcode::BiasBackward, {grad_output}, {grad_bias});
          accumulate(bias, grad_bias);
        }
      }
      break;
    }
    case ir::Opcode::Clamp: {
      // The bounds travel onto the backward op explicitly, stamped with the
      // forward's own defaults, so the two can never resolve a different
      // saturation range.
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(
          ir::Opcode::ClampBackward, {grad_output, operation.inputs[0]},
          {grad_input},
          {ir::Attribute::f64(
               ir::AttrKey::Lower,
               operation.f64(ir::AttrKey::Lower,
                             -std::numeric_limits<double>::infinity())),
           ir::Attribute::f64(
               ir::AttrKey::Upper,
               operation.f64(ir::AttrKey::Upper,
                             std::numeric_limits<double>::infinity()))});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Cast: {
      // Cast is the mixed-precision boundary op.  The gradient of
      // Cast(x, dt) with upstream gradient g is Cast(g, dtype(x)):
      // add_tensor copies the primal description, so the contribution lands
      // in the source storage dtype.
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::Cast, {grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Fill:
      // A Fill has no primal inputs.  Its active output is a graph constant,
      // so reverse mode terminates at this leaf.
      break;
    default:
      // Do not stop at the first one.  Porting a model means finding out
      // everything its graph needs, and one opcode per run is a slow way to
      // learn it -- so record this and keep sweeping.  The gradients built
      // after this point are incomplete, which is why the collected list is
      // fatal below before anything can read them.
      missing.insert(std::string(ir::opcode_name(operation.opcode)));
      break;
    }
  }

  if (!missing.empty()) {
    std::string names;
    for (const auto &name : missing)
      names += (names.empty() ? "" : ", ") + name;
    fail("autodiff has no rule for " + std::to_string(missing.size()) +
         " active opcode(s): " + names);
  }

  for (const auto primal : with_respect_to) {
    const auto gradient = resolve(primal);
    if (gradient == 0U)
      fail("autodiff target " + std::to_string(primal) +
           " is disconnected from the loss: nothing it feeds reaches the "
           "value being differentiated");
    auto tensor = std::find_if(result.program.tensors.begin(),
                               result.program.tensors.end(),
                               [&](const auto &v) { return v.id == gradient; });
    tensor->roles |= ir::TensorRole::Output;
    result.gradients.emplace(primal, gradient);
  }
  ir::verify(result.program);
  return result;
}

} // namespace dif::training
