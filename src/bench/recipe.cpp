#include "dif/bench/recipe.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"

#include <cstdlib>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

namespace dif::bench {
namespace {

const json::Value &required(const json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value)
    fail(std::string("benchmark recipe is missing required field '") + key +
         "'");
  return *value;
}

std::string optional_string(const json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value)
    return {};
  return value->string();
}

std::vector<std::string> string_list(const json::Value &value,
                                     const char *label) {
  if (!value.is_array())
    fail(std::string("benchmark recipe ") + label + " must be an array");
  std::vector<std::string> out;
  for (const auto &item : value.array())
    out.push_back(item.string());
  return out;
}

std::vector<std::pair<std::string, std::string>>
string_map(const json::Value &value, const char *label) {
  if (!value.is_object())
    fail(std::string("benchmark recipe ") + label + " must be an object");
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto &[key, item] : value.object())
    out.emplace_back(key, item.string());
  return out;
}

} // namespace

Recipe parse_recipe(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open benchmark recipe " + path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const auto document = json::parse(buffer.str());
  if (!document.is_object())
    fail("benchmark recipe is not a JSON object");
  if (required(document, "kind").string() != kRecipeKind)
    fail("file is not a diffusion-compiler benchmark recipe");
  if (required(document, "version").number() != kRecipeVersion)
    fail("unsupported benchmark recipe version");
  Recipe recipe;
  recipe.path = std::filesystem::absolute(path);
  recipe.name = required(document, "name").string();
  recipe.output_kind = required(document, "output_kind").string();
  if (recipe.output_kind != "image" && recipe.output_kind != "video")
    fail("benchmark recipe output_kind must be image or video");
  recipe.description = optional_string(document, "description");
  if (const auto *workload = document.find("workload")) {
    if (!workload->is_object())
      fail("benchmark recipe workload must be an object");
    recipe.workload = telemetry::from_parsed(*workload).object();
  }
  if (const auto *comparator = document.find("comparator")) {
    if (!comparator->is_object())
      fail("benchmark recipe comparator must be an object");
    recipe.comparator.present = true;
    recipe.comparator.name = required(*comparator, "name").string();
    recipe.comparator.wall_seconds =
        required(*comparator, "wall_seconds").number();
    if (const auto *ratio = comparator->find("target_ratio"))
      recipe.comparator.target_ratio = ratio->number();
    if (!(recipe.comparator.wall_seconds > 0.0))
      fail("benchmark recipe comparator wall_seconds must be positive");
  }
  if (const auto *variables = document.find("variables"))
    for (const auto &[key, value] : string_map(*variables, "variables"))
      recipe.variables.emplace(key, value);
  if (const auto *prompt = document.find("prompt")) {
    if (!prompt->is_object())
      fail("benchmark recipe prompt must be an object");
    recipe.prompt_file = optional_string(*prompt, "file");
  }
  if (const auto *inputs = document.find("inputs")) {
    if (!inputs->is_array())
      fail("benchmark recipe inputs must be an array");
    for (const auto &entry : inputs->array()) {
      RecipeInput input;
      input.name = required(entry, "name").string();
      input.path = required(entry, "path").string();
      recipe.inputs.push_back(std::move(input));
    }
  }
  if (const auto *files = document.find("model_files"))
    recipe.model_files = string_list(*files, "model_files");
  if (const auto *files = document.find("required_files"))
    recipe.required_files = string_list(*files, "required_files");
  if (const auto *tools = document.find("required_tools"))
    recipe.required_tools = string_list(*tools, "required_tools");
  if (const auto *environment = document.find("environment"))
    recipe.environment = string_map(*environment, "environment");
  const auto &stages = required(document, "stages");
  if (!stages.is_array() || stages.array().empty())
    fail("benchmark recipe stages must be a non-empty array");
  for (const auto &entry : stages.array()) {
    if (!entry.is_object())
      fail("benchmark recipe stage is not an object");
    RecipeStage stage;
    stage.name = required(entry, "name").string();
    stage.argv = string_list(required(entry, "argv"), "stage argv");
    if (stage.argv.empty())
      fail("benchmark recipe stage '" + stage.name + "' has an empty argv");
    if (const auto *after = entry.find("after"))
      stage.after = string_list(*after, "stage after");
    else if (!recipe.stages.empty())
      stage.after.push_back(recipe.stages.back().name);
    stage.cwd = optional_string(entry, "cwd");
    if (const auto *environment = entry.find("environment"))
      stage.environment = string_map(*environment, "stage environment");
    if (const auto *cache = entry.find("cache")) {
      if (!cache->is_object())
        fail("benchmark recipe stage '" + stage.name +
             "' cache must be an object with key and outputs");
      stage.cacheable = true;
      stage.cache.key = string_list(required(*cache, "key"), "stage cache key");
      stage.cache.outputs =
          string_list(required(*cache, "outputs"), "stage cache outputs");
      if (stage.cache.key.empty() || stage.cache.outputs.empty())
        fail("benchmark recipe stage '" + stage.name +
             "' cache needs at least one key file and one output");
    }
    recipe.stages.push_back(std::move(stage));
  }
  recipe.output = required(document, "output").string();
  return recipe;
}

namespace {

class Substitutor {
public:
  Substitutor(const ResolvedRecipe &resolved,
              const std::map<std::string, std::string> &variables)
      : resolved_(resolved), variables_(variables) {}

  std::string operator()(const std::string &text) const {
    std::string out;
    std::size_t index = 0U;
    while (index < text.size()) {
      const auto start = text.find("${", index);
      if (start == std::string::npos) {
        out += text.substr(index);
        break;
      }
      const auto stop = text.find('}', start);
      if (stop == std::string::npos)
        fail("unterminated ${ reference in benchmark recipe: " + text);
      out += text.substr(index, start - index);
      out += lookup(text.substr(start + 2U, stop - start - 2U));
      index = stop + 1U;
    }
    return out;
  }

private:
  std::string lookup(const std::string &name) const {
    if (name == "build")
      return resolved_.build_directory.string();
    if (name == "workdir")
      return resolved_.work_directory.string();
    if (name == "recipe_dir")
      return resolved_.recipe.path.parent_path().string();
    if (name == "prompt_file")
      return resolved_.prompt_file.string();
    if (name == "output")
      return resolved_.output.string();
    if (name.rfind("input:", 0U) == 0U) {
      const auto input_name = name.substr(6U);
      for (const auto &input : resolved_.inputs)
        if (input.name == input_name)
          return input.path;
      fail("benchmark recipe references unknown input '" + input_name + "'");
    }
    if (name.rfind("env:", 0U) == 0U) {
      const char *value = std::getenv(name.substr(4U).c_str());
      if (!value)
        fail("benchmark recipe references unset environment variable '" +
             name.substr(4U) + "'");
      return value;
    }
    const auto found = variables_.find(name);
    if (found == variables_.end())
      fail("benchmark recipe references unknown variable '" + name + "'");
    return found->second;
  }

  const ResolvedRecipe &resolved_;
  const std::map<std::string, std::string> &variables_;
};

} // namespace

ResolvedRecipe resolve_recipe(const Recipe &recipe,
                              const ResolveContext &context) {
  ResolvedRecipe resolved;
  resolved.recipe = recipe;
  resolved.build_directory = std::filesystem::absolute(context.build_directory);
  resolved.work_directory = std::filesystem::absolute(context.work_directory);
  resolved.variables = recipe.variables;
  for (const auto &[key, value] : context.overrides) {
    if (!resolved.variables.contains(key))
      fail("--set names a variable the recipe does not declare: " + key);
    resolved.variables[key] = value;
  }
  // Variables may reference the builtins and each other (single pass, in
  // key order), which keeps a checkpoint root reusable across paths.
  {
    Substitutor early(resolved, resolved.variables);
    std::map<std::string, std::string> expanded;
    for (const auto &[key, value] : resolved.variables)
      expanded[key] = early(value);
    resolved.variables = std::move(expanded);
  }
  Substitutor substitute(resolved, resolved.variables);
  if (!context.prompt_file.empty())
    resolved.prompt_file = std::filesystem::absolute(context.prompt_file);
  else if (!recipe.prompt_file.empty())
    resolved.prompt_file =
        std::filesystem::absolute(substitute(recipe.prompt_file));
  for (const auto &input : recipe.inputs)
    resolved.inputs.push_back({input.name, substitute(input.path)});
  resolved.output = std::filesystem::path(substitute(recipe.output));
  if (!resolved.output.is_absolute())
    resolved.output = resolved.work_directory / resolved.output;
  for (const auto &file : recipe.model_files)
    resolved.model_files.emplace_back(substitute(file));
  for (const auto &file : recipe.required_files)
    resolved.required_files.emplace_back(substitute(file));
  std::map<std::string, std::size_t> index_of;
  for (std::size_t index = 0; index < recipe.stages.size(); ++index) {
    const auto &stage = recipe.stages[index];
    if (index_of.contains(stage.name))
      fail("benchmark recipe repeats stage name '" + stage.name + "'");
    index_of.emplace(stage.name, index);
  }
  for (const auto &stage : recipe.stages) {
    ResolvedStage out;
    out.name = stage.name;
    for (const auto &argument : stage.argv)
      out.argv.push_back(substitute(argument));
    for (const auto &dependency : stage.after) {
      const auto found = index_of.find(dependency);
      if (found == index_of.end())
        fail("benchmark recipe stage '" + stage.name +
             "' depends on unknown stage '" + dependency + "'");
      if (found->second >= index_of.at(stage.name))
        fail("benchmark recipe stage '" + stage.name +
             "' may only depend on earlier stages");
      out.after.push_back(found->second);
    }
    if (!stage.cwd.empty())
      out.cwd = std::filesystem::path(substitute(stage.cwd));
    for (const auto &[key, value] : recipe.environment)
      out.environment.emplace_back(key, substitute(value));
    for (const auto &[key, value] : stage.environment)
      out.environment.emplace_back(key, substitute(value));
    out.cacheable = stage.cacheable;
    for (const auto &file : stage.cache.key)
      out.cache_key_files.emplace_back(substitute(file));
    for (const auto &file : stage.cache.outputs) {
      std::filesystem::path output(substitute(file));
      if (!output.is_absolute())
        output = resolved.work_directory / output;
      out.cache_outputs.push_back(std::move(output));
    }
    resolved.stages.push_back(std::move(out));
  }
  return resolved;
}

std::vector<PreflightProblem> preflight(const ResolvedRecipe &resolved) {
  std::vector<PreflightProblem> problems;
  std::set<std::string> seen;
  const auto check_file = [&](const std::filesystem::path &path) {
    if (path.empty() || !seen.insert(path.string()).second)
      return;
    if (!std::filesystem::exists(path))
      problems.push_back({"missing-file", path.string()});
  };
  check_file(resolved.prompt_file);
  for (const auto &input : resolved.inputs)
    check_file(input.path);
  for (const auto &file : resolved.model_files)
    check_file(file);
  for (const auto &file : resolved.required_files)
    check_file(file);
  const auto in_path = [](const std::string &tool) {
    if (tool.find('/') != std::string::npos)
      return std::filesystem::exists(tool);
    const char *path = std::getenv("PATH");
    if (!path)
      return false;
    std::stringstream entries(path);
    std::string entry;
    while (std::getline(entries, entry, ':'))
      if (!entry.empty() && std::filesystem::exists(
                                std::filesystem::path(entry) / tool))
        return true;
    return false;
  };
  for (const auto &tool : resolved.recipe.required_tools)
    if (!in_path(tool))
      problems.push_back({"missing-tool", tool});
  for (const auto &stage : resolved.stages) {
    const auto &program = stage.argv.front();
    if (!in_path(program))
      problems.push_back({"missing-binary", stage.name + ": " + program});
  }
  return problems;
}

} // namespace dif::bench
