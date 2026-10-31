"""Validate and render the canonical mathematical capability inventory.

This is a documentation/coverage tool, not a mathematical implementation or a
capability test. --check also verifies the committed readable view is current.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
STATUSES = {"existing", "partial", "planned", "in_progress", "validated"}
EXPECTED_FAMILIES = {
    "CALC", "JET", "RESP", "INV", "OPT", "GEN", "DIST", "PAIR", "LIN",
    "TENSOR", "KERNEL", "SPEC", "MULTI", "SYM", "DYN", "BOOL", "LATTICE",
    "QUANT", "COUNT", "FIELD", "GEOM", "DISC", "PRE", "ENV", "GRAPH",
    "PROV", "ATTR", "PROB", "DELTA", "UNC", "META", "TOPO",
}


def validate(data, root=ROOT):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported inventory schema version")
    families = data["families"]
    family_ids = [family["id"] for family in families]
    if len(family_ids) != len(set(family_ids)) or set(family_ids) != EXPECTED_FAMILIES:
        raise ValueError("Inventory must cover exactly the 32 requested mathematical families")
    capabilities = data["capabilities"]
    ids = [capability["id"] for capability in capabilities]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate capability ID")
    by_id = {capability["id"]: capability for capability in capabilities}
    waves = [wave["id"] for wave in data["waves"]]
    wave_order = {name: index for index, name in enumerate(waves)}
    if len(waves) != len(set(waves)):
        raise ValueError("Duplicate delivery wave ID")
    coverage = Counter()
    required = {"id", "family_id", "title", "status", "wave", "depends_on",
                "scope", "exactness", "budgets", "cpu", "gpu", "existing_evidence",
                "planned_files", "acceptance_tests", "benchmark_cases", "limitations", "validation"}
    for capability in capabilities:
        name = capability["id"]
        if not required <= set(capability):
            raise ValueError(f"{name}: missing fields {sorted(required - set(capability))}")
        if capability["status"] not in STATUSES or capability["wave"] not in waves:
            raise ValueError(f"{name}: invalid status or delivery wave")
        family = capability["family_id"]
        if family is not None:
            if family not in EXPECTED_FAMILIES:
                raise ValueError(f"{name}: unknown family {family}")
            coverage[family] += 1
        if len(capability["acceptance_tests"]) < 2 or len(capability["benchmark_cases"]) < 2:
            raise ValueError(f"{name}: concrete correctness and benchmark cases required")
        if not capability["budgets"] or not capability["scope"] or not capability["exactness"]:
            raise ValueError(f"{name}: domain, exactness and resource boundaries are required")
        for dependency in capability["depends_on"]:
            if dependency not in by_id or dependency == name:
                raise ValueError(f"{name}: invalid dependency {dependency}")
            if wave_order[by_id[dependency]["wave"]] > wave_order[capability["wave"]]:
                raise ValueError(f"{name}: dependency {dependency} is scheduled in a later wave")
        if capability["status"] in {"existing", "partial"} and not capability["existing_evidence"]:
            raise ValueError(f"{name}: an existing/partial claim requires source evidence")
        for evidence in capability["existing_evidence"]:
            path = Path(evidence["path"])
            if path.is_absolute() or ".." in path.parts or not (root/path).is_file():
                raise ValueError(f"{name}: missing or nonlocal existing evidence {path}")
        if capability["status"] == "validated":
            validation = capability["validation"]
            if not validation or not validation.get("tests") or not validation.get("benchmarks"):
                raise ValueError(f"{name}: validated claims require test and benchmark receipts")
            for path in validation["tests"] + validation["benchmarks"]:
                if not (root/path).is_file():
                    raise ValueError(f"{name}: missing validation receipt {path}")
            if not validation.get("source_sha256") or not validation.get("measured_runtime_commit"):
                raise ValueError(f"{name}: validated claims require measured source identity")
            for path, expected in validation["source_sha256"].items():
                source=root/path
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"{name}: current source differs from validated receipt: {path}")
            for path in capability["planned_files"]:
                if not (root/path).is_file():
                    raise ValueError(f"{name}: missing implemented file {path}")
            cuda = validation.get("cuda")
            if cuda is None and not validation.get("cuda_standard"):
                raise ValueError(f"{name}: validated claims require an actual-device CUDA receipt (validation.cuda)")
            if cuda is not None:
                if not cuda.get("kernels") or not cuda.get("device_receipts"):
                    raise ValueError(f"{name}: validation.cuda needs its kernel files and actual-device receipts")
                for path in cuda["kernels"] + cuda["device_receipts"] + cuda.get("compile_receipts", []) + cuda.get("emulated_receipts", []):
                    if not (root/path).is_file():
                        raise ValueError(f"{name}: missing CUDA file or receipt {path}")
    if set(coverage) != EXPECTED_FAMILIES:
        raise ValueError("A requested family has no actionable capability")
    state = {}

    def visit(name):
        if state.get(name) == 1:
            raise ValueError(f"Dependency cycle reaches {name}")
        if state.get(name) == 2:
            return
        state[name] = 1
        for dependency in by_id[name]["depends_on"]:
            visit(dependency)
        state[name] = 2

    for name in by_id:
        visit(name)
    for name in data["first_slice"]["capability_ids"]:
        if name not in by_id:
            raise ValueError(f"First slice references unknown capability {name}")
    return dict(families=len(families), capabilities=len(capabilities),
                foundations=sum(c["family_id"] is None for c in capabilities),
                statuses=dict(sorted(Counter(c["status"] for c in capabilities).items())))


def bullet_lines(values):
    return [f"- {value}" for value in values]


def capability_link(name):
    return f"[{name}](#{name.lower()})"


def render_capability(capability):
    c = capability
    lines = [f"<a id=\"{c['id'].lower()}\"></a>", "", f"### {c['id']}: {c['title']}", "",
             f"Status: **{c['status']}**. Delivery wave: **{c['wave']}**.", "",
             c["scope"], "", f"Exactness: {c['exactness']}", "",
             "Dependencies: " + (", ".join(capability_link(name) for name in c["depends_on"]) or "None.") , "",
             f"Native CPU: {c['cpu']}", "", f"GPU: {c['gpu']}", "",
             "Bounds:", "", *bullet_lines(c["budgets"]), ""]
    if c["existing_evidence"]:
        lines += ["Existing evidence (scoped building blocks, not a completion claim for extensions):", ""]
        for evidence in c["existing_evidence"]:
            path = evidence["path"]
            lines.append(f"- [{path}]({path}): {evidence['scope']}")
        lines.append("")
    if c["planned_files"]:
        if c["status"] == "validated":
            lines += ["Implemented files:", "",
                      *bullet_lines([f"[{path}]({path})" for path in c["planned_files"]]), ""]
        else:
            lines += ["Implementation targets (proposed until the capability is validated):", "",
                      *bullet_lines([f"`{path}`" for path in c["planned_files"]]), ""]
    lines += ["Acceptance tests:", "", *bullet_lines(c["acceptance_tests"]), "",
              "Benchmark cases:", "", *bullet_lines(c["benchmark_cases"]), ""]
    if c["limitations"]:
        lines += ["Additional limits:", "", *bullet_lines(c["limitations"]), ""]
    if c["validation"]:
        lines += ["Validation receipts:", ""]
        if c["validation"].get("measured_runtime_commit"):
            lines += [f"Measured runtime: `{c['validation']['measured_runtime_commit']}`.", ""]
        if c["validation"].get("notes"):
            lines += [c["validation"]["notes"], ""]
        for kind, paths in c["validation"].items():
            if isinstance(paths, list):
                lines += [f"- {kind}: [{path}]({path})" for path in paths]
        lines.append("")
    return lines


def render(data, summary):
    lines = [f"# {data['title']}", "",
             "Generated from [docs/algebra_inventory.json](docs/algebra_inventory.json). "
             "Edit that canonical file and run `python bench/render_algebra_inventory.py`; "
             "use `--check` to validate coverage, dependencies, evidence and this rendered view.", "",
             f"Inventory date: {data['created_date']}. Source baseline: `{data['baseline_commit']}`. "
             f"Working branch: `{data['workspace_branch']}`.", "", data["source_scope"], "", data["requested_scope"], "",
             f"Coverage: **{summary['families']} families, {summary['capabilities']} actionable capabilities, "
             f"including {summary['foundations']} shared foundations**.", "",
             "| Status | Capabilities |", "|---|---:|"]
    for status in ("existing", "partial", "planned", "in_progress", "validated"):
        lines.append(f"| {status} | {summary['statuses'].get(status,0)} |")
    lines += ["", "A planned target or acceptance case is not an implementation or a passed test. "
              "Prior native-stage benchmarks cover their recorded workloads only. "
              "A capability becomes validated only within its stated bounds and with linked test and benchmark receipts.", "",
              "## Status meanings", ""]
    lines += [f"- **{status}**: {meaning}" for status, meaning in data["status_definitions"].items()]
    lines += ["", "## Mathematical contracts", "", *bullet_lines(data["mathematical_contracts"]), "",
              "## Benchmark contracts", "", *bullet_lines(data.get("benchmark_contracts", [])), "",
              "## First implementation slice", "", f"Status: **{data['first_slice']['status']}**.", "",
              data["first_slice"]["scope"], "",
              "Capabilities: " + ", ".join(capability_link(name) for name in data["first_slice"]["capability_ids"]), "",
              f"API guide: [{data['first_slice']['api_guide']}]({data['first_slice']['api_guide']})", "",
              "Backend semantics:", "", *bullet_lines(data["first_slice"].get("backend_semantics",[])), "",
              "Explicit exclusions:", "", *bullet_lines(data["first_slice"]["exclusions"]), "",
              "## Delivery waves", ""]
    first_validation=data["first_slice"].get("validation")
    if first_validation:
        at=lines.index("## Delivery waves")
        receipt_lines=["First-slice validation:", "", first_validation["notes"], "",
                       f"Measured runtime: `{first_validation['measured_runtime_commit']}`.", ""]
        for kind in ("tests","benchmarks"):
            receipt_lines += [f"- {kind}: [{path}]({path})" for path in first_validation[kind]]
        integrity=data["first_slice"].get("source_integrity_receipt")
        if integrity:
            receipt_lines.append(f"- Source integrity: [{integrity}]({integrity})")
        receipt_lines.append("")
        lines[at:at]=receipt_lines
    current=data.get("current_slice")
    if current:
        at=lines.index("## Delivery waves")
        lines[at:at]=["## Current continuation slice", "", current["name"], "",
                      f"Status: **{current['status']}**.", "", current["scope"], "",
                      "Capabilities: " + ", ".join(capability_link(name) for name in current["capability_ids"]), "",
                      "Remaining work:", "", *bullet_lines(current.get("remaining",[])), ""]
    for wave in data["waves"]:
        lines += [f"### {wave['id']}: {wave['name']}", "", *bullet_lines(wave["deliverables"]), ""]
    lines += ["## Family coverage", "", "| Family | Scope | Capability IDs |", "|---|---|---|"]
    for family in data["families"]:
        ids = ", ".join(capability_link(c["id"]) for c in data["capabilities"] if c["family_id"] == family["id"])
        lines.append(f"| {family['id']}: {family['name']} | {family['scope']} | {ids} |")
    lines += ["", "## Shared native foundations", ""]
    for capability in data["capabilities"]:
        if capability["family_id"] is None:
            lines += render_capability(capability)
    for family in data["families"]:
        lines += [f"## {family['id']}: {family['name']}", "", family["scope"], ""]
        for capability in data["capabilities"]:
            if capability["family_id"] == family["id"]:
                lines += render_capability(capability)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"docs/algebra_inventory.json")
    parser.add_argument("--output", type=Path, default=ROOT/"ALGEBRA_INVENTORY.md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    summary = validate(data)
    rendered = render(data, summary)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Readable inventory is absent or stale; rerun without --check")
    else:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps(dict(**summary, dependencies_acyclic=True, existing_evidence_present=True,
                          rendered_view_current=True, output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
